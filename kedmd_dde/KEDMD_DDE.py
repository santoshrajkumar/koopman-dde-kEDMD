"""kEDMD on delay-embedded states: kernels, center placement, and the estimator.

    1. pick p kernel centers C in the lifted state space
    2. at each center c, fit y ~ a + B (x - c) to the snapshots in the ball B(c, rho);
       the intercept a estimates the flow map at the center, Fhat(c)
    3. A = K_FX^T (K_X + reg I)^+, from K_X = k(C, C) and K_FX = k(Fhat(C), C)

Prediction lifts z0 to psi = k(C, z0), iterates psi <- A psi, and decodes with
W^T = C^T (K_X + reg I)^+.

Centers are processed in blocks so their distances to the training cloud are one big
matrix op, and the local regressions are grouped by neighbourhood size so each group is
one batched matrix_rank/lstsq call instead of p small ones.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .Utils.device import as_tensor, get_device, get_dtype


# ----------------------------------------------------------------------- kernels

DEFAULT_CHUNK = 8192


def wendland_degree(dim, s=1):
    """Degree of the Wendland polynomial P_{d,s}: floor(dim/2) + 3 s + 1.

    dim is the dimension of the space the kernel must be positive definite on, which
    here is the lifted dimension nM. The degree has to grow with it: a Wendland function
    built for dimension d is positive definite on R^d and on every smaller space, but
    not in general on a larger one, and strict positive definiteness is what makes K_X
    invertible and the RKHS well defined.
    """
    return int(dim) // 2 + 3 * int(s) + 1


def wendland_c2(r_dist, epsilon, dim):
    """Wendland C^2 kernel on R^dim, phi(r) = (1-r/eps)_+^n (n r/eps + 1).

    The s = 1 member of the family, so n = wendland_degree(dim) - 1 = floor(dim/2) + 3.
    For dim in {2, 3} this is the familiar (1-r)^4 (4r+1); the planar runs lift to
    dim = 4 and 6 and need n = 5 and 6.
    """
    n = wendland_degree(dim) - 1
    r = as_tensor(r_dist) / epsilon
    val = torch.zeros_like(r)
    mask = r < 1.0
    rm = r[mask]
    val[mask] = ((1.0 - rm) ** n) * (n * rm + 1.0)
    return val


def pairwise_distances(X1, X2, chunk=None):
    """Euclidean distances between every row of X1 (N1, d) and X2 (N2, d).

    chunk bounds peak memory at chunk * N2 * d; pass 0 to disable chunking.
    """
    X1, X2 = as_tensor(X1), as_tensor(X2)
    chunk = DEFAULT_CHUNK if chunk is None else chunk

    # donot_use_mm_for_euclid_dist: the |x|^2 - 2x.y + |y|^2 identity loses precision
    # for nearby points, which is exactly what the local neighbourhoods are made of.
    if not chunk or X1.shape[0] <= chunk:
        return torch.cdist(X1, X2, compute_mode="donot_use_mm_for_euclid_dist")

    out = torch.empty((X1.shape[0], X2.shape[0]), device=X1.device, dtype=X1.dtype)
    for start in range(0, X1.shape[0], chunk):
        stop = start + chunk
        out[start:stop] = torch.cdist(
            X1[start:stop], X2, compute_mode="donot_use_mm_for_euclid_dist"
        )
    return out


def wendland_matrix(X1, X2, epsilon, chunk=None, dim=None):
    """Kernel matrix with [i, j] = k(X1_i, X2_j).

    dim defaults to the width of the points themselves, so the kernel degree follows the
    lifted dimension and a model cannot be fitted with one degree and evaluated with
    another.
    """
    X1 = as_tensor(X1)
    dim = X1.shape[1] if dim is None else dim
    return wendland_c2(pairwise_distances(X1, X2, chunk=chunk), epsilon, dim)


# --------------------------------------------------------------- kernel centers

def first_argmax(v):
    # torch.argmax does not fix a tie-break; the greedy sampler needs one to be repeatable
    return int(torch.nonzero(v == v.max(), as_tuple=False)[0].item())


def first_argmin(v):
    return int(torch.nonzero(v == v.min(), as_tuple=False)[0].item())


def farthest_point_sampling(X, n_centers, seed=0, include_origin_like=True,
                            return_fill_distances=False):
    """Greedily pick n_centers well-separated rows of X.

    With return_fill_distances, also returns h[p-1] = fill distance of the first p
    centers, which the running min_dists vector already holds.
    """
    X = as_tensor(X)
    N = X.shape[0]

    if n_centers >= N and not return_fill_distances:
        return X.clone()
    n_take = min(int(n_centers), N)

    if include_origin_like:
        first_idx = first_argmin(torch.linalg.norm(X, dim=1))
    else:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        first_idx = int(torch.randint(N, (1,), generator=gen).item())

    centers_idx = [first_idx]
    min_dists = torch.linalg.norm(X - X[first_idx], dim=1)
    fills = [min_dists.max()] if return_fill_distances else None

    for _ in range(1, n_take):
        next_idx = first_argmax(min_dists)
        centers_idx.append(next_idx)
        torch.minimum(min_dists, torch.linalg.norm(X - X[next_idx], dim=1), out=min_dists)
        if return_fill_distances:
            fills.append(min_dists.max())

    idx = torch.as_tensor(centers_idx, device=X.device, dtype=torch.long)
    if return_fill_distances:
        return X[idx], torch.stack(fills)
    return X[idx]


def uniform_grid_centers_box(n_centers, mins, maxs):
    """Tensor-product grid of n_centers points over [mins, maxs]. Lifted dimension 2 only."""
    side = int(n_centers ** 0.5)
    if side * side != n_centers:
        raise ValueError(f"n_centers={n_centers} is not a perfect square")

    mins, maxs = as_tensor(mins), as_tensor(maxs)
    if mins.shape != (2,) or maxs.shape != (2,):
        raise ValueError("uniform_grid_centers_box is written for a lifted dimension of 2")

    device, dtype = get_device(), get_dtype()
    x1 = torch.linspace(float(mins[0]), float(maxs[0]), side, device=device, dtype=dtype)
    x2 = torch.linspace(float(mins[1]), float(maxs[1]), side, device=device, dtype=dtype)
    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")     # 'xy' matches numpy's meshgrid
    return torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1), side


def empirical_fill_distance(X, centers, chunk=4096):
    """h = max over x in X of min over c in centers of ||x - c||."""
    X, centers = as_tensor(X), as_tensor(centers)

    worst = 0.0
    for start in range(0, X.shape[0], chunk):
        dist = pairwise_distances(X[start:start + chunk], centers, chunk=0)
        worst = max(worst, float(dist.min(dim=1).values.max()))
    return worst


def fill_distance_curve(X, max_centers, seed=0, include_origin_like=True):
    """Fill distance for every p = 1 .. max_centers, from one greedy sweep.

    Farthest-point sampling is nested: the first p points it picks are the p-center set
    it would have produced on its own, so one sweep answers every smaller budget.
    """
    _, fills = farthest_point_sampling(X, max_centers, seed, include_origin_like,
                                       return_fill_distances=True)
    return fills


def fill_distance_at(X, n_centers, seed=0, include_origin_like=True):
    """Fill distance that n_centers farthest-point samples achieve on X."""
    return float(fill_distance_curve(X, n_centers, seed, include_origin_like)[-1])


def min_centers_for_fill_distance(X, target, max_centers=1024, seed=0,
                                  include_origin_like=True):
    """Fewest centers reaching a fill distance of at most target. Returns (p, h)."""
    if not target > 0:
        raise ValueError(f"target fill distance must be positive, got {target}")

    curve = fill_distance_curve(X, max_centers, seed, include_origin_like)
    reached = torch.nonzero(curve <= target, as_tuple=False)   # curve is non-increasing
    if reached.numel() == 0:
        raise ValueError(
            f"No center count up to {max_centers} reaches a fill distance of {target:g}; "
            f"the best is {float(curve[-1]):g}. Raise max_centers."
        )

    p = int(reached[0].item()) + 1
    return p, float(curve[p - 1])


def matched_center_configs(datasets, target, base_cfg=None, max_centers=1024, seed=0,
                           verbose=True):
    """Per-M configs using the fewest centers that all reach the same fill distance.

    The lifted space gains a dimension per delay coordinate, so a fixed center count
    covers it more thinly as M rises; matching h instead compares the lift alone.

    datasets maps M -> DelayDataset (or M -> X). target is a fill distance, or an
    (M, n_centers) pair to take one from.
    """
    clouds = {M: getattr(ds, "X", ds) for M, ds in datasets.items()}

    if isinstance(target, (tuple, list)):
        ref_M, ref_p = target
        if ref_M not in clouds:
            raise KeyError(f"reference M={ref_M} is not among {sorted(clouds)}")
        target = fill_distance_at(clouds[ref_M], ref_p, seed)
        if verbose:
            print(f"[centers] target h = {target:.6e} (M={ref_M} with p={ref_p})")
    elif verbose:
        print(f"[centers] target h = {float(target):.6e}")

    out = {}
    for M in sorted(clouds):
        p, h = min_centers_for_fill_distance(clouds[M], target, max_centers, seed)
        cfg = dict(base_cfg or {})
        cfg["n_centers"] = p
        cfg["fill_distance"] = h
        out[M] = cfg
        if verbose:
            print(f"[centers]   M={M}: n_centers={p}, h={h:.6e}")
    return out


# ------------------------------------------------------------------- estimator

DEFAULT_DECODER_REG = 1e-8

# numpy.linalg.pinv truncates at 1e-15 * largest singular value; torch defaults to
# max(m, n) * eps, which would discard extra singular values of the near-singular
# kernel matrix. Pinned so results match the reference implementation.
PINV_RTOL = 1e-15

DEFAULT_CENTER_BLOCK = 256          # cap; the real block comes from auto_center_block
DISTANCE_MEMORY_FRACTION = 0.2      # of free device memory, per distance block
CPU_MEMORY_BUDGET = 2 << 30         # assumed available when sizing a block on the CPU
BLOCK_OVERHEAD = 1.6                # distances plus the radius mask and topk workspace


def _free_bytes(device):
    device = torch.device(device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        free, _ = torch.cuda.mem_get_info(idx)
        return free
    return CPU_MEMORY_BUDGET


def auto_center_block(n_train, itemsize, device, cap=DEFAULT_CENTER_BLOCK):
    """How many centers can share one (block, N) distance matrix.

    Sized from free memory, so a large training set shrinks the block instead of
    overflowing it.
    """
    budget = _free_bytes(device) * DISTANCE_MEMORY_FRACTION
    per_center = max(1.0, n_train * itemsize * BLOCK_OVERHEAD)
    return int(min(cap, max(1, budget // per_center)))


def auto_solve_batch(k, nM, itemsize, device):
    """How many k-point local regressions fit in one batched call.

    Each center holds its neighbourhood twice (states and successors) plus the design
    matrix, so the cost is k * (3 nM + 1) per center.
    """
    budget = _free_bytes(device) * DISTANCE_MEMORY_FRACTION
    per_center = max(1.0, k * (3 * nM + 1) * itemsize * BLOCK_OVERHEAD)
    return int(max(1, budget // per_center))


def _pinv(M, rtol=PINV_RTOL):
    return torch.linalg.pinv(M, rtol=rtol)


def _lstsq(Z_bar, Y_local):
    """Least-squares solve, batched over the leading dimension if present."""
    # Z_bar is full rank here, so the solution is unique and the driver is immaterial;
    # CUDA only offers gels, and gelsd on the CPU mirrors numpy.
    driver = "gels" if Z_bar.is_cuda else "gelsd"
    return torch.linalg.lstsq(Z_bar, Y_local, driver=driver).solution


@dataclass
class KEDMDModel:
    """A fitted model plus the diagnostics collected while fitting it."""

    centers: torch.Tensor
    epsilon: float
    A: torch.Tensor
    Fhat_centers: torch.Tensor
    K_X: torch.Tensor
    K_X_inv: torch.Tensor
    info: dict
    decoder_reg: float = DEFAULT_DECODER_REG
    assumption3_report: "Assumption3Report" = field(default=None, repr=False, compare=False)
    _decoder: torch.Tensor = field(default=None, repr=False, compare=False)

    @property
    def device(self):
        return self.centers.device

    @property
    def n_centers(self):
        return self.centers.shape[0]

    @property
    def lifted_dim(self):
        return self.centers.shape[1]

    @property
    def decoder(self):
        """W^T mapping kernel features back to the lifted state, (nM, p)."""
        if self._decoder is None:
            eye = torch.eye(self.K_X.shape[0], device=self.K_X.device, dtype=self.K_X.dtype)
            self._decoder = self.centers.T @ _pinv(self.K_X + self.decoder_reg * eye)
        return self._decoder

    def lift(self, z):
        """Kernel features: (nM,) gives (p,), and (batch, nM) gives (p, batch)."""
        z = as_tensor(z)
        if z.ndim == 1:
            return wendland_matrix(self.centers, z[None, :], self.epsilon).reshape(-1)
        return wendland_matrix(self.centers, z, self.epsilon)

    def rollout(self, z0, n_steps):
        """Autonomous prediction from one z0, (n_steps, nM). Row 0 is the decoded z0."""
        return self.rollout_many(as_tensor(z0)[None, :], n_steps)[0]

    def rollout_many(self, z0s, n_steps):
        """Autonomous prediction from (batch, nM) initial states, (batch, n_steps, nM).

        All trajectories advance together, so this is n_steps matrix products rather
        than n_steps * batch matrix-vector products.
        """
        z0s = as_tensor(z0s)
        W_T = self.decoder
        psi = self.lift(z0s)                               # (p, batch)

        traj = torch.empty(
            (n_steps, psi.shape[1], W_T.shape[0]), device=psi.device, dtype=psi.dtype
        )
        for k in range(n_steps):
            traj[k] = (W_T @ psi).T
            psi = self.A @ psi

        return traj.permute(1, 0, 2).contiguous()


def _select_neighbourhoods(
    X, centers, rho, d_neighbors, min_required, min_local_points, center_block
):
    """Index sets for every center's local fit, plus per-center bookkeeping.

    More than d_neighbors points in B(c, rho) -> the d_neighbors nearest, by distance.
    Otherwise every point in the ball, by index. Fewer than nM + 1 either way -> the
    min_local_points nearest overall, counted as a fallback.

    Returns (local_idx, stats) with stats holding the aggregate counts plus the
    per-center n_in_radius and is_fallback vectors the Assumption 3 report reads.
    """
    N = X.shape[0]
    p = centers.shape[0]

    local_idx = [None] * p
    n_in_radius = torch.zeros(p, dtype=torch.long, device=X.device)
    is_fallback = torch.zeros(p, dtype=torch.bool, device=X.device)
    used_fallback = 0
    total_in_radius = 0
    total_used = 0

    for start in range(0, p, center_block):
        blk = centers[start:start + center_block]
        D = pairwise_distances(blk, X, chunk=0)          # (B, N)
        in_radius = D <= rho
        counts = in_radius.sum(dim=1)
        n_in_radius[start:start + counts.shape[0]] = counts
        # One host transfer per block; the branch below needs the counts on the CPU.
        n_in = counts.cpu().tolist()

        topk_rows = defaultdict(list)                    # k -> rows needing k nearest
        radius_rows = []

        for r, n in enumerate(n_in):
            total_in_radius += n

            if d_neighbors is not None and n > d_neighbors:
                k_sel, mode = d_neighbors, "topk"
            else:
                k_sel, mode = n, "radius"

            if k_sel < min_required:
                k_sel, mode = min(min_local_points, N), "topk"
                used_fallback += 1
                is_fallback[start + r] = True

            total_used += k_sel
            (topk_rows[k_sel] if mode == "topk" else radius_rows).append(r)

        for k, rows in topk_rows.items():
            sel = torch.as_tensor(rows, device=D.device)
            idx = D[sel].topk(k, dim=1, largest=False, sorted=True).indices
            for j, r in enumerate(rows):
                local_idx[start + r] = idx[j]

        for r in radius_rows:
            local_idx[start + r] = in_radius[r].nonzero(as_tuple=False).reshape(-1)

        del D, in_radius

    stats = {
        "used_fallback": used_fallback,
        "total_in_radius": total_in_radius,
        "total_used": total_used,
        "n_in_radius": n_in_radius,
        "is_fallback": is_fallback,
    }
    return local_idx, stats


def _repair_rank_deficient(
    X, Y, c, min_required, min_local_points, rank_recovery, rank_fallback_count
):
    """Re-pick a neighbourhood for a center whose design matrix was rank deficient.

    Returns (Z_bar, Y_local), or (None, None) if no full-rank set was found.
    """
    N = X.shape[0]
    distances = torch.linalg.norm(X - c, dim=1)

    def design(idx):
        X_local = X[idx]
        return torch.cat(
            [torch.ones((len(idx), 1), device=X.device, dtype=X.dtype), X_local - c],
            dim=1,
        ), Y[idx]

    if rank_recovery == "multiplier":
        idx = distances.topk(min(N, rank_fallback_count), largest=False,
                             sorted=True).indices
        Z_bar, Y_local = design(idx)
        if int(torch.linalg.matrix_rank(Z_bar)) >= min_required:
            return Z_bar, Y_local
        return None, None

    # "expand": grow the nearest-neighbour set one point at a time.
    order = torch.argsort(distances)
    for k in range(min_local_points, N + 1):
        Z_bar, Y_local = design(order[:k])
        if int(torch.linalg.matrix_rank(Z_bar)) >= min_required:
            return Z_bar, Y_local
    return None, None


# --------------------------------------------------------- Assumption 3 check

def _rank_tol(sigma, m, n):
    """torch.linalg.matrix_rank's default cutoff, max(m, n) * eps * sigma_max."""
    eps = torch.finfo(sigma.dtype).eps
    return max(m, n) * eps * sigma[..., :1]


@dataclass
class Assumption3Report:
    """Per-center evidence for rank(Z_bar_ell) = nM + 1 at a given (rho, d).

    Two neighbourhoods are scored for every center. The *strict* one is the paper's:
    the d nearest training points inside B(z_ell, rho) and nothing else, which is what
    the constants Z_bar and rho in Theorem 1 are defined over. The *effective* one is
    what build_kedmd actually fits, which falls back to the min_local_points nearest
    points overall when the ball holds fewer than nM + 1 — those centers satisfy
    Assumption 3 but no longer respect the radius, so their delta_{ell,j} are not
    bounded by rho and the R_d rho^2 term of (21) does not cover them.

    All tensors are indexed by center and live on the package device.
    """

    rho: float
    d_neighbors: int
    lifted_dim: int                 # nM
    min_required: int               # nM + 1
    n_train: int
    min_local_points: int
    n_in_radius: torch.Tensor       # |B(z_ell, rho) ∩ X|
    n_used: torch.Tensor            # d_ell, the size of the effective neighbourhood
    is_fallback: torch.Tensor       # effective neighbourhood left the ball
    rank: torch.Tensor              # rank(Z_bar_ell) on the effective neighbourhood
    sigma_min: torch.Tensor         # smallest singular value of Z_bar_ell
    sigma_max: torch.Tensor
    pinv_norm: torch.Tensor         # ||Z_bar_ell^dagger|| = 1 / sigma_min, inf if deficient
    max_offset: torch.Tensor        # max_j ||z_{ell,j} - z_ell||, the realised radius

    # ------------------------------------------------------------- verdicts

    @property
    def full_rank(self):
        """Assumption 3 on the neighbourhood build_kedmd fits."""
        return self.rank >= self.min_required

    @property
    def strict_ok(self):
        """Assumption 3 on the paper's neighbourhood: full rank without leaving B(c, rho)."""
        return self.full_rank & ~self.is_fallback

    @property
    def holds(self):
        return bool(self.strict_ok.all())

    @property
    def holds_effective(self):
        return bool(self.full_rank.all())

    @property
    def n_centers(self):
        return int(self.rank.shape[0])

    @property
    def n_underfilled(self):
        """Centers whose ball cannot supply nM + 1 points, so d_ell < nM + 1."""
        return int((self.n_in_radius < self.min_required).sum())

    @property
    def n_fallback(self):
        return int(self.is_fallback.sum())

    @property
    def n_deficient(self):
        """Centers still rank deficient after the fallback."""
        return int((~self.full_rank).sum())

    # ------------------------------------------------------- bound constants

    @property
    def Z_bar(self):
        """max_ell ||Z_bar_ell^dagger||, the constant Z_bar of (22c)-(22d)."""
        finite = self.pinv_norm[torch.isfinite(self.pinv_norm)]
        return float(finite.max()) if finite.numel() else float("inf")

    @property
    def Z_bar_strict(self):
        """Z_bar over the centers that satisfy Assumption 3 inside the radius."""
        ok = self.strict_ok
        vals = self.pinv_norm[ok]
        vals = vals[torch.isfinite(vals)]
        return float(vals.max()) if vals.numel() else float("inf")

    @property
    def rho_eff(self):
        """max_ell max_j ||delta_{ell,j}||: the radius the R_d rho^2 term really needs.

        Equal to at most rho when no center falls back, and larger otherwise.
        """
        return float(self.max_offset.max())

    @property
    def d_eff(self):
        """max_ell d_ell, the d entering the sqrt(d) factor of (22c)-(22d)."""
        return int(self.n_used.max())

    @property
    def regression_scale(self):
        """sqrt(d) * Z_bar * rho^2: the part of R_d rho^2 that (rho, d) control.

        Only the (rho, d)-dependent factors; C_0, L_k, p, D_F, ||Psi||_H and ||K_X^-1||
        are common to every configuration and cancel when two are compared.
        """
        return float(self.d_eff ** 0.5) * self.Z_bar * self.rho_eff ** 2

    # ------------------------------------------------------------- reporting

    def summary(self):
        """Plain-dict form, suitable for storing in a model's info."""
        return {
            "holds": self.holds,
            "holds_effective": self.holds_effective,
            "rho": self.rho,
            "d_neighbors": self.d_neighbors,
            "min_required": self.min_required,
            "n_centers": self.n_centers,
            "n_underfilled": self.n_underfilled,
            "n_fallback": self.n_fallback,
            "n_deficient": self.n_deficient,
            "min_points_in_radius": int(self.n_in_radius.min()),
            "median_points_in_radius": int(self.n_in_radius.median()),
            "max_points_in_radius": int(self.n_in_radius.max()),
            "min_local_samples_used": int(self.n_used.min()),
            "d_eff": self.d_eff,
            "rho_eff": self.rho_eff,
            "Z_bar": self.Z_bar,
            "Z_bar_strict": self.Z_bar_strict,
            "worst_sigma_min": float(self.sigma_min.min()),
            "regression_scale": self.regression_scale,
        }

    def __str__(self):
        s = self.summary()
        verdict = "SATISFIED" if s["holds"] else "VIOLATED"
        lines = [
            f"Assumption 3 for rho = {s['rho']:g}, d = {s['d_neighbors']}, "
            f"nM = {self.lifted_dim} (needs rank {s['min_required']}): {verdict}",
            f"  centers                                p = {s['n_centers']}",
            f"  points in B(c, rho)                      min {s['min_points_in_radius']}, "
            f"median {s['median_points_in_radius']}, max {s['max_points_in_radius']}",
            f"  centers with fewer than nM+1 in ball     = {s['n_underfilled']}",
            f"  centers escaping the ball (fallback)     = {s['n_fallback']}",
            f"  centers rank deficient after fallback    = {s['n_deficient']}",
            f"  smallest sigma_min(Z_bar_l)              = {s['worst_sigma_min']:.3e}",
            f"  Z_bar = max_l ||Z_bar_l^dagger||         = {s['Z_bar']:.3e}"
            + ("" if s["holds"] else f"  (over in-radius centers: {s['Z_bar_strict']:.3e})"),
            f"  realised radius max_l,j ||delta_l,j||    = {s['rho_eff']:.3e}"
            + ("" if s["rho_eff"] <= s["rho"] * (1 + 1e-12) else "   > rho"),
            f"  effective d = max_l d_l                  = {s['d_eff']}",
            f"  sqrt(d) Z_bar rho_eff^2                  = {s['regression_scale']:.3e}",
        ]
        if not s["holds"]:
            lines.append(
                "  -> Theorem 1 does not apply as stated: the R_d rho^2 term assumes "
                "every\n     delta_{l,j} lies within rho. Raise rho, lower p, or add data."
            )
        return "\n".join(lines)


def check_assumption3(
    X,
    centers,
    rho,
    d_neighbors=25,
    min_local_points=None,
    center_block=None,
    verbose=False,
):
    """Test Assumption 3 for the neighbourhoods that (rho, d_neighbors) induce.

    X is the training cloud (N, nM) and centers the set X = {z_l} of (8), so the check
    scores exactly the local data matrices Z_bar_l that build_kedmd would form. Returns
    an Assumption3Report; see its docstring for the strict/effective distinction.

    Rank is decided from the singular values of Z_bar_l with the cutoff
    torch.linalg.matrix_rank uses, so the verdict matches build_kedmd's internal test,
    and sigma_min is reported alongside it because a rank that is nominally full but
    with sigma_min at round-off makes Z_bar = max_l ||Z_bar_l^dagger|| blow up and the
    bound (21) vacuous.
    """
    X = as_tensor(X)
    centers = as_tensor(centers)

    N, nM = X.shape
    p = centers.shape[0]
    min_required = nM + 1

    if min_local_points is None:
        min_local_points = (
            min_required if d_neighbors is None else max(d_neighbors, min_required)
        )
    if center_block is None:
        center_block = auto_center_block(N, X.element_size(), X.device)

    local_idx, stats = _select_neighbourhoods(
        X, centers, rho, d_neighbors, min_required, min_local_points, center_block
    )

    rank = torch.zeros(p, dtype=torch.long, device=X.device)
    sigma_min = torch.zeros(p, dtype=X.dtype, device=X.device)
    sigma_max = torch.zeros(p, dtype=X.dtype, device=X.device)
    max_offset = torch.zeros(p, dtype=X.dtype, device=X.device)
    n_used = torch.zeros(p, dtype=torch.long, device=X.device)

    # Same grouping as build_kedmd: one batched SVD per neighbourhood size.
    by_size = defaultdict(list)
    for i, idx in enumerate(local_idx):
        by_size[idx.shape[0]].append(i)

    for k, rows in by_size.items():
        step = auto_solve_batch(k, nM, X.element_size(), X.device)

        for s in range(0, len(rows), step):
            chunk = rows[s:s + step]
            sel = torch.as_tensor(chunk, device=X.device)
            I = torch.stack([local_idx[i] for i in chunk])          # (B, k)
            C = centers[sel]                                         # (B, nM)
            offsets = X[I] - C[:, None, :]                            # (B, k, nM)

            ones = torch.ones((len(chunk), k, 1), device=X.device, dtype=X.dtype)
            Z_bar = torch.cat([ones, offsets], dim=2)                # (B, k, nM+1)

            sv = torch.linalg.svdvals(Z_bar)                          # (B, min(k, nM+1))
            tol = _rank_tol(sv, k, min_required)

            n_used[sel] = k
            rank[sel] = (sv > tol).sum(dim=1)
            sigma_max[sel] = sv[:, 0]
            # A neighbourhood with k < nM + 1 cannot be full rank; its sigma_min is the
            # smallest of only k values, so record 0 rather than a misleadingly large one.
            sigma_min[sel] = sv[:, -1] if k >= min_required else 0.0
            max_offset[sel] = torch.linalg.norm(offsets, dim=2).max(dim=1).values

            del Z_bar, offsets, I

    pinv_norm = torch.where(
        rank >= min_required,
        1.0 / sigma_min.clamp_min(torch.finfo(X.dtype).tiny),
        torch.full_like(sigma_min, float("inf")),
    )

    report = Assumption3Report(
        rho=float(rho),
        d_neighbors=d_neighbors,
        lifted_dim=nM,
        min_required=min_required,
        n_train=N,
        min_local_points=int(min_local_points),
        n_in_radius=stats["n_in_radius"],
        n_used=n_used,
        is_fallback=stats["is_fallback"],
        rank=rank,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        pinv_norm=pinv_norm,
        max_offset=max_offset,
    )
    if verbose:
        print(report)
    return report


def build_kedmd(
    X,
    Y,
    n_centers=121,
    rho=0.15,
    epsilon=1.0,
    d_neighbors=25,
    reg=1e-8,
    seed=0,
    centers=None,
    rank_recovery="multiplier",
    min_local_points=None,
    compute_fill_distance=True,
    center_block=None,
    assumption3=False,
):
    """Fit a kEDMD model from snapshot pairs X, Y of shape (N, nM).

    rho is the radius of the local neighbourhood; d_neighbors caps how many points in
    the ball are used, or None to use all of them. epsilon is the kernel support
    radius, reg the Tikhonov shift before pseudo-inverting K_X.

    centers overrides the farthest-point sampling, e.g. with a uniform grid.
    rank_recovery is what to do when a local design matrix is rank deficient:
    "multiplier" jumps to max(3 k, nM + 3) nearest points, "expand" grows the set one
    point at a time. Both fall back to the successor of the nearest training point.
    min_local_points is used when the ball holds too few, default
    max(d_neighbors, nM + 1). compute_fill_distance costs an O(N p) sweep.
    assumption3 runs check_assumption3 on the same neighbourhoods and stores the
    verdict in info["assumption3"] and the full report on the model; it costs one
    batched SVD per neighbourhood size.
    """
    if rank_recovery not in ("multiplier", "expand"):
        raise ValueError("rank_recovery must be 'multiplier' or 'expand'")

    get_device()                      # resolve and announce the device on first use
    X = as_tensor(X)
    Y = as_tensor(Y)

    N, nM = X.shape
    min_required = nM + 1

    if centers is None:
        centers = farthest_point_sampling(
            X, n_centers=n_centers, seed=seed, include_origin_like=True
        )
    else:
        centers = as_tensor(centers)
    p = centers.shape[0]

    if min_local_points is None:
        min_local_points = min_required if d_neighbors is None else max(d_neighbors, min_required)

    # Nearest-point count used to repair a rank-deficient design matrix.
    rank_span = min_required if d_neighbors is None else d_neighbors
    rank_fallback_count = max(3 * rank_span, nM + 3)

    if center_block is None:
        center_block = auto_center_block(N, X.element_size(), X.device)

    local_idx, sel_stats = _select_neighbourhoods(
        X, centers, rho, d_neighbors, min_required, min_local_points, center_block
    )
    used_fallback = sel_stats["used_fallback"]
    total_in_radius = sel_stats["total_in_radius"]
    total_used = sel_stats["total_used"]

    Fhat_centers = torch.zeros_like(centers)
    bad_rank_count = 0

    # Group centers by neighbourhood size so each size is one batched solve.
    by_size = defaultdict(list)
    for i, idx in enumerate(local_idx):
        by_size[idx.shape[0]].append(i)

    for k, rows in by_size.items():
        # A batch gathers (B, k, nM) neighbourhoods, so with radius-only selection on a
        # large training set a single group can be tens of GiB. Split it by element
        # count. Each system is still solved independently, but batched and unbatched
        # LAPACK paths round differently, so the batch size moves results at the 1e-12
        # level -- well inside the round-off already separating this from NumPy.
        step = auto_solve_batch(k, nM, X.element_size(), X.device)

        for s in range(0, len(rows), step):
            chunk = rows[s:s + step]
            sel = torch.as_tensor(chunk, device=X.device)
            I = torch.stack([local_idx[i] for i in chunk])         # (B, k)
            C = centers[sel]                                        # (B, nM)
            X_local = X[I]                                          # (B, k, nM)
            Y_local = Y[I]

            ones = torch.ones((len(chunk), k, 1), device=X.device, dtype=X.dtype)
            Z_bar = torch.cat([ones, X_local - C[:, None, :]], dim=2)   # (B, k, nM+1)
            del X_local

            full_rank = torch.linalg.matrix_rank(Z_bar) >= min_required
            if bool(full_rank.all()):
                Fhat_centers[sel] = _lstsq(Z_bar, Y_local)[:, 0, :]
                del Z_bar, Y_local, I
                continue

            ok = full_rank.nonzero(as_tuple=False).reshape(-1)
            if ok.numel():
                Fhat_centers[sel[ok]] = _lstsq(Z_bar[ok], Y_local[ok])[:, 0, :]

            for j in (~full_rank).nonzero(as_tuple=False).reshape(-1).tolist():
                i = chunk[j]
                Z_fix, Y_fix = _repair_rank_deficient(
                    X, Y, centers[i], min_required, min_local_points,
                    rank_recovery, rank_fallback_count,
                )
                if Z_fix is None:
                    nearest = int(torch.argmin(torch.linalg.norm(X - centers[i], dim=1)))
                    Fhat_centers[i] = Y[nearest]
                    bad_rank_count += 1
                else:
                    Fhat_centers[i] = _lstsq(Z_fix, Y_fix)[0, :]
            del Z_bar, Y_local, I

    K_X = wendland_matrix(centers, centers, epsilon)
    K_FX = wendland_matrix(Fhat_centers, centers, epsilon)

    eye = torch.eye(p, device=K_X.device, dtype=K_X.dtype)
    K_X_inv = _pinv(K_X + reg * eye)
    A = K_FX.T @ K_X_inv

    info = {
        "used_fallback": used_fallback,
        "bad_rank_count": bad_rank_count,
        "num_centers": p,
        "rho": rho,
        "epsilon": epsilon,
        "kernel_degree": wendland_degree(nM),
        "d_neighbors": d_neighbors,
        "min_required": min_required,
        "avg_local_samples_in_radius": total_in_radius / p,
        "avg_used_local_samples": total_used / p,
        "device": str(X.device),
        "dtype": str(X.dtype),
    }
    if compute_fill_distance:
        info["fill_distance"] = empirical_fill_distance(X, centers)

    report = None
    if assumption3:
        report = check_assumption3(
            X, centers, rho, d_neighbors=d_neighbors,
            min_local_points=min_local_points, center_block=center_block,
        )
        info["assumption3"] = report.summary()

    return KEDMDModel(
        centers=centers,
        epsilon=epsilon,
        A=A,
        Fhat_centers=Fhat_centers,
        K_X=K_X,
        K_X_inv=K_X_inv,
        info=info,
        assumption3_report=report,
    )


def rollout_lifted(A, centers, epsilon, z0, n_steps, reg=DEFAULT_DECODER_REG):
    """Functional form of KEDMDModel.rollout, rebuilding the decoder every call."""
    A = as_tensor(A)
    centers = as_tensor(centers)

    K_X = wendland_matrix(centers, centers, epsilon)
    eye = torch.eye(K_X.shape[0], device=K_X.device, dtype=K_X.dtype)
    W_T = centers.T @ _pinv(K_X + reg * eye)

    psi = wendland_matrix(centers, as_tensor(z0)[None, :], epsilon).reshape(-1)

    traj = torch.empty((n_steps, W_T.shape[0]), device=psi.device, dtype=psi.dtype)
    for k in range(n_steps):
        traj[k] = W_T @ psi
        psi = A @ psi

    return traj
