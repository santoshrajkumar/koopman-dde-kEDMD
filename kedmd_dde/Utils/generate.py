"""Generate the delay-embedding datasets from the DDEs themselves, using jitcdde.

The Python replacement for the original MATLAB generators, with the same delays,
initial-condition boxes, time span and split structure. For resolution M and delay tau
the lifted state stacks the solution at M points evenly spaced over [t - tau, t],
oldest first, so the newest dim entries are x(t):

    z(t) = [ x(t - tau); x(t - tau + dh); ... ; x(t) ],   dh = tau / (M - 1)

Snapshots start at the first grid time whose whole history window lies inside the span,
which fixes nSnap. jitcdde is imported lazily, so importing kedmd_dde does not need it.
"""

import time as _time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.io import savemat

# Sample times are canonicalised to this many decimals before being matched by value.
TIME_DECIMALS = 9

# Tighter than dde23's 1e-3 / 1e-6; tighter still makes jitcdde fail on the
# initial-history discontinuity.
DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-10


def default_time_grid(t_start=-1.0, t_end=10.0, dt=0.01):
    """The tspan = -1:0.01:10 grid, built to avoid accumulated float error."""
    n = int(round((t_end - t_start) / dt)) + 1
    return np.round(t_start + dt * np.arange(n), TIME_DECIMALS)


@dataclass
class DDESystem:
    """A delay system, its delay, and the box its initial conditions come from."""

    name: str
    dim: int
    tau: float
    box: np.ndarray                    # (dim, 2) lower/upper per component
    rhs: Callable                      # (y, t) -> list of symbolic expressions
    description: str = ""
    pool_growth: int = 2               # pool factor per extra delay coordinate

    def __post_init__(self):
        self.box = np.asarray(self.box, dtype=float).reshape(self.dim, 2)


def _scalar_rhs(y, t, tau=1.0, eta=1.0, n=2.0):
    """Hayes / Mackey-Glass type scalar DDE: y' = eta / (1 + y(t-tau)^n) - y."""
    return [eta / (1 + y(0, t - tau) ** n) - y(0)]


def _planar_rhs(y, t, tau=1.636):
    """Non-dimensionalised delayed tumour-immune model.

    ``u = x1 / 1.636``, ``v = x2 / 20.19``, time rescaled by ``T = 1 / 1.636``.
    """
    u, v = y(0), y(1)
    ud, vd = y(0, t - tau), y(1, t - tau)
    return [
        0.0441 + 0.6913 * ud * vd / (1 + vd) - 0.0384 * ud * vd - 0.2288 * u,
        v * (1 - 0.04038 * v) - u * v,
    ]


# Scalar system, as in the original dde_code.m.
SCALAR_DDE = DDESystem(
    name="scalar",
    dim=1,
    tau=1.0,
    box=[[-1.0, 1.0]],
    rhs=_scalar_rhs,
    description="y' = 1/(1 + y(t-1)^2) - y",
    pool_growth=2,
)

# Planar system, as in the original dde_planar_flat.m.
PLANAR_DDE = DDESystem(
    name="planar",
    dim=2,
    tau=1.636,
    box=[[0.01, 1.5], [0.1, 1.0]],
    rhs=_planar_rhs,
    description="non-dimensionalised delayed tumour-immune model",
    pool_growth=4,
)


def sample_initial_conditions(system, n_traj, seed):
    """Uniform draws from the system's box, (n_traj, dim)."""
    rng = np.random.default_rng(seed)
    lo, hi = system.box[:, 0], system.box[:, 1]
    return lo + (hi - lo) * rng.random((n_traj, system.dim))


def grid_tolerance(times):
    """Slack for comparing sample times, a small fraction of the grid spacing.

    A fixed epsilon is too brittle: a step like 1/300 is not representable, so
    times[300] - tau lands ~1e-8 below times[0] and a 1e-9 test throws away a snapshot.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return 10 ** -TIME_DECIMALS
    return max(10 ** -TIME_DECIMALS, 1e-3 * float(np.min(np.diff(np.sort(times)))))


def snapshot_start_index(times, tau):
    """First index whose full history window [t - tau, t] lies inside times."""
    valid = np.nonzero(times - tau >= times[0] - grid_tolerance(times))[0]
    if valid.size == 0:
        raise ValueError(
            f"No valid sampling time: tau={tau} exceeds the span {times[0]}..{times[-1]}"
        )
    return int(valid[0])


def required_times(times, tau, M_values):
    """Every time the solution is needed at, the union of the delay-window grids.

    Integrating exactly at these avoids interpolating the solution afterwards.
    """
    start = snapshot_start_index(times, tau)
    base = times[start:]

    wanted = []
    for M in M_values:
        dh = tau / (M - 1)
        for j in range(M):
            wanted.append(base - tau + j * dh)

    allt = np.round(np.concatenate(wanted), TIME_DECIMALS)
    uniq = np.unique(allt)
    if uniq[0] < times[0] - grid_tolerance(times):
        raise ValueError("Delay window reaches before the start of the integration span")
    return uniq


@dataclass
class TrajectoryBank:
    """Solutions of many trajectories, sampled on one shared set of times."""

    times: np.ndarray                  # (T,) sorted, rounded to TIME_DECIMALS
    values: np.ndarray                 # (n_traj, T, dim)
    x0: np.ndarray                     # (n_traj, dim)
    system: DDESystem = field(default=None, repr=False)

    @property
    def n_traj(self):
        return self.values.shape[0]

    def at(self, query, traj_idx=None):
        """Solution at query times, (n_selected, len(query), dim).

        Query times must be members of self.times.
        """
        q = np.round(np.asarray(query, dtype=float), TIME_DECIMALS)
        idx = np.searchsorted(self.times, q)
        if idx.max(initial=0) >= self.times.size or not np.array_equal(self.times[idx], q):
            raise KeyError("Query time is not one of the integrated sample times")
        values = self.values if traj_idx is None else self.values[traj_idx]
        return values[:, idx, :]


def integrate_trajectories(system, x0_all, times, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL,
                           verbose=True):
    """Integrate from every initial condition, sampled at times, into a TrajectoryBank.

    The history is constant at x0 for t <= times[0], as dde23(f, lags, @(t) x0, tspan).
    The compiled right-hand side is built once and reused via purge_past, which is what
    makes a pool of ~1000 trajectories cheap.
    """
    import warnings

    from jitcdde import jitcdde, t as _t, y as _y

    x0_all = np.asarray(x0_all, dtype=float).reshape(-1, system.dim)
    times = np.asarray(times, dtype=float)
    t0 = float(times[0])

    if verbose:
        print(f"[gen] compiling {system.name} RHS ({system.description}) ...", flush=True)

    DDE = jitcdde(system.rhs(_y, _t), n=system.dim, verbose=False)
    DDE.compile_C(simplify=False, do_cse=False, chunk_size=15)

    values = np.empty((len(x0_all), len(times), system.dim), dtype=float)
    started = _time.perf_counter()

    with warnings.catch_warnings():
        # jitcdde warns whenever a requested output time falls inside the step it has
        # already taken; the value is then read off that step's Hermite polynomial,
        # which is the intended dense output and exactly what deval does in MATLAB.
        warnings.filterwarnings("ignore", message=".*target time is smaller.*")
        # The initial-history discontinuity is handled by tolerance, not by jitcdde's
        # adjust_diff()/step_on_discontinuities(). Both are wrong here: adjust_diff
        # gives the past anchor a non-zero derivative, so the history is no longer the
        # constant x0 that MATLAB's `@(t) x0` specifies (checked against the closed-form
        # solution on the first delay interval: 1e-2 error), and step_on_discontinuities
        # advances past the early snapshots we need. At rtol=atol=1e-10 the plain
        # integration reproduces that closed form to 5e-10.
        warnings.filterwarnings("ignore", message=".*initial discontinuities.*")

        for k, x0 in enumerate(x0_all):
            DDE.purge_past()
            DDE.constant_past(x0, time=t0)
            DDE.set_integration_parameters(rtol=rtol, atol=atol)

            values[k, 0] = x0
            for i, ti in enumerate(times[1:], start=1):
                values[k, i] = DDE.integrate(float(ti))

            if verbose and ((k + 1) % 100 == 0 or k + 1 == len(x0_all)):
                el = _time.perf_counter() - started
                print(f"[gen]   {k + 1}/{len(x0_all)} trajectories  ({el:.1f}s)", flush=True)

    return TrajectoryBank(times=np.round(times, TIME_DECIMALS), values=values,
                          x0=x0_all, system=system)


def raw_system(dim, tau, name="raw", description="user-supplied trajectories",
               pool_growth=1):
    """A DDESystem carrying only what the packaging step reads: dim and tau.

    build_split_dataset never calls the right-hand side or samples the box, so
    trajectories from any solver can be packaged by pairing them with one of these.
    """
    return DDESystem(name=name, dim=int(dim), tau=float(tau),
                     box=np.zeros((int(dim), 2)), rhs=None, description=description,
                     pool_growth=pool_growth)


def _normalise_raw(times, values, dim=None):
    """Coerce raw solver output to (times, values) with values (n_traj, T, dim).

    Accepts (T,), (n_traj, T), (T, dim), (n_traj, T, dim) and (n_traj, dim, T),
    disambiguated against len(times) and, when given, dim. Times are sorted here;
    duplicates are an error because they make the sample lookup ambiguous.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float)
    T = times.size

    if v.ndim == 1:
        v = v.reshape(1, -1, 1)
    elif v.ndim == 2:
        if dim is not None and v.shape[-1] == dim and dim != 1:
            v = v[None, :, :]                       # (T, dim), one trajectory
        elif v.shape[1] == T and v.shape[0] != T:
            v = v[:, :, None]                       # (n_traj, T), scalar system
        elif v.shape[0] == T:
            v = v[None, :, :]                       # (T, dim), one trajectory
        else:
            raise ValueError(
                f"values{v.shape} has no axis of length len(times)={T}"
            )
    elif v.ndim == 3:
        if v.shape[1] == T and v.shape[2] != T:
            pass                                    # (n_traj, T, dim)
        elif v.shape[2] == T:
            v = np.transpose(v, (0, 2, 1))          # (n_traj, dim, T)
        else:
            raise ValueError(
                f"values{v.shape} has no axis of length len(times)={T}"
            )
    else:
        raise ValueError(f"values must have 1-3 axes, got {v.ndim}")

    if v.shape[1] != T:
        raise ValueError(f"values{v.shape} does not line up with len(times)={T}")
    if dim is not None and v.shape[2] != dim:
        raise ValueError(f"values imply dim={v.shape[2]}, but dim={dim} was given")

    order = np.argsort(times)
    times, v = times[order], v[:, order, :]
    if np.any(np.diff(np.round(times, TIME_DECIMALS)) == 0):
        raise ValueError("Duplicate sample times in the raw data")
    return times, v


def bank_from_raw(times, values, tau, dim=None, sample_at=None, x0=None,
                  name="raw", description="user-supplied trajectories", verbose=True):
    """Build a TrajectoryBank from trajectories some other solver produced.

    tau is not inferable from the data and must match the system the trajectories came
    from. sample_at is the times the bank must answer for, normally
    required_times(grid, tau, M_values): points already in times are copied exactly,
    any others are filled by a cubic spline. x0 defaults to the first stored sample.

    Interpolation is a last resort, not a feature — a cubic spline on a grid of spacing
    h carries an O(h^4) error that the rest of the pipeline treats as exact solver
    output. Better to re-run the solver with required_times as its output times.
    """
    times, v = _normalise_raw(times, values, dim)
    system = raw_system(v.shape[2], tau, name=name, description=description)

    if x0 is None:
        x0 = v[:, 0, :].copy()
    else:
        x0 = np.asarray(x0, dtype=float).reshape(-1, system.dim)
        if len(x0) != len(v):
            raise ValueError(f"x0 has {len(x0)} rows but there are {len(v)} trajectories")

    if sample_at is not None:
        v, times = _resample(times, v, np.asarray(sample_at, dtype=float), verbose)

    if verbose:
        print(f"[gen] bank from raw data: {v.shape[0]} trajectories, dim={system.dim}, "
              f"tau={system.tau:g}, {len(times)} sample times over "
              f"[{times[0]:g}, {times[-1]:g}]")

    return TrajectoryBank(times=np.round(times, TIME_DECIMALS), values=v, x0=x0,
                          system=system)


def _resample(times, values, wanted, verbose=True):
    """Evaluate values at wanted, copying stored samples where they exist and cubic
    splining the rest. Extrapolation is refused rather than silently allowed."""
    from scipy.interpolate import CubicSpline

    wanted = np.round(np.asarray(wanted, dtype=float).reshape(-1), TIME_DECIMALS)
    have = np.round(times, TIME_DECIMALS)
    tol = grid_tolerance(have)

    if wanted[0] < have[0] - tol or wanted[-1] > have[-1] + tol:
        raise ValueError(
            f"Requested times span [{wanted[0]:g}, {wanted[-1]:g}] but the raw data only "
            f"covers [{have[0]:g}, {have[-1]:g}] — the delay window reaches outside it"
        )

    idx = np.clip(np.searchsorted(have, wanted), 0, have.size - 1)
    exact = have[idx] == wanted

    out = np.empty((values.shape[0], wanted.size, values.shape[2]), dtype=float)
    out[:, exact, :] = values[:, idx[exact], :]

    n_interp = int((~exact).sum())
    if n_interp:
        if have.size < 4:
            raise ValueError("Cubic interpolation needs at least 4 raw sample times")
        out[:, ~exact, :] = CubicSpline(have, values, axis=1)(wanted[~exact])
        if verbose:
            print(f"[gen] {n_interp}/{wanted.size} requested times are not in the raw "
                  f"data; filled by cubic spline (see bank_from_raw notes)")

    return out, wanted


def _embed(bank, times, tau, M, traj_idx, start):
    """Delay-embedded states, (n_traj, nSnap + 1, dim * M).

    One more column than there are snapshots, so successors are just a shift.
    """
    dim = bank.system.dim
    dh = tau / (M - 1)
    base = times[start:]                              # (nSnap + 1,)
    offsets = -tau + dh * np.arange(M)                # oldest .. newest

    # (nSnap + 1, M) query grid -> (n_selected, nSnap + 1, M, dim)
    grid = base[:, None] + offsets[None, :]
    sampled = bank.at(grid.reshape(-1), traj_idx=traj_idx)
    sampled = sampled.reshape(len(traj_idx), base.size, M, dim)

    # Flatten (M, dim) oldest-first, matching MATLAB's column-major reshape(Y, [], 1).
    return sampled.reshape(len(traj_idx), base.size, M * dim)


def build_split_dataset(bank, times, M, train_idx, test_idx):
    """The data_matrix / test_data pair for one history resolution.

    train_idx and test_idx are 0-based into the pool; the stored *_traj_indices fields
    are 1-based, to match the MATLAB files.
    """
    system = bank.system
    tau = system.tau
    start = snapshot_start_index(times, tau)
    # MATLAB's start_idx is 1-based, so nSnap = len(tt) - start_idx = len - start - 1.
    # times[start:] then holds nSnap + 1 entries: the snapshot times plus one successor.
    n_snap = len(times) - start - 1

    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Train and test index sets must be disjoint")

    embedded_train = _embed(bank, times, tau, M, train_idx, start)
    # Z uses snapshots [0, nSnap), Zp the same window shifted one step forward.
    Z = embedded_train[:, :n_snap, :].reshape(-1, M * system.dim).T
    Zp = embedded_train[:, 1:n_snap + 1, :].reshape(-1, M * system.dim).T

    embedded_test = _embed(bank, times, tau, M, test_idx, start)
    # (dim*M, nSnap, nTest)
    Z_test = np.transpose(embedded_test[:, :n_snap, :], (2, 1, 0))

    data_matrix = {
        "Z": np.ascontiguousarray(Z),
        "Zp": np.ascontiguousarray(Zp),
        "dim": system.dim,
        "M": M,
        "tau": tau,
        "Delta_h": tau / (M - 1),
        "nSnap": n_snap,
        "train_traj_indices": train_idx + 1,
    }
    test_data = {
        "Z_test": np.ascontiguousarray(Z_test),
        "dim": system.dim,
        "M": M,
        "tau": tau,
        "Delta_h": tau / (M - 1),
        "nSnap": n_snap,
        "nTest": len(test_idx),
        "test_traj_indices": test_idx + 1,
    }
    return data_matrix, test_data


def pool_sizes(M_values, n_traj_base, growth):
    """Pool size per M: n_traj_base * growth ** (M - min(M)).

    The pool grows with the delay coordinates so the sample density in the lifted space
    stays comparable as M increases.
    """
    return {M: n_traj_base * growth ** (M - min(M_values)) for M in M_values}


def _check_pools(bank, pools, n_test):
    biggest = max(pools.values())
    if biggest > bank.n_traj:
        raise ValueError(
            f"Pool of {biggest} trajectories requested but the bank holds "
            f"{bank.n_traj}. Supply more trajectories or lower the pool sizes."
        )
    if min(pools.values()) <= n_test:
        raise ValueError(
            f"Every pool must exceed nTest={n_test}; smallest is {min(pools.values())}"
        )


def make_flat_dataset_from_bank(bank, times, M_values=(2, 3), pools=None, n_test=15,
                                seed=1, verbose=True):
    """Package a TrajectoryBank into data_matrix_<M> / test_data_<M> variables.

    The bank may come from integrate_trajectories or from bank_from_raw — this step
    never touches the solver. times is the snapshot grid; successors are one step of it
    apart, which fixes the timestep of the learned operator.

    pools maps M -> pool size, defaulting to the whole bank for every M. One test set
    is drawn from the smallest pool and shared, so every M is scored on the same
    trajectories.
    """
    M_values = list(M_values)
    pools = {M: bank.n_traj for M in M_values} if pools is None else dict(pools)
    _check_pools(bank, pools, n_test)

    # Common test set drawn from the smallest pool, as in the MATLAB scripts, so the
    # same indices are valid in every (larger) pool.
    rng = np.random.default_rng(seed)
    test_idx = rng.permutation(pools[min(M_values)])[:n_test]

    out = {}
    for M in M_values:
        train_idx = np.setdiff1d(np.arange(pools[M]), test_idx)
        dm, td = build_split_dataset(bank, times, M, train_idx, test_idx)
        out[f"data_matrix_{M}"] = dm
        out[f"test_data_{M}"] = td
        if verbose:
            print(f"[gen]   M={M}: pool={pools[M]}, train={len(train_idx)}, "
                  f"Z={dm['Z'].shape}, Z_test={td['Z_test'].shape}")
    return out


def make_flat_dataset(system, M_values=(2, 3), n_traj_base=115, n_test=15,
                      pool_growth=None, times=None, seed=1, verbose=True,
                      rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
    """Integrate system and package it, as dde_code.m and dde_planar_flat.m do."""
    times = default_time_grid() if times is None else times
    M_values = list(M_values)
    growth = system.pool_growth if pool_growth is None else pool_growth

    pools = pool_sizes(M_values, n_traj_base, growth)
    if n_traj_base <= n_test:
        raise ValueError("n_traj_base must exceed n_test")

    if verbose:
        print(f"[gen] {system.name} flat: pools {pools}, nTest={n_test}")

    x0_all = sample_initial_conditions(system, max(pools.values()), seed)
    sample_at = required_times(times, system.tau, M_values)
    if verbose:
        print(f"[gen] {len(sample_at)} sample times per trajectory "
              f"over [{sample_at[0]:g}, {sample_at[-1]:g}]")

    bank = integrate_trajectories(system, x0_all, sample_at, rtol, atol, verbose)
    return make_flat_dataset_from_bank(bank, times, M_values, pools, n_test, seed,
                                       verbose)


def save_dataset(variables, path, verbose=True):
    """Write a .mat file that load_dataset can read back."""
    savemat(path, variables, do_compression=True, oned_as="row")
    if verbose:
        import os
        print(f"[gen] wrote {path} ({os.path.getsize(path) / 1024**2:.1f} MiB)")
    return path
