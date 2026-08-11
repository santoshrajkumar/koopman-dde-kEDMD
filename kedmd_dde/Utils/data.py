"""Loading the datasets.

Two kinds of file are read. An embedded .mat holds data_matrix_<M> / test_data_<M>,
one pair per history resolution. A raw .npz holds only the
trajectories, and the delay embedding, split and successor pairing happen at load time
— that file is about 2M times smaller, since Z and Zp repeat every sample 2M times.

load_dataset takes either and returns a DelayDataset on the package device.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.io import loadmat

from .device import as_tensor

_HERE = os.path.dirname(os.path.abspath(__file__))              # kedmd_dde/Utils
_PACKAGE_ROOT = os.path.dirname(_HERE)                          # kedmd_dde
PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

# The records ship inside the package, so a name resolves the same way from a wheel as
# from a checkout. A source tree may also keep a data/ beside the package — freshly
# generated or too big to ship — and that is searched second.
DATA_DIR = os.path.join(_PACKAGE_ROOT, "data")
EXTRA_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Figures are output, not package content: they go beside the package in a checkout and
# into the working directory otherwise, never into site-packages.
_IN_CHECKOUT = os.path.exists(os.path.join(PROJECT_ROOT, "pyproject.toml"))
FIGURE_DIR = os.path.join(PROJECT_ROOT if _IN_CHECKOUT else os.getcwd(), "figures")

DEFAULT_DT = 0.01                 # assumed for an embedded file, which records no step

# Key names a raw record is searched for, so files from other tools load unchanged.
RAW_VALUE_KEYS = ("values", "y", "x", "Y", "X", "sol", "states", "traj")
RAW_TIME_KEYS = ("times", "t", "tspan", "time", "tt")


def data_path(name):
    """Resolve name against the packaged data/, then a data/ beside the package.

    Absolute or existing paths pass through. A name that is in neither directory comes
    back as the packaged path, so the error names where it was expected.
    """
    if os.path.isabs(name) or os.path.exists(name):
        return name
    packaged = os.path.join(DATA_DIR, name)
    if not os.path.exists(packaged):
        extra = os.path.join(EXTRA_DATA_DIR, name)
        if os.path.exists(extra):
            return extra
    return packaged


def figure_path(name):
    """Resolve name against figures/, creating the directory if needed."""
    if os.path.isabs(name):
        return name
    os.makedirs(FIGURE_DIR, exist_ok=True)
    return os.path.join(FIGURE_DIR, name)


def matobj_to_dict(obj):
    """Recursively turn scipy.io.loadmat structs into plain dicts."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            if obj.size == 1:
                return matobj_to_dict(obj.item())
            return [matobj_to_dict(o) for o in obj.flat]
        return obj

    if hasattr(obj, "_fieldnames"):
        return {name: matobj_to_dict(getattr(obj, name)) for name in obj._fieldnames}

    return obj


def _field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class DelayDataset:
    """One (M, dim) slice of a dataset, held on the package device."""

    X: torch.Tensor        # (N, nM)  training states
    Y: torch.Tensor        # (N, nM)  one-step successors
    Z_test: torch.Tensor   # (nM, nSnap, nTest)
    M: int
    dim: int
    tau: float
    dt: float = 0.01
    name: str = ""

    @property
    def device(self):
        return self.X.device

    @property
    def lifted_dim(self):
        return self.X.shape[1]

    @property
    def n_train(self):
        return self.X.shape[0]

    @property
    def n_steps(self):
        return self.Z_test.shape[1]

    @property
    def n_test(self):
        return self.Z_test.shape[2]

    @property
    def current_state_slice(self):
        """Slice picking x(t) — the newest block — out of a lifted state."""
        return slice((self.M - 1) * self.dim, self.M * self.dim)

    def test_trajectory(self, i):
        return self.Z_test[:, :, i].T

    def test_trajectories(self):
        """All test trajectories as (nTest, n_steps, nM)."""
        return self.Z_test.permute(2, 1, 0).contiguous()

    def initial_states(self):
        """The initial lifted state of every test trajectory, (nTest, nM)."""
        return self.Z_test[:, 0, :].T.contiguous()

    def to(self, device=None, dtype=None):
        self.X = self.X.to(device=device, dtype=dtype)
        self.Y = self.Y.to(device=device, dtype=dtype)
        self.Z_test = self.Z_test.to(device=device, dtype=dtype)
        return self

    def summary(self):
        return (
            f"M={self.M}, dim={self.dim}, nM={self.lifted_dim}, "
            f"train={tuple(self.X.shape)}, test={tuple(self.Z_test.shape)}, "
            f"device={self.device}"
        )


# ----------------------------------------------------------------- raw trajectories

def load_raw(filename):
    """Read a raw trajectory record from .npz / .npy / .mat into a dict."""
    path = str(data_path(filename))
    if path.endswith(".npz"):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    if path.endswith(".npy"):
        return {"values": np.load(path)}
    return {k: v for k, v in loadmat(path, squeeze_me=True).items()
            if not k.startswith("__")}


def _pick(raw, name, candidates):
    for k in (name,) + tuple(candidates):
        if k in raw:
            return raw[k]
    raise KeyError(f"raw data has no '{name}'; found {sorted(raw)}")


def raw_values(raw):
    """The states as (nTraj, T, dim), plus dt and tau."""
    v = np.asarray(_pick(raw, "values", RAW_VALUE_KEYS), dtype=float)
    if v.ndim == 2:
        v = v[:, :, None]                      # (nTraj, T), scalar system
    elif v.ndim != 3:
        raise ValueError(f"raw values must be (nTraj, T, dim) or (nTraj, T), got {v.shape}")

    tau = float(_pick(raw, "tau", ("tau_d", "delay")))

    dt = raw.get("dt")
    if dt is None:
        times = np.asarray(_pick(raw, "times", RAW_TIME_KEYS), dtype=float).reshape(-1)
        if times.size != v.shape[1]:
            raise ValueError(
                f"times has {times.size} entries but values has {v.shape[1]} samples"
            )
        steps = np.round(np.diff(times), 9)
        # Successors are one grid step apart, so a non-uniform grid has no single step.
        if steps.size == 0 or not np.all(steps == steps[0]):
            raise ValueError("raw 'times' is not uniformly spaced — pass 'dt' explicitly")
        dt = float(steps[0])
    return v, float(dt), tau


def window_stride(tau, dt, M):
    """Index offsets of one delay window on the raw grid, or None if it misses it.

    The M window points are tau/(M-1) apart, so M-1 must divide tau/dt for them to be
    stored samples. The tolerance is relative because a step like 1/300 is not
    representable and gives tau/dt = 300.00003.
    """
    lag = tau / dt
    step = tau / (M - 1) / dt
    if any(abs(v - round(v)) > 1e-6 * max(1.0, abs(v)) for v in (lag, step)):
        return None
    return int(round(lag)), int(round(step))


def raw_available_M(raw, M_max=12):
    """History resolutions a raw record can represent without interpolation."""
    values, dt, tau = raw_values(raw)
    return [M for M in range(2, M_max + 1)
            if window_stride(tau, dt, M) is not None
            and values.shape[1] - int(round(tau / dt)) - 1 >= 1]


def _embed(values, n_snap, M, step, offset):
    """Delay-embed (nTraj, T, dim) into (nTraj, n_snap, M*dim) on the device.

    Each of the M blocks is one strided slice copied straight into its column block,
    so nothing bigger than the result is allocated. Blocks run oldest first, so the
    newest dim columns are x(t).
    """
    n_traj, _, dim = values.shape
    out = torch.empty((n_traj, n_snap, M * dim), dtype=values.dtype, device=values.device)
    blocks = out.view(n_traj, n_snap, M, dim)
    for j in range(M):
        a = offset + j * step
        blocks[:, :, j, :] = values[:, a:a + n_snap, :]
    return out


def dataset_from_raw(raw, M, n_test=100, seed=1, n_traj=None, name="raw"):
    """Build the M slice of a dataset from raw trajectories, in memory.

    raw holds values (nTraj, T, dim), tau, and either dt or a uniform times array.
    n_test, seed and n_traj choose the split here rather than in a file; the draw
    depends only on the seed and the pool size, so every M sees the same test set.

    Window points are found by index, never interpolated, so where dt divides tau this
    is bit-for-bit what gen_data/ writes into a .mat.
    """
    values, dt, tau = raw_values(raw)

    stride = window_stride(tau, dt, M)
    if stride is None:
        raise ValueError(
            f"M={M} needs samples at spacing tau/(M-1)={tau / (M - 1):g}, which is not a "
            f"multiple of dt={dt:g}; the delay window falls between raw samples. Either "
            f"save the raw data on a finer grid or resample it with "
            f"kedmd_dde.Utils.generate.bank_from_raw(sample_at=required_times(...))."
        )
    lag, step = stride

    n_pool = values.shape[0] if n_traj is None else int(n_traj)
    if n_pool > values.shape[0]:
        raise ValueError(f"n_traj={n_pool} but the raw data holds {values.shape[0]}")
    if n_pool <= n_test:
        raise ValueError(f"pool of {n_pool} trajectories cannot spare nTest={n_test}")

    # First snapshot is the earliest whose history window fits; the last sample is kept
    # as a successor.
    n_snap = values.shape[1] - lag - 1
    if n_snap < 1:
        raise ValueError(
            f"{values.shape[1]} samples per trajectory is too few for a delay of "
            f"{lag} steps"
        )

    rng = np.random.default_rng(seed)
    test_idx = rng.permutation(n_pool)[:n_test]
    train_idx = np.setdiff1d(np.arange(n_pool), test_idx)

    v = as_tensor(values)
    train = _embed(v[train_idx], n_snap + 1, M, step, offset=0)
    X = train[:, :n_snap, :].reshape(-1, M * values.shape[2])
    Y = train[:, 1:, :].reshape(-1, M * values.shape[2])

    test = _embed(v[test_idx], n_snap, M, step, offset=0)
    Z_test = test.permute(2, 1, 0).contiguous()          # (nM, nSnap, nTest)

    return DelayDataset(
        X=X, Y=Y, Z_test=Z_test, M=M, dim=values.shape[2], tau=tau, dt=dt,
        name=f"{name}:M={M}",
    )


# ------------------------------------------------------------------ embedded files

def _raw_mat(filename):
    return loadmat(data_path(filename), squeeze_me=True, struct_as_record=False)


def _is_raw(raw):
    if any(str(k).startswith("data_matrix_") for k in raw):
        return False
    return any(k in raw for k in RAW_VALUE_KEYS)


def _check_embedded(raw, filename):
    if not any(k.startswith("data_matrix_") for k in raw):
        raise ValueError(
            f"Unrecognised .mat layout in {filename}: expected 'data_matrix_<M>' "
            f"variables, found {sorted(k for k in raw if not k.startswith('__'))}"
        )


def _dataset_from_structs(data_matrix, test_data, dt, name):
    Z = np.asarray(_field(data_matrix, "Z"), dtype=float)        # (nM, N)
    Zp = np.asarray(_field(data_matrix, "Zp"), dtype=float)      # (nM, N)
    Z_test = np.asarray(_field(test_data, "Z_test"), dtype=float)

    return DelayDataset(
        X=as_tensor(Z.T).contiguous(),
        Y=as_tensor(Zp.T).contiguous(),
        Z_test=as_tensor(Z_test),
        M=int(_field(data_matrix, "M")),
        dim=int(_field(data_matrix, "dim")),
        tau=float(_field(data_matrix, "tau")),
        dt=dt,
        name=name,
    )


def available_M(filename):
    """History resolutions available: what an embedded file stores, or what a raw one
    can represent without interpolation."""
    if not str(filename).endswith(".mat"):
        return raw_available_M(load_raw(filename))

    raw = _raw_mat(filename)
    if _is_raw(raw):
        return raw_available_M(raw)
    _check_embedded(raw, filename)
    return sorted(int(k.split("_")[-1]) for k in raw if k.startswith("data_matrix_"))


def load_metadata(filename):
    """The settings stored alongside a raw record, or None for an embedded file."""
    if not str(filename).endswith(".mat"):
        raw = load_raw(filename)
        return {k: v for k, v in raw.items() if k not in RAW_VALUE_KEYS} or None
    raw = _raw_mat(filename)
    if _is_raw(raw):
        return {k: v for k, v in raw.items() if k not in RAW_VALUE_KEYS} or None
    return None


def _checked_dt(dt, ds, filename):
    # Callers use the same dt for the plot time axis, so a mismatch would mislabel
    # every figure rather than fail.
    if dt is not None and abs(dt - ds.dt) > 1e-12:
        raise ValueError(
            f"{filename} was sampled at dt={ds.dt:g}, but dt={dt:g} was requested. "
            f"The raw record's step is the timestep of the learned operator and the "
            f"time axis of the rollout — pass dt={ds.dt:g} or omit it."
        )
    return ds


def load_dataset(filename, M, dt=None, **raw_kwargs):
    """Load the slice for history resolution M.

    filename is a name inside data/, any path, or a raw record already in memory.
    raw_kwargs (n_test, seed, n_traj) go to dataset_from_raw and only apply to raw
    input, since an embedded file has its split already stored.
    """
    if not isinstance(filename, (str, os.PathLike)):
        return _checked_dt(dt, dataset_from_raw(filename, M, **raw_kwargs), "the raw data")

    name = os.path.basename(str(filename))
    if not str(filename).endswith(".mat"):
        ds = dataset_from_raw(load_raw(filename), M, name=name, **raw_kwargs)
        return _checked_dt(dt, ds, filename)

    raw = _raw_mat(filename)
    if _is_raw(raw):
        ds = dataset_from_raw(raw, M, name=name, **raw_kwargs)
        return _checked_dt(dt, ds, filename)
    if raw_kwargs:
        raise TypeError(
            f"{filename} is an embedded dataset with its split already stored; "
            f"{sorted(raw_kwargs)} only apply to raw trajectories"
        )
    _check_embedded(raw, filename)
    dt = DEFAULT_DT if dt is None else dt

    dkey, tkey = f"data_matrix_{M}", f"test_data_{M}"
    if dkey not in raw:
        raise KeyError(f"{filename} has no {dkey}; available M: {available_M(filename)}")
    return _dataset_from_structs(raw[dkey], raw[tkey], dt, f"{filename}:M={M}")
