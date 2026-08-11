"""Device and dtype for the whole package.

KEDMD_DEVICE picks cuda / cuda:1 / cpu, default CUDA when available. KEDMD_DTYPE picks
float64 (default) or float32. float64 is the default because the estimator inverts
kernel matrices close to singular, and the reference results were computed in it.
"""

import os

import torch

_DTYPES = {
    "float64": torch.float64, "double": torch.float64, "f64": torch.float64,
    "float32": torch.float32, "float": torch.float32, "f32": torch.float32,
}

# FP64:FP32 throughput ratio by compute capability.
_FP64_PENALTY = {
    (8, 6): 64, (8, 9): 64, (7, 5): 32, (6, 1): 32, (12, 0): 64, (10, 0): 2,
    (9, 0): 2, (8, 0): 2, (7, 0): 2,
}

_device = None
_dtype = None
_announced = False


def _resolve_device():
    requested = os.environ.get("KEDMD_DEVICE", "").strip()
    if requested:
        dev = torch.device(requested)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"KEDMD_DEVICE={requested!r} but CUDA is not available to this "
                "PyTorch build. Unset it to fall back to CPU."
            )
        return dev
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_device(announce=True):
    """The device every computation runs on, resolved once per process."""
    global _device, _announced
    if _device is None:
        _device = _resolve_device()
    if announce and not _announced:
        _announced = True
        print(describe_device(_device))
    return _device


def set_device(device):
    """Override the device. None re-resolves from the environment."""
    global _device, _announced
    _device = torch.device(device) if device is not None else None
    _announced = False
    return get_device(announce=False)


def get_dtype():
    """The floating dtype every computation uses."""
    global _dtype
    if _dtype is None:
        name = os.environ.get("KEDMD_DTYPE", "float64").strip().lower()
        if name not in _DTYPES:
            raise ValueError(
                f"KEDMD_DTYPE={name!r} not understood; use one of {sorted(_DTYPES)}"
            )
        _dtype = _DTYPES[name]
    return _dtype


def set_dtype(dtype):
    """Override the dtype, as a torch dtype or a name like 'float32'."""
    global _dtype
    if isinstance(dtype, str):
        dtype = _DTYPES[dtype.strip().lower()]
    _dtype = dtype
    return _dtype


def describe_device(device=None):
    """Report of the compute device, its configuration and the dtype."""
    device = torch.device(device) if device is not None else get_device(announce=False)
    dtype = get_dtype()
    lines = []

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        cc = (p.major, p.minor)
        free, total = torch.cuda.mem_get_info(idx)

        lines.append(f"[kedmd_dde] Compute device: GPU — {p.name} (cuda:{idx})")
        lines.append(
            f"            compute capability sm_{p.major}{p.minor} | "
            f"{p.multi_processor_count} SMs | "
            f"{total / 1024**3:.1f} GiB total, {free / 1024**3:.1f} GiB free"
        )
        lines.append(
            f"            torch {torch.__version__} | CUDA runtime {torch.version.cuda} | "
            f"cuDNN {torch.backends.cudnn.version()}"
        )

        penalty = _FP64_PENALTY.get(cc)
        note = ""
        if dtype == torch.float64 and penalty and penalty > 8:
            note = (
                f"  <- FP64 runs at 1/{penalty} of FP32 on sm_{p.major}{p.minor}; "
                "set KEDMD_DTYPE=float32 to trade precision for speed"
            )
        lines.append(f"            dtype: {dtype}{note}")
    else:
        lines.append(f"[kedmd_dde] Compute device: CPU ({torch.get_num_threads()} threads)")
        reason = (
            "no CUDA-capable GPU visible to PyTorch"
            if not torch.cuda.is_available()
            else "a CUDA GPU is available but the CPU was selected explicitly"
        )
        lines.append(f"            torch {torch.__version__} | {reason}")
        lines.append(f"            dtype: {dtype}")

    return "\n".join(lines)


def print_device_info(device=None, brief=False):
    """Print describe_device, and mark the automatic announcement as done.

    brief prints one line naming the device instead of the full report, which is what
    the examples use; either way the automatic announcement is suppressed.
    """
    global _announced
    _announced = True
    if not brief:
        print(describe_device(device))
        return

    device = torch.device(device) if device is not None else get_device(announce=False)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    print(f"[kedmd_dde] device: {device}")


def as_tensor(a, device=None, dtype=None):
    """Move an array, tensor or scalar onto the package device and dtype."""
    device = device if device is not None else get_device()
    dtype = dtype if dtype is not None else get_dtype()
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(a, device=device, dtype=dtype)


def to_numpy(a):
    """Detach a tensor to a NumPy array; pass anything else through."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return a


def synchronize(device=None):
    """Block until queued GPU work has finished — needed for honest timings."""
    device = device if device is not None else get_device(announce=False)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
