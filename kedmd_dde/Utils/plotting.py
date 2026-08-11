"""Shared matplotlib styling and the figure types used across the sweeps.

The boundary where tensors become NumPy arrays: each helper converts only what it is
about to draw.
"""

import matplotlib.pyplot as plt
import numpy as np

from .data import figure_path
from .device import to_numpy

# rcParams shared by every figure in examples/.
PAPER_STYLE = {
    "figure.figsize": (6.4, 4.8),
    "figure.dpi": 140,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "black",
}


def use_paper_style(**overrides):
    """Apply PAPER_STYLE, with per-figure overrides."""
    plt.rcParams.update(PAPER_STYLE)
    if overrides:
        plt.rcParams.update(overrides)


def save_figure(name, **kwargs):
    """Save the current figure into figures/ and return the path used."""
    path = figure_path(name)
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(path, **kwargs)
    return path


def error_curves(summary, band="q10_q90", center="mean", floor=1e-12,
                 lower_floor=None, synthetic_factor=None):
    """Centre curve and shaded band for one error summary, as NumPy arrays.

    band is one of q10_q90, iqr, std, log_std, synthetic. "synthetic" is a
    presentational log-symmetric band of fixed width, not a statistic of the test set,
    and should be labelled as such wherever it is used. floor clips the centre and
    upper curves so the log axis stays finite.
    """
    if lower_floor is None:
        lower_floor = floor

    get = lambda key: np.asarray(to_numpy(summary[key]), dtype=float)

    if center == "median":
        center_curve = get("median_err")
    elif center == "log_mean":
        center_curve = get("log_mean_err")
    elif center == "mean":
        center_curve = get("mean_err")
    else:
        raise ValueError("center must be one of: 'mean', 'median', 'log_mean'")

    if band == "q10_q90":
        lower, upper = get("q10_err"), get("q90_err")
    elif band == "iqr":
        lower, upper = get("q25_err"), get("q75_err")
    elif band == "std":
        lower = np.maximum(get("mean_err") - get("std_err"), floor)
        upper = get("mean_err") + get("std_err")
    elif band == "log_std":
        lower, upper = get("log_std_lower"), get("log_std_upper")
    elif band == "synthetic":
        if synthetic_factor is None:
            raise ValueError("band='synthetic' requires synthetic_factor")
        lower = center_curve / synthetic_factor
        upper = center_curve * synthetic_factor
    else:
        raise ValueError(
            "band must be one of: 'q10_q90', 'iqr', 'std', 'log_std', 'synthetic'"
        )

    return (
        np.maximum(center_curve, floor),
        np.maximum(lower, lower_floor),
        np.maximum(upper, floor),
    )


def plot_error_vs_time(
    results,
    summaries,
    label_fn,
    ax=None,
    dt=None,
    band="q10_q90",
    center="mean",
    alpha=0.16,
    band_color="line",
    floor=1e-12,
    lower_floor=None,
    scale_fn=None,
    synthetic_factor_fn=lambda res: 1.2 + res["cfg"]["rho"],
    title="",
    xlim=None,
    ylabel=r"$\mu_{z_k}$",
    xlabel="Time (s)",
):
    """Log-scale prediction error against time, one curve per result.

    results and summaries are matched outputs of run_experiment and summarize_errors,
    label_fn maps a result to its legend label. dt defaults to each result's own.
    band_color="line" matches the band to its curve, "cycle" takes the next colour in
    the cycle -- so the first curve's band is drawn in the second curve's colour, which
    is why "line" is the default. scale_fn multiplies a curve and its band.

    xlim defaults to the span of the rollout, (0, (n_steps-1) * dt), so a record whose
    horizon is not the 10 s of the published runs is not drawn against an axis that
    overshoots it.
    """
    ax = ax or plt.gca()
    t_max = 0.0

    for res, summ in zip(results, summaries):
        step = dt if dt is not None else res.get("dt", 0.01)

        center_curve, lower, upper = error_curves(
            summ,
            band=band,
            center=center,
            floor=floor,
            lower_floor=lower_floor,
            synthetic_factor=synthetic_factor_fn(res) if band == "synthetic" else None,
        )
        t_eval = np.arange(len(center_curve)) * step

        if scale_fn is not None:
            factor = scale_fn(res)
            center_curve, lower, upper = (
                center_curve * factor, lower * factor, upper * factor,
            )

        line, = ax.plot(t_eval, center_curve, label=label_fn(res))
        fill_kw = {"color": line.get_color()} if band_color == "line" else {}
        ax.fill_between(t_eval, lower, upper, alpha=alpha, **fill_kw)
        t_max = max(t_max, float(t_eval[-1]) if len(t_eval) else 0.0)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim if xlim is not None else (0.0, t_max))
    ax.set_title(title)
    ax.grid(True, which="major", ls="--", alpha=0.7)
    ax.grid(True, which="minor", ls=":", alpha=0.4)
    ax.legend(loc="best")
    return ax


def plot_state_vs_time(
    result,
    traj_indices,
    component,
    ax=None,
    dt=None,
    ylabel=r"$x(t)$",
    xlim=None,
    true_color="tab:blue",
    pred_color="tab:orange",
    linewidth=None,
):
    """True vs predicted time series of one component, overlaid per trajectory.

    Only the first trajectory contributes legend entries, so the legend stays a
    two-entry True/Predicted key. xlim defaults to the span of the rollout.
    linewidth defaults to PAPER_STYLE's lines.linewidth, as plot_error_vs_time uses.
    """
    ax = ax or plt.gca()
    step = dt if dt is not None else result.get("dt", 0.01)
    lw = plt.rcParams["lines.linewidth"] if linewidth is None else linewidth
    t_max = 0.0

    for k, idx in enumerate(traj_indices):
        true_traj = to_numpy(result["all_true"][idx])
        pred_traj = to_numpy(result["all_pred"][idx])
        t = np.arange(true_traj.shape[0]) * step

        ax.plot(t, true_traj[:, component], color=true_color, linewidth=lw,
                linestyle="-", label="True" if k == 0 else None)
        ax.plot(t, pred_traj[:, component], color=pred_color, linewidth=lw,
                linestyle="--", label="Predicted" if k == 0 else None)
        t_max = max(t_max, float(t[-1]) if len(t) else 0.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim if xlim is not None else (0.0, t_max))
    ax.grid(True, which="major", ls="--", alpha=0.7)
    ax.grid(True, which="minor", ls=":", alpha=0.4)
    ax.legend(loc="best")
    return ax


def plot_phase_portrait(result, dims=(0, 1), ax=None, title="",
                        true_style=("k-", 0.35), pred_style=("r--", 0.6)):
    """Every test trajectory in the plane or space of dims.

    dims selects two or three components of the lifted state; with three, ax must be a
    3D axes.
    """
    if len(dims) not in (2, 3):
        raise ValueError("dims must name 2 or 3 lifted-state components")

    if ax is None:
        ax = plt.gcf().add_subplot(projection="3d") if len(dims) == 3 else plt.gca()

    true_fmt, true_alpha = true_style
    pred_fmt, pred_alpha = pred_style

    all_true = to_numpy(result["all_true"])
    all_pred = to_numpy(result["all_pred"])

    for k, (true_traj, pred_traj) in enumerate(zip(all_true, all_pred)):
        ax.plot(*[true_traj[:, d] for d in dims], true_fmt, linewidth=1.0,
                alpha=true_alpha, label="True" if k == 0 else "")
        ax.plot(*[pred_traj[:, d] for d in dims], pred_fmt, linewidth=1.0,
                alpha=pred_alpha, label="Predicted" if k == 0 else "")

    labels = [rf"$z_{{{d + 1}}}$" for d in dims]
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    if len(dims) == 3:
        ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.grid(True)
    return ax
