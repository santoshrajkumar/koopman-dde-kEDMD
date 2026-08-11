"""Fit a model, roll out every test trajectory, and summarise the error."""

import time

import torch

from .device import get_device, synchronize


def run_experiment(dataset, cfg, seed=0, verbose=True, **build_kwargs):
    """Fit one kEDMD model on dataset and predict all of its test trajectories.

    cfg holds n_centers, rho, epsilon, d_neighbors and reg; anything else is ignored,
    so sweep configs can carry extra bookkeeping fields. build_kwargs go to build_kedmd.
    """
    from ..KEDMD_DDE import build_kedmd    # imported here: KEDMD_DDE imports Utils

    get_device()                      # resolve and announce the device on first use

    if verbose:
        print("\n===================================")
        print(f"Running experiment for M = {dataset.M} (nM = {dataset.lifted_dim})")
        print("Training X shape:", tuple(dataset.X.shape))
        print("Training Y shape:", tuple(dataset.Y.shape))
        print("Z_test shape:", tuple(dataset.Z_test.shape))
        print("===================================")

    synchronize()
    t0 = time.perf_counter()

    model = build_kedmd(
        dataset.X,
        dataset.Y,
        n_centers=cfg["n_centers"],
        rho=cfg["rho"],
        epsilon=cfg["epsilon"],
        d_neighbors=cfg.get("d_neighbors", 25),
        reg=cfg.get("reg", 1e-8),
        seed=seed,
        **build_kwargs,
    )

    synchronize()
    t_fit = time.perf_counter() - t0

    all_true = dataset.test_trajectories()
    all_pred = model.rollout_many(dataset.initial_states(), dataset.n_steps)

    synchronize()
    t_total = time.perf_counter() - t0

    if verbose:
        print("\nConfig:", cfg)
        print("Info:", model.info)
        print(f"Timing: fit {t_fit:.2f}s, +rollout {t_total - t_fit:.2f}s "
              f"= {t_total:.2f}s on {model.centers.device}")

    return {
        "cfg": cfg,
        "info": model.info,
        "model": model,
        "centers": model.centers,
        "A": model.A,
        "all_true": all_true,
        "all_pred": all_pred,
        "M": dataset.M,
        "lifted_dim": dataset.lifted_dim,
        "dt": dataset.dt,
        "fit_seconds": t_fit,
        "total_seconds": t_total,
    }


def trajectory_errors(result, floor=1e-12, component=None):
    """Per-trajectory error norms, shape (n_test, n_steps).

    component=None uses the full lifted-state 2-norm, which is not comparable across M:
    the lifted state stacks M copies of the signal, so its norm grows like sqrt(M) and
    carries that factor into the error. An int gives the absolute error of one
    component, a slice the 2-norm over a block — pass dataset.current_state_slice to
    score x(t) alone, which is comparable.
    """
    true_traj = result["all_true"]
    pred_traj = result["all_pred"]

    if not isinstance(true_traj, torch.Tensor):
        true_traj = torch.stack(list(true_traj))
        pred_traj = torch.stack(list(pred_traj))

    if component is None:
        err = torch.linalg.norm(pred_traj - true_traj, dim=2)
    elif isinstance(component, slice):
        err = torch.linalg.norm(
            pred_traj[:, :, component] - true_traj[:, :, component], dim=2
        )
    else:
        err = (pred_traj[:, :, component] - true_traj[:, :, component]).abs()

    return err.clamp_min(floor)


def summarize_errors(results, floor=1e-12, component=None):
    """Mean/median/std, the 10-90 percentile curves, and the geometric mean and band.

    Values stay on the package device; the plotting helpers convert.
    """
    summaries = []

    for res in results:
        all_err = trajectory_errors(res, floor=floor, component=component)

        log_err = torch.log10(all_err)
        log_mean = log_err.mean(dim=0)
        log_std = log_err.std(dim=0, unbiased=False)     # ddof=0, as numpy defaults to

        # torch.quantile interpolates linearly like numpy.percentile; the median comes
        # from here because torch.median takes the lower of two middle values instead.
        q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90],
                         device=all_err.device, dtype=all_err.dtype)
        q10, q25, q50, q75, q90 = torch.quantile(all_err, q, dim=0)

        summaries.append({
            "all_err": all_err,
            "mean_err": all_err.mean(dim=0),
            "median_err": q50,
            "std_err": all_err.std(dim=0, unbiased=False),
            "q10_err": q10,
            "q25_err": q25,
            "q75_err": q75,
            "q90_err": q90,
            "log_mean_err": 10 ** log_mean,
            "log_std_lower": 10 ** (log_mean - log_std),
            "log_std_upper": 10 ** (log_mean + log_std),
        })

    return summaries
