"""Learn a Koopman model from a record and predict held-out trajectories.
----------------------------
A raw record holds *trajectories only* -- no delay embedding, no train/test split, no
snapshot pairs. load_dataset builds all of that at load time, so one file serves every M
and stays about 2M times smaller than the embedded form. Save the integrator output
as-is, in a .npz (or .npy / .mat):

    values  (nTraj, T, d)  required   values[i, k] is x(t_k) on trajectory i
    tau     scalar         required   the delay of the system
    dt      scalar         one of     the sampling step
    times   (T,)           these two  the sample times, uniformly spaced; dt inferred
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from kedmd_dde import (
    load_dataset,
    plot_state_vs_time,
    print_device_info,
    run_experiment,
    save_figure,
    use_paper_style,
)

# A record shipped with the package, or a path to one of your own in the layout the
# module docstring describes: values (nTraj, T, d), tau, and dt or a uniform times.
DATA = "raw_scalar.npz"      # or "raw_planar.npz", or a path to your own record
M = 3                        # history resolution; M - 1 must divide tau/dt
CFG = {"n_centers": 100, "rho": 0.6, "epsilon": 1.0, "d_neighbors": 25}
N_PICK = 6                   # test trajectories drawn for the figure
PICK_SEED = 56
SAVE_STEM = "learn_and_predict"
SHOW = True


def main():
    print_device_info(brief=True)
    use_paper_style()

    # ---- 1. load: delay-embed, split off a test set, pair successors ----------------
    ds = load_dataset(DATA, M=M)
    print(ds.summary())

    # ---- 2. learn ------------------------------------------------------------------
    # run_experiment fits the model and rolls out every test trajectory in one call.
    res = run_experiment(ds, CFG, verbose=False, assumption3=True)
    model = res["model"]

    a3 = res["info"]["assumption3"]
    print(f"p = {res['info']['num_centers']}, h = {res['info']['fill_distance']:.3e}, "
          f"Assumption 3: {'ok' if a3['holds'] else 'FAIL'}")

    # ---- 3. predict one trajectory by hand -----------------------------------------
    # Same thing run_experiment did for the whole test set, spelled out: lift the first
    # delay-embedded state, iterate the Koopman matrix, decode. Nothing but z0 is used.
    z0 = ds.initial_states()[0]                       # (nM,)
    pred = model.rollout(z0, ds.n_steps)              # (n_steps, nM)
    true = ds.test_trajectory(0)                      # (n_steps, nM)
    cur = ds.current_state_slice                      # the x(t) block of a lifted state
    print(f"trajectory 0: {ds.n_steps} steps of dt = {ds.dt:g} predicted from z0 alone, "
          f"final-step error {float(torch.linalg.norm(pred[-1, cur] - true[-1, cur])):.3e}")

    # ---- 4. reconstruction error over the whole test set ---------------------------
    # Scored on x(t) only: the lifted state stacks M copies of the signal, so an error
    err = res["all_pred"][:, :, cur] - res["all_true"][:, :, cur]
    ref = res["all_true"][:, :, cur]
    rmse = err.pow(2).mean().sqrt()
    print(f"RMSE over {ds.n_test} test trajectories = {float(rmse):.4e}  "
          f"({float(rmse / ref.std(unbiased=False)):.2%} of std(true))")

    # ---- 5. true vs predicted, one figure per state component ----------------------
    gen = torch.Generator().manual_seed(PICK_SEED)
    picks = torch.randperm(ds.n_test, generator=gen)[:N_PICK].tolist()

    for i in range(ds.dim):
        plt.figure()
        label = r"$x(t)$" if ds.dim == 1 else rf"$x_{{{i + 1}}}(t)$"
        plot_state_vs_time(res, picks, component=cur.start + i, dt=ds.dt, ylabel=label)
        suffix = "" if ds.dim == 1 else f"_x{i + 1}"
        print("saved:", save_figure(f"{SAVE_STEM}{suffix}.pdf"))

    if SHOW and plt.get_backend().lower() != "agg":
        plt.show()
    return res


if __name__ == "__main__":
    main()
