"""Every figure in the paper, and the data behind them, from one module.

Each script in ``examples/`` is a function here, taking no arguments: the settings it
ran with are module constants, one block per figure, so a variation is an edit at the
top rather than a flag::

    from kedmd_dde.Utils.pub_gen import m_vary, all_figures

    m_vary()          # writes figures/scalar_M_vary.pdf, returns the fits
    all_figures()     # all five, one window at the end

or from the command line::

    python -m kedmd_dde.Utils.pub_gen                  # every figure
    python -m kedmd_dde.Utils.pub_gen m_vary p_vary    # just these
    python -m kedmd_dde.Utils.pub_gen --data           # only the data step
    python -m kedmd_dde.Utils.pub_gen --no-show --yes  # unattended

Each figure first calls :func:`ensure_data`, which integrates a record only when the one
it needs is nowhere to be found — the records ship inside the package, so this normally
does nothing. :func:`gen_data` is the deliberate version: when a record is already in
``kedmd_dde/data/`` it does not overwrite it, but offers to write a fresh one into the
``data/`` directory beside the package instead. Integration needs ``jitcdde``, and both
ask before installing it.

A record beside the package is *shadowed* by the packaged one when it is loaded by bare
name — ``data_path`` searches ``kedmd_dde/data/`` first — so load a regenerated record
by the path ``gen_data`` returns::

    paths = gen_data()                                # say yes at the prompt
    ds = load_dataset(paths["scalar"], M=3)
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import DATA_DIR, EXTRA_DATA_DIR, data_path, load_dataset, raw_available_M
from .device import print_device_info
from .experiments import run_experiment, summarize_errors
from .generate import (
    PLANAR_DDE, SCALAR_DDE, default_time_grid, integrate_trajectories,
    sample_initial_conditions,
)
from .plotting import (
    plot_error_vs_time, plot_state_vs_time, save_figure, use_paper_style,
)

SHOW = True                   # False saves the figures without opening a window

SYSTEMS = {"scalar": SCALAR_DDE, "planar": PLANAR_DDE}
RAW_FILES = {"scalar": "raw_scalar.npz", "planar": "raw_planar.npz"}

# Per system: pool size, end of the record, and save step. The record always starts at
# -tau, the earliest time a snapshot can have a full history window. Each dt is written
# as a fraction of tau so that tau/dt is a whole number with plenty of divisors — that
# is what decides which M the record can represent (M-1 must divide tau/dt).
DEFAULTS = {
    "scalar": dict(n_traj=1100, t_end=3.1, dt=SCALAR_DDE.tau / 10),     # M = 2,3,6,11
    "planar": dict(n_traj=5100, t_end=3.2, dt=PLANAR_DDE.tau / 20),     # M = 2,3,5,6,11
}


# ============================================================== data

def _packaged(system):
    """Where the shipped record for a system lives, whether or not it is there."""
    return Path(DATA_DIR) / RAW_FILES[system]


def _resolved(system):
    """The record the loader would actually open, or None if there is none."""
    path = Path(data_path(RAW_FILES[system]))
    return path if path.exists() else None


def _ask(question, default=False, assume_yes=False):
    """Yes/no prompt. Off a terminal it takes the default rather than blocking."""
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} -> no terminal to ask on, taking "
              f"'{'yes' if default else 'no'}'")
        return default
    reply = input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not reply else reply in ("y", "yes")


def _ensure_jitcdde(assume_yes=False):
    """Import jitcdde, offering to pip install it first. Raises if it stays missing."""
    if importlib.util.find_spec("jitcdde") is not None:
        return
    print("jitcdde integrates the DDEs and is not installed.")
    if not _ask("Install it now with pip?", default=False, assume_yes=assume_yes):
        raise SystemExit(
            "generating data needs jitcdde: pip install jitcdde "
            "(a C compiler must be on PATH)"
        )
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jitcdde"])
    importlib.invalidate_caches()
    if importlib.util.find_spec("jitcdde") is None:
        raise SystemExit("pip finished but jitcdde still does not import")


def generate(system, n_traj=None, t_start=None, t_end=None, dt=None, seed=1,
             float32=False, out=DATA_DIR, name=None, verbose=False):
    """Integrate one DDESystem and write its raw record. Returns (path, record).

    The record holds the solution once — values (nTraj, T, dim), times, tau, dt, dim,
    x0 and the settings — not the delay embedding, which happens at load time.
    """
    d = DEFAULTS[system.name]
    n_traj = d["n_traj"] if n_traj is None else n_traj
    t_end = d["t_end"] if t_end is None else t_end
    dt = d["dt"] if dt is None else dt
    t_start = -system.tau if t_start is None else t_start

    if t_start > -system.tau + 1e-12:
        raise SystemExit(
            f"t_start must be at most -tau = {-system.tau:g} so the first snapshot has "
            f"a full history window"
        )

    times = default_time_grid(t_start, t_end, dt)
    x0_all = sample_initial_conditions(system, n_traj, seed)
    bank = integrate_trajectories(system, x0_all, times, verbose=verbose)

    record = {
        "values": bank.values.astype(np.float32 if float32 else np.float64),
        "times": times,
        "dt": np.float64(dt),
        "tau": np.float64(system.tau),
        "dim": np.int64(system.dim),
        "x0": bank.x0,
        "seed": np.int64(seed),
        "system": system.description,
        "generator": "jitcdde",
    }

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name or f"raw_{system.name}.npz")
    np.savez_compressed(path, **record)
    return path, record


def _report(path, record, quiet=False):
    """Print what was written and which M it can represent."""
    M_ok = raw_available_M(record)
    if not quiet:
        size = path.stat().st_size / 1024 ** 2
        print(f"  wrote {path} ({size:.1f} MiB) — M available: "
              f"{', '.join(map(str, M_ok)) or 'none'}")
    if not M_ok:
        # No M can be built from this record and load_dataset would reject each one in
        # turn, so say which of the two causes it is.
        dt_used, tau = float(record["dt"]), float(record["tau"])
        lag = tau / dt_used
        if abs(lag - round(lag)) > 1e-6 * max(1.0, lag):
            why = (f"dt={dt_used:g} does not divide tau={tau:g}, so no delay window "
                   f"lands on saved samples")
        else:
            why = (f"only {len(record['times'])} samples per trajectory, and the delay "
                   f"alone spans {round(lag)} of them — raise t_end")
        print(f"  warning: this record supports no M: {why}", flush=True)


def ensure_data(*systems, assume_yes=False, verbose=False, quiet=False):
    """Make sure the records the figures need exist, generating only what is missing.

    Nothing is asked and nothing is written when a record is already there, which is the
    normal case since both ship inside the package. A record that has gone missing is
    regenerated into ``kedmd_dde/data/`` with the settings in DEFAULTS, which reproduces
    the shipped one bit for bit. Returns {system: path}.
    """
    names = list(systems) or sorted(SYSTEMS)
    paths = {}
    todo = []
    for name in names:
        found = _resolved(name)
        if found is None:
            todo.append(name)
        else:
            paths[name] = found

    if not todo:
        return paths

    if not quiet:
        print(f"missing raw {'record' if len(todo) == 1 else 'records'}: "
              f"{', '.join(RAW_FILES[n] for n in todo)} — generating into {DATA_DIR}")
    _ensure_jitcdde(assume_yes=assume_yes)
    for name in todo:
        if not quiet:
            print(f"Generating raw data for the {name} DDE ...", flush=True)
        path, record = generate(SYSTEMS[name], out=DATA_DIR, verbose=verbose)
        _report(path, record, quiet=quiet)
        paths[name] = path
    return paths


def gen_data(system="all", n_traj=None, t_start=None, t_end=None, dt=None, seed=1,
             float32=False, out=None, name=None, verbose=False, assume_yes=False,
             quiet=False):
    """Generate the raw records, leaving anything already in the package alone.

    A system whose record is missing from ``kedmd_dde/data/`` is generated into it. A
    system whose record is there is never overwritten; instead the prompt offers to
    integrate a fresh one into ``data/`` beside the package, which is where a record
    made with settings of your own belongs. Declining leaves that system untouched.

    out overrides both destinations and skips that prompt — pass it to write anywhere.
    n_traj, t_start, t_end, dt, seed and float32 override DEFAULTS for every system
    generated. name renames the file, so it applies to a single system and writes beside
    the package without asking: under a name of your own it is not the shipped record.
    assume_yes answers every prompt, including the jitcdde install, with yes.

    Returns {system: path}: what was written, or what was already there.
    """
    names = sorted(SYSTEMS) if system == "all" else [system]
    unknown = [n for n in names if n not in SYSTEMS]
    if unknown:
        raise ValueError(f"unknown system(s) {unknown}; choose from {sorted(SYSTEMS)}")
    if name and len(names) > 1:
        raise ValueError("name applies to a single system, not to 'all'")

    plan, paths = {}, {}
    for n in names:
        if out is not None:
            plan[n] = Path(out)
            continue
        if name:
            # A record under a name of your own is not the shipped one, so it belongs
            # beside the package rather than in it, and there is nothing to ask about.
            plan[n] = Path(EXTRA_DATA_DIR)
            continue
        packaged = _packaged(n)
        if not packaged.exists():
            plan[n] = Path(DATA_DIR)
            continue
        print(f"{RAW_FILES[n]} is already in the package at {packaged.parent}.")
        if _ask(f"Generate a fresh {n} record into {EXTRA_DATA_DIR} instead "
                f"(the packaged one is kept)?", default=False, assume_yes=assume_yes):
            plan[n] = Path(EXTRA_DATA_DIR)
        else:
            paths[n] = packaged

    if plan:
        _ensure_jitcdde(assume_yes=assume_yes)
    for n, dest in plan.items():
        if not quiet:
            print(f"Generating raw data for the {n} DDE ...", flush=True)
        path, record = generate(
            SYSTEMS[n], n_traj=n_traj, t_start=t_start, t_end=t_end, dt=dt, seed=seed,
            float32=float32, out=dest, name=name, verbose=verbose and not quiet,
        )
        _report(path, record, quiet=quiet)
        paths[n] = path
        if dest == Path(EXTRA_DATA_DIR) and not quiet:
            # data_path searches the package first, so the bare name still opens the
            # shipped record; this copy has to be asked for by path.
            print(f"  note: load it with load_dataset(r\"{path}\", M=...) — the bare "
                  f"name '{RAW_FILES[n]}' still resolves to the packaged record")
    return paths


# ============================================================== figures

def _finish(save_name):
    """Save the current figure, report the path, and return it."""
    plt.tight_layout()
    path = save_figure(save_name)
    print("saved:", path)
    return path


def _maybe_show():
    if SHOW:
        plt.show()


def _setup():
    print_device_info(brief=True)
    use_paper_style()


# ---- M_vary: scalar DDE, prediction error against time across resolutions M -------
# Each M gets the fewest centers reaching the fill distance the smallest M attains with
# REFERENCE_CENTERS, so the curves compare the lift and not the sampling density.
M_VARY_DATA = "raw_scalar.npz"
M_VARY_SAVE = "scalar_M_vary.pdf"
M_VARY_M_VALUES = [2, 3, 6]         # M - 1 must divide tau/dt
M_VARY_REFERENCE_CENTERS = 1500     # centers for the smallest M; the rest match it
M_VARY_MAX_CENTERS = 4096           # search budget when matching
M_VARY_RHO = 0.6
M_VARY_BASE_CFG = {"epsilon": 1.0, "d_neighbors": 200}


def m_vary():
    """Scalar DDE: prediction error against time across history resolutions M."""
    from ..KEDMD_DDE import matched_center_configs   # KEDMD_DDE imports Utils

    ensure_data("scalar")
    _setup()

    datasets = {M: load_dataset(M_VARY_DATA, M=M) for M in M_VARY_M_VALUES}
    configs = matched_center_configs(
        datasets, target=(M_VARY_M_VALUES[0], M_VARY_REFERENCE_CENTERS),
        base_cfg=M_VARY_BASE_CFG, max_centers=M_VARY_MAX_CENTERS, verbose=False,
    )

    results = []
    for M in M_VARY_M_VALUES:
        res = run_experiment(datasets[M], dict(configs[M], rho=M_VARY_RHO),
                             verbose=False, assumption3=True)
        results.append(res)
        holds = res["info"]["assumption3"]["holds"]
        print(f"M = {M}, p = {res['info']['num_centers']}, "
              f"h_X = {res['info']['fill_distance']:.6e}, "
              f"Assumption 3: {'satisfied' if holds else 'VIOLATED'}")

    plt.figure()
    plot_error_vs_time(
        results, summarize_errors(results),
        label_fn=lambda res: rf"$M={res['M']}$",
        dt=datasets[M_VARY_M_VALUES[0]].dt,
    )
    _finish(M_VARY_SAVE)
    _maybe_show()
    return results


# ---- p_vary: scalar DDE, prediction error against time across center counts p -----
# Each curve is labelled with p; the fill distance h and the Assumption 3 verdict at
# each p go to stdout.
P_VARY_DATA = "raw_scalar.npz"
P_VARY_SAVE = "p_vary.pdf"
P_VARY_M = 3                        # M - 1 must divide tau/dt
P_VARY_P_VALUES = [10, 50, 100, 500, 1500]
P_VARY_BASE_CFG = {"rho": 0.6, "epsilon": 1.0, "d_neighbors": 25}


def p_vary():
    """Scalar DDE: prediction error against time across kernel-center counts p."""
    ensure_data("scalar")
    _setup()

    dataset = load_dataset(P_VARY_DATA, M=P_VARY_M)

    results = []
    for p in P_VARY_P_VALUES:
        res = run_experiment(dataset, dict(P_VARY_BASE_CFG, n_centers=p),
                             verbose=False, assumption3=True)
        results.append(res)
        holds = res["info"]["assumption3"]["holds"]
        print(f"p = {p}: h = {res['info']['fill_distance']:.3e}, "
              f"Assumption 3: {'satisfied' if holds else 'VIOLATED'}")

    plt.figure()
    plot_error_vs_time(
        results, summarize_errors(results),
        label_fn=lambda res: rf"$p={res['cfg']['n_centers']}$",
        dt=dataset.dt,
    )
    _finish(P_VARY_SAVE)
    _maybe_show()
    return results


# ---- rho_vary: scalar DDE, prediction error against time across local-fit radii ----
# d_neighbors is None so that every point in B(c, rho) enters the local fit and rho
# alone sets the locality; with a cap the 25 nearest neighbours sit far inside even the
# smallest ball and the sweep measures nothing. Assumption 3 is checked at each radius.
RHO_VARY_DATA = "raw_scalar.npz"
RHO_VARY_SAVE = "rho_vary.pdf"
RHO_VARY_M = 3                      # M - 1 must divide tau/dt
RHO_VARY_RHO_VALUES = [0.9, 0.6, 0.3, 0.1]
RHO_VARY_BASE_CFG = {"n_centers": 500, "epsilon": 1.0, "d_neighbors": None}
RHO_VARY_RANK_RECOVERY = "expand"   # grow a rank-deficient neighbourhood point by point


def rho_vary():
    """Scalar DDE: prediction error against time across local-fit radii rho."""
    ensure_data("scalar")
    _setup()

    dataset = load_dataset(RHO_VARY_DATA, M=RHO_VARY_M)

    results = []
    for rho in RHO_VARY_RHO_VALUES:
        results.append(run_experiment(
            dataset, dict(RHO_VARY_BASE_CFG, rho=rho), verbose=False,
            rank_recovery=RHO_VARY_RANK_RECOVERY, assumption3=True,
        ))

    # p is fixed, so the centers and the fill distance are the same for every rho.
    print(f"p = {RHO_VARY_BASE_CFG['n_centers']}, "
          f"h = {results[0]['info']['fill_distance']:.3e}")
    print(f"{'rho':>7} {'Assumption 3':>14}")
    for res in results:
        holds = res["info"]["assumption3"]["holds"]
        print(f"{res['cfg']['rho']:>7g} {'satisfied' if holds else 'VIOLATED':>14}")

    plt.figure()
    plot_error_vs_time(
        results, summarize_errors(results),
        label_fn=lambda res: rf"$\rho={res['cfg']['rho']:g}$",
        dt=dataset.dt,
    )
    _finish(RHO_VARY_SAVE)
    _maybe_show()
    return results


# ---- scalar_phase: true vs predicted x(t) for a few trajectories, with RMSE --------
# One fit, rolled out over the whole test set. The RMSE is normalised by the standard
# deviation of the true signal as well as by its RMS: the signal sits well away from
# zero, so RMS carries the offset and gives the flattering number.
SCALAR_PHASE_DATA = "raw_scalar.npz"
SCALAR_PHASE_SAVE = "scalar_phase_new.pdf"
SCALAR_PHASE_M = 3                  # M - 1 must divide tau/dt
SCALAR_PHASE_CFG = {"n_centers": 100, "rho": 0.6, "epsilon": 1.0, "d_neighbors": 25}
SCALAR_PHASE_N_PICK = 6             # trajectories drawn; the seed chooses which
SCALAR_PHASE_PICK_SEED = 56


def scalar_phase():
    """Scalar DDE: true vs predicted x(t) for a few test trajectories, plus RMSE."""
    ensure_data("scalar")
    _setup()

    dataset = load_dataset(SCALAR_PHASE_DATA, M=SCALAR_PHASE_M)
    result = run_experiment(dataset, SCALAR_PHASE_CFG, verbose=False, assumption3=True)

    current = dataset.current_state_slice
    err = result["all_pred"][:, :, current] - result["all_true"][:, :, current]
    ref = result["all_true"][:, :, current]
    rmse = err.pow(2).mean().sqrt()

    a3 = result["info"]["assumption3"]
    print(f"h = {result['info']['fill_distance']:.3e}, "
          f"Assumption 3: {'ok' if a3['holds'] else 'FAIL'} "
          f"({a3['n_underfilled']} thin balls, rho_eff {a3['rho_eff']:.3e})")
    print(f"RMSE = {float(rmse):.4e}  "
          f"({float(rmse / ref.pow(2).mean().sqrt()):.2%} of RMS(true), "
          f"{float(rmse / ref.std(unbiased=False)):.2%} of std(true))")

    gen = torch.Generator().manual_seed(SCALAR_PHASE_PICK_SEED)
    traj = torch.randperm(dataset.n_test, generator=gen)[:SCALAR_PHASE_N_PICK].tolist()

    plt.figure()
    plot_state_vs_time(result, traj, component=current.start, dt=dataset.dt,
                       ylabel=r"$x(t)$")
    _finish(SCALAR_PHASE_SAVE)
    _maybe_show()
    return result


# ---- planar_phase: true vs predicted x1(t) and x2(t), with aggregated RMSE ---------
# One fit, rolled out over the whole test set, one figure per component. The Wendland
# degree follows nM, so the kernel at nM = 4 is narrower than the scalar one and epsilon
# has to grow with it; rho = 0.5 is the smallest radius leaving no thin ball at p = 1200.
PLANAR_PHASE_DATA = "raw_planar.npz"
PLANAR_PHASE_SAVE_STEM = "planar_phase_new"
PLANAR_PHASE_M = 3                  # M - 1 must divide tau/dt
PLANAR_PHASE_CFG = {"n_centers": 1500, "rho": 0.6, "epsilon": 1.0, "d_neighbors": 500}
PLANAR_PHASE_N_PICK = 6             # trajectories drawn; the seed chooses which
PLANAR_PHASE_PICK_SEED = 56
PLANAR_PHASE_LABELS = (r"$x_1(t)$", r"$x_2(t)$")


def planar_phase():
    """Planar DDE: true vs predicted x1(t) and x2(t), with aggregated RMSE."""
    ensure_data("planar")
    _setup()

    dataset = load_dataset(PLANAR_PHASE_DATA, M=PLANAR_PHASE_M)
    result = run_experiment(dataset, PLANAR_PHASE_CFG, verbose=False, assumption3=True)

    current = dataset.current_state_slice
    err = result["all_pred"][:, :, current] - result["all_true"][:, :, current]
    ref = result["all_true"][:, :, current]

    a3 = result["info"]["assumption3"]
    print(f"h = {result['info']['fill_distance']:.3e}, "
          f"kernel degree = {result['info']['kernel_degree']}, "
          f"Assumption 3: {'ok' if a3['holds'] else 'FAIL'} "
          f"({a3['n_underfilled']} thin balls, rho_eff {a3['rho_eff']:.3e})")
    print(f"{'component':>10} {'RMSE':>12} {'NRMSE':>9}")
    for i, label in enumerate(PLANAR_PHASE_LABELS):
        rmse = err[:, :, i].pow(2).mean().sqrt()
        print(f"{label.strip('$').replace('(t)', ''):>10} {float(rmse):>12.4e} "
              f"{float(rmse / ref[:, :, i].pow(2).mean().sqrt()):>8.2%}")
    rmse = err.pow(2).mean().sqrt()
    print(f"{'both':>10} {float(rmse):>12.4e} "
          f"{float(rmse / ref.pow(2).mean().sqrt()):>8.2%}")

    gen = torch.Generator().manual_seed(PLANAR_PHASE_PICK_SEED)
    traj = torch.randperm(dataset.n_test, generator=gen)[:PLANAR_PHASE_N_PICK].tolist()

    for offset, ylabel in enumerate(PLANAR_PHASE_LABELS):
        plt.figure()
        plot_state_vs_time(result, traj, component=current.start + offset,
                           dt=dataset.dt, ylabel=ylabel)
        _finish(f"{PLANAR_PHASE_SAVE_STEM}_x{offset + 1}.pdf")
    _maybe_show()
    return result


FIGURES = {
    "m_vary": m_vary,
    "p_vary": p_vary,
    "rho_vary": rho_vary,
    "scalar_phase": scalar_phase,
    "planar_phase": planar_phase,
}


def all_figures(names=None):
    """Run every figure in order and return {name: what that function returned}.

    The windows are held back until the end so a full run is not five stops.
    """
    global SHOW
    names = list(FIGURES) if names is None else list(names)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise ValueError(f"unknown figure(s) {unknown}; choose from {list(FIGURES)}")

    wanted, SHOW = SHOW, False
    try:
        out = {}
        for n in names:
            print(f"\n===== {n} =====")
            out[n] = FIGURES[n]()
    finally:
        SHOW = wanted
    if wanted:
        plt.show()
    return out


def main(argv=None):
    global SHOW
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No choices=: argparse validates the empty default against them, so "all of them"
    # would be rejected. all_figures names the bad one instead.
    ap.add_argument("figures", nargs="*", default=[],
                    metavar="{" + ",".join(FIGURES) + "}",
                    help="figures to make (default: all of them)")
    ap.add_argument("--data", action="store_true",
                    help="only run gen_data, which prompts before writing anything")
    ap.add_argument("--system", choices=sorted(SYSTEMS) + ["all"], default="all",
                    help="with --data: which system to generate (default: all)")
    ap.add_argument("--no-show", action="store_true",
                    help="save the figures without opening a window")
    ap.add_argument("--yes", action="store_true",
                    help="answer every prompt with yes, including the jitcdde install")
    ap.add_argument("--verbose", action="store_true",
                    help="with --data: show the integrator's per-trajectory progress")
    args = ap.parse_args(argv)

    if args.no_show:
        SHOW = False
    try:
        if args.data:
            return gen_data(system=args.system, verbose=args.verbose,
                            assume_yes=args.yes)
        return all_figures(args.figures or None)
    except ValueError as exc:                      # a name that is not on the menu
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
