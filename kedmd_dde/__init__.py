"""kEDMD for delay differential equations, on PyTorch.

Kernel EDMD with a local affine surrogate, applied to delay-embedded states of scalar
and planar DDEs. The sweeps that produce the paper figures live in examples/.

    from kedmd_dde import load_dataset, run_experiment, summarize_errors

    ds = load_dataset("raw_scalar.npz", M=2)
    result = run_experiment(ds, {"n_centers": 121, "rho": 0.3,
                                 "epsilon": 1.0, "d_neighbors": 25})
    summary, = summarize_errors([result])

The method itself is in KEDMD_DDE.py; everything around it is in Utils/. Computation
runs on CUDA when a GPU is visible and on the CPU otherwise; KEDMD_DEVICE and
KEDMD_DTYPE override that.
"""

from .KEDMD_DDE import (
    Assumption3Report,
    KEDMDModel,
    build_kedmd,
    check_assumption3,
    empirical_fill_distance,
    farthest_point_sampling,
    fill_distance_at,
    fill_distance_curve,
    matched_center_configs,
    min_centers_for_fill_distance,
    pairwise_distances,
    rollout_lifted,
    uniform_grid_centers_box,
    wendland_c2,
    wendland_degree,
    wendland_matrix,
)
from .Utils import (
    DATA_DIR,
    FIGURE_DIR,
    PAPER_STYLE,
    PROJECT_ROOT,
    DelayDataset,
    as_tensor,
    available_M,
    data_path,
    dataset_from_raw,
    describe_device,
    error_curves,
    figure_path,
    get_device,
    get_dtype,
    load_dataset,
    load_metadata,
    load_raw,
    matobj_to_dict,
    plot_error_vs_time,
    plot_phase_portrait,
    plot_state_vs_time,
    print_device_info,
    raw_available_M,
    run_experiment,
    save_figure,
    set_device,
    set_dtype,
    summarize_errors,
    synchronize,
    to_numpy,
    trajectory_errors,
    use_paper_style,
)

__version__ = "0.1.0"

__all__ = [
    # device
    "get_device", "set_device", "get_dtype", "set_dtype", "describe_device",
    "print_device_info", "as_tensor", "to_numpy", "synchronize",
    # kernels
    "wendland_c2", "wendland_degree", "wendland_matrix", "pairwise_distances",
    # centers
    "farthest_point_sampling", "uniform_grid_centers_box", "empirical_fill_distance",
    "fill_distance_curve", "fill_distance_at", "min_centers_for_fill_distance",
    "matched_center_configs",
    # model
    "build_kedmd", "rollout_lifted", "KEDMDModel",
    # assumption 3
    "check_assumption3", "Assumption3Report",
    # data
    "load_dataset", "load_metadata", "available_M", "DelayDataset",
    "load_raw", "dataset_from_raw", "raw_available_M",
    "data_path", "figure_path", "matobj_to_dict",
    "PROJECT_ROOT", "DATA_DIR", "FIGURE_DIR",
    # experiments
    "run_experiment", "summarize_errors", "trajectory_errors",
    # plotting
    "use_paper_style", "PAPER_STYLE", "error_curves",
    "plot_error_vs_time", "plot_state_vs_time", "plot_phase_portrait", "save_figure",
]
