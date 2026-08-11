"""Everything around the estimator: device policy, data, generation, plotting, sweeps."""

from .data import (
    DATA_DIR,
    FIGURE_DIR,
    PROJECT_ROOT,
    DelayDataset,
    available_M,
    data_path,
    dataset_from_raw,
    figure_path,
    load_dataset,
    load_metadata,
    load_raw,
    matobj_to_dict,
    raw_available_M,
)
from .device import (
    as_tensor,
    describe_device,
    get_device,
    get_dtype,
    print_device_info,
    set_device,
    set_dtype,
    synchronize,
    to_numpy,
)
from .experiments import run_experiment, summarize_errors, trajectory_errors
from .plotting import (
    PAPER_STYLE,
    error_curves,
    plot_error_vs_time,
    plot_phase_portrait,
    plot_state_vs_time,
    save_figure,
    use_paper_style,
)
