"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""


# Take in exp_config or run_configs
# If exp_config, get run_configs
import os
import pickle  # nosec
from pathlib import Path
from typing import Dict, List

from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.export_results.analysis.data_filtering import (
    Boxplot_x_val,
    boxplot_data_to_y_series,
    get_boxplot_datapoints,
)
from snncompare.helper import file_exists
from snncompare.run_config.Run_config import Run_config


@typechecked
def load_boxplot_data(
    exp_config: Exp_config,
    completed_run_configs: List[Run_config],
) -> Dict[float, Dict[str, List[float]]]:
    """Creates a dotted boxplot."""
    pickle_path: str = "results/pickles"
    if not Path(pickle_path):
        os.mkdir(pickle_path)
    robustness_plot_data: Dict[float, Dict[str, List[float]]] = {}
    filepath: str = f"{pickle_path}/{exp_config.unique_id}.pkl"
    if file_exists(filepath=filepath):
        robustness_plot_data = load_pickle_boxplot(filepath=filepath)
    else:
        for rad_setting in reversed(exp_config.radiations):
            # Get run configs belonging to this radiation type/level.
            wanted_run_configs: List[Run_config] = []
            for run_config in completed_run_configs:
                if run_config.radiation == rad_setting:
                    wanted_run_configs.append(run_config)

            # Get results per line.
            boxplot_data: Dict[
                str, Dict[int, Boxplot_x_val]
            ] = get_boxplot_datapoints(
                adaptations=exp_config.adaptations,
                wanted_run_configs=wanted_run_configs,
                seeds=exp_config.seeds,
            )

            y_series: Dict[str, List[float]] = boxplot_data_to_y_series(
                boxplot_data=boxplot_data
            )
            robustness_plot_data[rad_setting.probability_per_t] = y_series
        store_pickle_boxplot(
            filepath=filepath, robustness_plot_data=robustness_plot_data
        )

    return robustness_plot_data


@typechecked
def store_pickle_boxplot(
    *, filepath: str, robustness_plot_data: Dict[float, Dict[str, List[float]]]
) -> None:
    """Stores run_config list into pickle file."""
    with open(filepath, "wb") as handle:
        pickle.dump(
            robustness_plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL
        )


@typechecked
def load_pickle_boxplot(
    *, filepath: str
) -> Dict[float, Dict[str, List[float]]]:
    """Stores run_config list into pickle file."""
    with open(filepath, "rb") as handle:
        robustness_plot_data: Dict[
            float, Dict[str, List[float]]
        ] = pickle.load(
            handle
        )  # nosec
    return robustness_plot_data
