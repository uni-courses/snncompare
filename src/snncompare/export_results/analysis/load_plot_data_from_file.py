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
    """Creates a dotted boxplot.

    Args:
    :exp_config: (Exp_config), The experimental configuration, including
    various settings and parameters.
    :completed_run_configs: (List[Run_config]), A list of completed run
    configurations, each representing a specific run of the experiment.
    Returns:
    A dictionary containing data for a dotted boxplot, with radiation
    probabilities as keys and corresponding data as values.
    """
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
    """Stores the data for generating boxplots of robustness measures into a
    pickle file.

    Args:
    :filepath: (str), The path to the pickle file where the data will be
    stored.
    :robustness_plot_data: (Dict[float, Dict[str, List[float]]]), A
    dictionary containing data for generating boxplots of robustness
    measures. The keys are floating-point numbers representing different
    aspects of the robustness, and the values are dictionaries where the
    keys are strings representing plot names, and the values are lists of
    floating-point values representing data points for the corresponding
    plot.
    Returns:
    This function does not return any value; it stores the data in a pickle
    file specified by the 'filepath' argument.
    """
    with open(filepath, "wb") as handle:
        pickle.dump(
            robustness_plot_data, handle, protocol=pickle.HIGHEST_PROTOCOL
        )


@typechecked
def load_pickle_boxplot(
    *, filepath: str
) -> Dict[float, Dict[str, List[float]]]:
    """Loads data from a pickle file containing a boxplot representation of
    robustness values for different configurations.

    Args:
    :filepath: (str), The path to the pickle file that contains the boxplot
    data.
    Returns:
    A dictionary containing robustness data in the form of a boxplot. The
    outer dictionary has float keys representing some values, and each
    associated value is an inner dictionary with string keys and lists of
    floats as values.
    """
    with open(filepath, "rb") as handle:
        robustness_plot_data: Dict[
            float, Dict[str, List[float]]
        ] = pickle.load(
            handle
        )  # nosec
    return robustness_plot_data
