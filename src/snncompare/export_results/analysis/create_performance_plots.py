"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""

# Take in exp_config or run_configs
# If exp_config, get run_configs
from typing import Dict, List, Tuple

from easyplot.box_plot.box_plot import create_box_plot
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.exp_config.run_config.Run_config import Run_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_results.load_json_to_nx_graph import (
    load_verified_json_graphs_from_json,
)
from snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from snncompare.helper import generate_run_configs
from snncompare.import_results.check_completed_stages import (
    has_outputted_stage,
)


# pylint: disable = R0903
class Boxplot_x_serie:
    """Stores an x_serie for the boxplot."""

    @typechecked
    def __init__(
        self,
        correct_results: int,
        wrong_results: int,
    ) -> None:

        self.correct_results: int = correct_results
        self.wrong_results: int = wrong_results


# pylint: disable = R0903
class Boxplot_data:
    """Stores the data that is to be exported into a box plot."""

    @typechecked
    def __init__(self, x_series: List[Boxplot_x_serie]) -> None:
        self.x_series: List[Boxplot_x_serie] = x_series


@typechecked
def create_performance_plots(exp_config: Exp_config) -> None:
    """Ensures all performance boxplots are created."""

    # Determine which lines are plotted, e.g.
    # which algorithm with/without adaptation.

    # Get run_configs
    run_configs = generate_run_configs(exp_config, specific_run_config=None)

    # Get list of required run_configs.
    (
        completed_run_configs,
        missing_run_configs,
    ) = get_completed_and_missing_run_configs(run_configs)

    # Verify all json results are available.

    # If not, create a list of run_configs that still need to be completed.
    # prompt user whether user wants to complete these run_configs.
    for missing_run_config in missing_run_configs:
        # Execute those run_configs
        Experiment_runner(
            exp_config=exp_config,
            specific_run_config=missing_run_config,
            perform_run=True,
        )
        # Terminate code.

    # Get json data per run_config.
    run_config_nx_graphs = get_json_data(completed_run_configs)

    # Get results per line.
    boxplot_data: Dict[str, Boxplot_x_serie] = get_boxplot_datapoints(
        run_config_nx_graphs=run_config_nx_graphs
    )

    y_series = boxplot_data_to_y_series(boxplot_data)

    # Generate line plots
    # Generate box plots.
    create_box_plot(
        extensions=["png"],
        filename="boxplot",
        legendPosition=0,
        output_dir="latex/Images",
        x_axis_label="x-axis label [units]",
        y_axis_label="y-axis label [units]",
        y_series=y_series,
    )

    # Allow user to select subset of parameter ranges.
    # (This is done in the experiment setting)
    # graph_sizes 10-30
    # m_values 0 to 10
    # adaptation 0-4
    # radiation 0,0.1,0.25

    # Get results per data

    # Create dummy box plots.

    # Create dummy line plots.


@typechecked
def get_json_data(run_configs: List[Run_config]) -> Dict:
    """Loads the data from the relevant .json dicts and returns it."""
    run_config_nx_graphs: Dict[Run_config, Dict] = {}
    for run_config in run_configs:
        run_config_nx_graphs[run_config] = load_verified_json_graphs_from_json(
            run_config=run_config,
            expected_stages=[1, 2, 4],
        )

    # TODO: write logif if file does not exist.
    return run_config_nx_graphs


@typechecked
def get_completed_and_missing_run_configs(
    run_configs: List[Run_config],
) -> Tuple[List[Run_config], List[Run_config]]:
    """Returns the run configs that still need to be ran."""
    missing_run_configs: List[Run_config] = []
    completed_run_configs: List[Run_config] = []
    for run_config in run_configs:
        if not has_outputted_stage(
            run_config=run_config,
            stage_index=4,
            to_run={
                "stage_1": True,
                "stage_2": True,
                "stage_3": False,
                "stage_4": True,
            },
        ):
            missing_run_configs.append(run_config)
        else:
            completed_run_configs.append(run_config)
    print(f"Want:{len(run_configs)}, missing:{len(missing_run_configs)}")
    return completed_run_configs, missing_run_configs


@typechecked
def get_boxplot_datapoints(
    run_config_nx_graphs: Dict,
) -> Dict[str, Boxplot_x_serie]:
    """Returns the run configs that still need to be ran."""

    boxplot_data: Dict[str, Boxplot_x_serie] = get_mdsa_boxplot_data()
    # Create x-axis categories (no redundancy, n-redundancy).
    for run_config, graphs_dict in run_config_nx_graphs.items():
        # Get the results per x-axis category per graph type.
        for algo_name in run_config.algorithm.keys():
            if algo_name == "MDSA":

                graph_names = get_expected_stage_1_graph_names(run_config)
                for graph_name in graph_names:
                    if graph_name != "input_graph":
                        add_graph_scores(
                            boxplot_data=boxplot_data,
                            graph_type=graph_name,
                            result=graphs_dict[graph_name]["graph"]["results"],
                        )

    # return data.
    return boxplot_data


@typechecked
def add_graph_scores(
    boxplot_data: Dict[str, Boxplot_x_serie], graph_type: str, result: Dict
) -> None:
    """Adds the scores for the graphs.."""
    if result:
        boxplot_data[graph_type].correct_results += 1
    else:
        boxplot_data[graph_type].wrong_results += 1
        print(f"{graph_type}, {boxplot_data[graph_type].__dict__}")


@typechecked
def get_mdsa_boxplot_data() -> Dict[str, Boxplot_x_serie]:
    """Creates the boxplot data objects for the MDSA algorithm."""
    boxplot_data: Dict[str, Boxplot_x_serie] = {
        "snn_algo_graph": Boxplot_x_serie(
            correct_results=0,
            wrong_results=0,
        ),
        "adapted_snn_graph": Boxplot_x_serie(
            correct_results=0,
            wrong_results=0,
        ),
        "rad_snn_algo_graph": Boxplot_x_serie(
            correct_results=0,
            wrong_results=0,
        ),
        "rad_adapted_snn_graph": Boxplot_x_serie(
            correct_results=0,
            wrong_results=0,
        ),
    }
    return boxplot_data


@typechecked
def boxplot_data_to_y_series(
    boxplot_data: Dict[str, Boxplot_x_serie]
) -> Dict[str, List[float]]:
    """Converts boxplot_data into x_labels and y_series for boxplot plotting.

    TODO: do this in Boxplot_x_series itself.
    """
    data: Dict[str, List[float]] = {}
    # Initialise dataseries.
    for name in boxplot_data.keys():
        data[name] = []

    for name, y_vals in boxplot_data.items():
        data[name].append(
            y_vals.correct_results
            / (y_vals.correct_results + y_vals.wrong_results)
        )
    return data
