"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""

# Take in exp_config or run_configs
# If exp_config, get run_configs
import copy
import pickle  # nosec
from typing import Dict, List, Tuple

from simplt.box_plot.box_plot import create_box_plot
from typeguard import typechecked

from snncompare.create_configs import get_adaptations_or_radiations
from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_results.load_json_to_nx_graph import (
    load_verified_json_graphs_from_json,
)
from snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from snncompare.import_results.check_completed_stages import (
    has_outputted_stage_jsons,
)
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config


# pylint: disable = R0903
class Boxplot_x_val:
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
    def __init__(self, x_series: List[Boxplot_x_val]) -> None:
        self.x_series: List[Boxplot_x_val] = x_series


# pylint: disable=R0912
# pylint: disable=R0914
@typechecked
def create_performance_plots(
    *,
    completed_run_configs: List[Run_config],
    exp_config: Exp_config,
    output_config: Output_config,
) -> None:
    """Ensures all performance boxplots are created.

    Loops through the wanted run configs, then gets the radiations that are
    tested.
    Then gets the redundancy levels per radiation that are tested.
    Per radiation, creates a separate plot. Per plot it:
      - creates 4 columns:
        - no adaptation, no radiation
        - adaptation, no radiation
        - no adaptation, radiation
        - adaptation, radiation
    Per column, it contains a list of seeds, each seed has 2 counts.
        - the nr of correct results.
        - the nr of incorrect results.
    the boxplot then turns this data into a boxplot.

    So to get this data,
    - per run config
      -loops through the seeds,
        - per radiation level,
          - per column
            Get the result of a run config and store it in the boxplot data.
    """
    print(output_config.extra_storing_config)

    count: int = 0

    _, radiations = retry_get_boxplot_data(
        exp_config=exp_config, run_configs=completed_run_configs
    )
    for radiation in radiations:
        for radiation_name, radiation_value in reversed(radiation.items()):
            count = count + 1
            for adaptation in exp_config.adaptations:
                # Get run configs belonging to this radiation type/level.
                wanted_run_configs: List[Run_config] = []
                for run_config in completed_run_configs:
                    if run_config.radiation == radiation:
                        wanted_run_configs.append(run_config)

                # Get results per line.
                boxplot_data: Dict[
                    str, Dict[int, Boxplot_x_val]
                ] = get_boxplot_datapoints(
                    adaptations=exp_config.adaptations,
                    wanted_run_configs=wanted_run_configs,
                    seeds=exp_config.seeds,
                )

                y_series = boxplot_data_to_y_series(boxplot_data=boxplot_data)

                # Generate box plots.
                create_box_plot(
                    extensions=["png"],
                    filename=(
                        f"{count}_boxplot_{radiation_name}="
                        + f"{radiation_value}_{adaptation}"
                    ),
                    legendPosition=0,
                    output_dir="latex/Images",
                    x_axis_label="x-axis label [units]",
                    y_axis_label="y-axis label [units]",
                    y_series=y_series,
                    title=f"{radiation_name}={radiation_value}",
                    x_axis_label_rotation=45,
                )


@typechecked
def retry_get_boxplot_data(
    *,
    exp_config: Exp_config,
    run_configs: List[Run_config],
) -> Tuple[List, List]:
    """So to get this data,

    - per run config
      -loops through the seeds,
        - per radiation level,
          - per column
            Get the result of a run config and store it in the boxplot data.
    """

    adaptations: List = []
    radiations: List = []
    seeds: List = []
    for run_config in run_configs:
        for adaptation in get_adaptations_or_radiations(
            adaptations_or_radiations=exp_config.adaptations,
        ):
            adaptations.append(adaptation)
        for radiation in get_adaptations_or_radiations(
            adaptations_or_radiations=exp_config.radiations,
        ):
            radiations.append(radiation)
        for seed in exp_config.seeds:
            seeds.append(seed)

    # TODO: assert each combination of radiation, adaptation and seed exists
    # in the run_configs.

    # Create empty boxplot data.

    for radiation in radiations:
        # Create 4 boxplot columns.

        for adaptation in adaptations:
            # Per boxplot column count results.
            for seed in seeds:
                for run_config in run_configs:
                    if (
                        run_config.adaptation == adaptation
                        and run_config.radiation == radiation
                        and run_config.seed == seed
                    ):
                        pass
    return adaptations, radiations


@typechecked
def get_completed_and_missing_run_configs(
    *,
    run_configs: List[Run_config],
) -> Tuple[List[Run_config], List[Run_config]]:
    """Returns the run configs that still need to be ran."""
    missing_run_configs: List[Run_config] = []
    completed_run_configs: List[Run_config] = []
    for run_config in run_configs:
        if not has_outputted_stage_jsons(
            expected_stages=[1, 2, 4],  # Assume results have been created.
            run_config=run_config,
            stage_index=4,
        ):
            missing_run_configs.append(run_config)
        else:
            completed_run_configs.append(run_config)
    if len(missing_run_configs) > 0:
        print(f"Want:{len(run_configs)}, missing:{len(missing_run_configs)}")
    return completed_run_configs, missing_run_configs


@typechecked
def get_boxplot_datapoints(
    *,
    adaptations: Dict[str, List[int]],
    wanted_run_configs: List[Run_config],
    # run_config_nx_graphs: Dict,
    seeds: List[int],
) -> Dict[str, Dict[int, Boxplot_x_val]]:
    """Returns the run configs that still need to be ran."""

    boxplot_data: Dict[str, Dict[int, Boxplot_x_val]] = get_mdsa_boxplot_data(
        adaptations=adaptations,
        graph_names=get_expected_stage_1_graph_names(
            run_config=wanted_run_configs[0]
        ),
        seeds=seeds,
    )

    # Create x-axis categories (no redundancy, n-redundancy).
    # for run_config, graphs_dict in run_config_nx_graphs.items():
    for wanted_run_config in wanted_run_configs:
        # Get the results per x-axis category per graph type.
        for algo_name in wanted_run_config.algorithm.keys():
            if algo_name == "MDSA":
                graph_names = get_expected_stage_1_graph_names(
                    run_config=wanted_run_config
                )
                graphs_dict: Dict = load_verified_json_graphs_from_json(
                    run_config=wanted_run_config,
                    expected_stages=[1, 2, 4],
                )

                x_labels, results = get_x_labels(
                    adaptations=adaptations,
                    graphs_dict=graphs_dict,
                    graph_names=graph_names,
                )
                for x_label in x_labels:
                    add_graph_scores(
                        boxplot_data=boxplot_data,
                        x_label=x_label,
                        result=results[x_label],
                        seed=wanted_run_config.seed,
                    )

    return boxplot_data


@typechecked
def get_x_labels(
    *,
    adaptations: Dict[str, List[int]],
    graphs_dict: Dict,
    graph_names: List[str],
) -> Tuple[List[str], Dict]:
    """Returns the x-axis labels per dataserie/boxplot."""
    x_labels: List[str] = []
    results = {}
    for graph_name in graph_names:
        if graph_name == "rad_adapted_snn_graph":
            for name, values in adaptations.items():
                for value in values:
                    x_labels.append(f"{name}:{value}")
                    results[x_labels[-1]] = graphs_dict[graph_name]["graph"][
                        "results"
                    ]
        elif graph_name != "input_graph":
            x_labels.append(graph_name)
            results[x_labels[-1]] = graphs_dict[graph_name]["graph"]["results"]
    return x_labels, results


@typechecked
def add_graph_scores(
    *,
    boxplot_data: Dict[str, Dict[int, Boxplot_x_val]],
    x_label: str,
    result: Dict,
    seed: int,
) -> None:
    """Adds the scores for the graphs.."""
    if result["passed"]:
        boxplot_data[x_label][seed].correct_results += 1
    else:
        boxplot_data[x_label][seed].wrong_results += 1


@typechecked
def get_mdsa_boxplot_data(
    *,
    adaptations: Dict[str, List[int]],
    graph_names: List[str],
    seeds: List[int],
) -> Dict[str, Dict[int, Boxplot_x_val]]:
    """Creates the empty boxplot data objects for the MDSA algorithm."""

    # Create an x-series object, per seed store the scores of the
    # runs with the different graph sizes and m_vals.
    x_series_data: Dict[int, Boxplot_x_val] = {}
    for seed in seeds:
        x_series_data[seed] = Boxplot_x_val(
            correct_results=0,
            wrong_results=0,
        )

    # Per boxplot store multiple x-series:
    # - snn_algo_graph
    # - rad_snn_algo_graph
    # - adapted_snn_algo_graph (should all be 100%)
    # - rad_adapted_snn_algo_graph (per adaptation type, show its score)
    boxplot_data: Dict[str, Dict[int, Boxplot_x_val]] = {}
    for graph_name in graph_names:
        if graph_name == "rad_adapted_snn_graph":
            for name, values in adaptations.items():
                for value in values:
                    # Create multiple columns in boxplot, to show the
                    # adaptation effectivity, show 1 column with score per
                    # adaptation type.
                    boxplot_data[f"{name}:{value}"] = copy.deepcopy(
                        x_series_data
                    )
        elif graph_name != "input_graph":
            # Just post all scores on a single column; the avg score per snn
            # graph. For snn_algo and adapted_snn_algo graph they should both
            # be 100%, for the rad_snn_algo graph, there are no adaptations, so
            # the score is just whatever the avg score is.
            boxplot_data[graph_name] = copy.deepcopy(x_series_data)
    return boxplot_data


@typechecked
def boxplot_data_to_y_series(
    *, boxplot_data: Dict[str, Dict[int, Boxplot_x_val]]
) -> Dict[str, List[float]]:
    """Converts boxplot_data into x_labels and y_series for boxplot plotting.

    TODO: do this in Boxplot_x_vals itself.
    """

    # Initialise dataseries.
    data: Dict[str, List[float]] = {}
    for name in boxplot_data.keys():
        data[name] = []

    for name, seed_and_y_vals in boxplot_data.items():
        for y_val in seed_and_y_vals.values():
            data[name].append(
                y_val.correct_results
                / (y_val.correct_results + y_val.wrong_results)
            )
    return data


def store_pickle(*, run_configs: List[Run_config], filepath: str) -> None:
    """Stores run_config list into pickle file."""
    with open(filepath, "wb") as handle:
        pickle.dump(run_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(*, filepath: str) -> List[Run_config]:
    """Stores run_config list into pickle file."""
    with open(filepath, "rb") as handle:
        run_configs: List[Run_config] = pickle.load(handle)  # nosec
    return run_configs
