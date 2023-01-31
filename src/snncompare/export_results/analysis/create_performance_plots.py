"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""

# Take in exp_config or run_configs
# If exp_config, get run_configs
import copy
import pickle  # nosec
from pathlib import Path
from pprint import pprint
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
from snncompare.helper import (
    generate_run_configs,
    get_adaptation_and_radiations,
)
from snncompare.import_results.check_completed_stages import (
    has_outputted_stage,
)


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
def create_performance_plots(*, exp_config: Exp_config) -> None:
    """Ensures all performance boxplots are created."""

    pickle_run_configs_filepath: str = (
        "latex/Images/completed_run_configs.pickle"
    )
    run_configs: List[Run_config]

    # Get run_configs
    if Path(pickle_run_configs_filepath).is_file():
        completed_run_configs = load_pickle(
            filepath=pickle_run_configs_filepath
        )
    else:
        run_configs = generate_run_configs(
            exp_config=exp_config, specific_run_config=None
        )
        # Get list of required run_configs.
        (
            completed_run_configs,
            missing_run_configs,
        ) = get_completed_and_missing_run_configs(run_configs=run_configs)

        # Create a list of run_configs that still need to be completed.
        # prompt user whether user wants to complete these run_configs.
        for missing_run_config in missing_run_configs:
            # Execute those run_configs
            Experiment_runner(
                exp_config=exp_config,
                specific_run_config=missing_run_config,
                perform_run=True,
            )

        store_pickle(
            run_configs=completed_run_configs,
            filepath=pickle_run_configs_filepath,
        )
    print("Loaded run_configs")

    count: int = 0
    adaptations_radiations = get_adaptation_and_radiations(
        exp_config=exp_config
    )
    radiations = list(map(lambda x: x[1], adaptations_radiations))

    unique_radiations = [
        dict(t) for t in {tuple(d.items()) for d in radiations}
    ]
    adaptations_per_radiation: List = []
    for i, unique_rad in enumerate(unique_radiations):
        adaptations_per_radiation.append([])
    for i, unique_rad in enumerate(unique_radiations):
        for adaptation, radiation in adaptations_radiations:
            if unique_rad == radiation:
                adaptations_per_radiation[i].append(adaptation)
    rad_adap_dict: Dict = {}
    print(f"adaptations_per_radiation={adaptations_per_radiation}")
    pprint(rad_adap_dict)

    # Sort the unique_radiations.
    keys = list(map(lambda x: list(x.keys()), unique_radiations))
    print(keys)
    unique_keys = []
    for elem in keys:
        for key in elem:
            if key not in unique_keys:
                unique_keys.append(key)

    for unique_key in unique_keys:
        # pylint: disable=W0640
        sorted_radiations = sorted(
            unique_radiations, key=lambda d: d[unique_key]
        )
    print(f"sorted_radiations={sorted_radiations}")
    original_order = []
    for radiation in sorted_radiations:
        for value in radiation.values():
            for i, original_radiation in enumerate(unique_radiations):
                for key, val in original_radiation.items():
                    if value == val:
                        original_order.append(i)
    print(f"original_order={original_order}")

    for i in original_order:
        radiation = unique_radiations[i]
        for radiation_name, radiation_value in reversed(radiation.items()):
            count = count + 1

            print(f"radiation={radiation}")

            # Get run configs belonging to this radiation type/level.
            wanted_run_configs: List[Run_config] = []
            for run_config in completed_run_configs:
                if run_config.radiation == radiation:
                    wanted_run_configs.append(run_config)

            # Get results per line.
            boxplot_data: Dict[
                str, Dict[int, Boxplot_x_val]
            ] = get_boxplot_datapoints(
                adaptations=adaptations_per_radiation[i],
                wanted_run_configs=wanted_run_configs,
                seeds=exp_config.seeds,
            )

            y_series = boxplot_data_to_y_series(boxplot_data=boxplot_data)

            # Generate box plots.
            create_box_plot(
                extensions=["png"],
                filename=f"{count}_boxplot_{radiation_name}={radiation_value}",
                legendPosition=0,
                output_dir="latex/Images",
                x_axis_label="x-axis label [units]",
                y_axis_label="y-axis label [units]",
                y_series=y_series,
                title=f"{radiation_name}={radiation_value}",
                x_axis_label_rotation=45,
            )


@typechecked
def get_completed_and_missing_run_configs(
    *,
    run_configs: List[Run_config],
) -> Tuple[List[Run_config], List[Run_config]]:
    """Returns the run configs that still need to be ran."""
    missing_run_configs: List[Run_config] = []
    completed_run_configs: List[Run_config] = []
    for run_config in run_configs:
        if not has_outputted_stage(
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
    adaptations: List[Dict[str, int]],
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
    adaptations: List[Dict[str, int]],
    graphs_dict: Dict,
    graph_names: List[str],
) -> Tuple[List[str], Dict]:
    """Returns the x-axis labels per dataserie/boxplot."""
    x_labels: List[str] = []
    results = {}
    for graph_name in graph_names:
        if graph_name == "rad_adapted_snn_graph":
            for adaptation in adaptations:
                for name, value in adaptation.items():
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
    adaptations: List[Dict[str, int]],
    graph_names: List[str],
    seeds: List[int],
) -> Dict[str, Dict[int, Boxplot_x_val]]:
    """Creates the boxplot data objects for the MDSA algorithm."""
    x_series_data: Dict[int, Boxplot_x_val] = {}
    for seed in seeds:
        x_series_data[seed] = Boxplot_x_val(
            correct_results=0,
            wrong_results=0,
        )
    boxplot_data: Dict[str, Dict[int, Boxplot_x_val]] = {}
    for graph_name in graph_names:
        if graph_name == "rad_adapted_snn_graph":
            for adaptation in adaptations:
                for name, value in adaptation.items():
                    boxplot_data[f"{name}:{value}"] = copy.deepcopy(
                        x_series_data
                    )
        elif graph_name != "input_graph":
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
