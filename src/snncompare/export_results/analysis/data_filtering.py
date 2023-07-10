"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""

import copy

# Take in exp_config or run_configs
# If exp_config, get run_configs
from typing import Dict, List, Tuple

from rich.progress import track
from snnadaptation.Adaptation import Adaptation
from typeguard import typechecked

from snncompare.helper import get_snn_graph_names
from snncompare.import_results.load_stage4 import load_stage4_results
from snncompare.run_config.Run_config import Run_config


# pylint: disable = R0903
class Boxplot_x_val:
    """Stores the scores for a column in the boxplot.

    A column has a score in range [0,1] and the score in this range is
    created by the ratio of correct vs incorrect solutions computed by
    the snn graphs.
    """

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
    """Stores a list of columns that will be placed into a single box plot."""

    @typechecked
    def __init__(self, x_series: List[Boxplot_x_val]) -> None:
        self.x_series: List[Boxplot_x_val] = x_series


@typechecked
def get_boxplot_datapoints(
    *,
    adaptations: list[Adaptation],
    wanted_run_configs: List[Run_config],
    # run_config_nx_graphs: Dict,
    seeds: List[int],
) -> Dict[str, Dict[int, "Boxplot_x_val"]]:
    """Returns the run configs that still need to be ran."""
    # Creates a boxplot storage object, with the name of the column as string,
    # and in the value a dict with seed as key, and boxplot score (nr_of_wrongs
    # vs nr of right) in the value.
    # Hence it creates the x-axis categories: no redundancy, n-redundancy.
    boxplot_data: Dict[str, Dict[int, Boxplot_x_val]] = get_mdsa_boxplot_data(
        adaptations=adaptations,
        graph_names=get_snn_graph_names(),
        seeds=seeds,
    )

    for wanted_run_config in track(
        wanted_run_configs, total=len(wanted_run_configs)
    ):
        # TODO: load the boxplot data pickle if it exists, otherwise
        # load it from json data.

        # Get the results per x-axis category per graph type.
        for algo_name in wanted_run_config.algorithm.keys():
            if algo_name == "MDSA":
                stage_4_results_dict = load_stage4_results(
                    run_config=wanted_run_config,
                    stage_4_results_dict=None,
                )

                # Get the graphs names that were used in the run.
                graph_names: List[str] = get_snn_graph_names()

                # Specify the x-axis labels and get snn result dicts.
                x_labels, results = get_x_labels(
                    run_config_adaptation=wanted_run_config.adaptation,
                    adaptations=adaptations,
                    graphs_dict=stage_4_results_dict,
                    graph_names=graph_names,
                    simulator=wanted_run_config.simulator,
                )

                # Per column, compute the graph scores, and store them into the
                # boxplot_data.
                for x_label in x_labels:
                    # TODO: verify this check is correct.
                    # pylint: disable=C0201
                    if x_label in results.keys():
                        # TODO: p_values plot only if adaptation only is False.
                        add_graph_scores(
                            boxplot_data=boxplot_data,
                            x_label=x_label,
                            result=results[x_label],
                            seed=wanted_run_config.seed,
                        )

                # TODO: export boxplot data as pickle per run config.
            else:
                raise NotImplementedError(
                    f"Error, {algo_name} is not yet supported."
                )

    return boxplot_data


@typechecked
def get_x_labels(
    *,
    run_config_adaptation: Adaptation,
    adaptations: List[Adaptation],
    graphs_dict: Dict,
    graph_names: List[str],
    simulator: str,
) -> Tuple[List[str], Dict]:
    """Returns a tuple of the x-axis labels per column, and the accompanying
    snn graph results."""
    x_labels: List[str] = []
    results = {}
    for graph_name in graph_names:
        if graph_name == "rad_adapted_snn_graph":
            # TODO: fix.
            for adaptation in adaptations:
                # Create a new column and xlabel in the results dict, and
                # store the snn graph results in there.
                x_labels.append(adaptation.get_name())
                # Only add the results of the run_config adaptation graph if
                # the (generic) adaptation name is the same as that of the
                # run_config adaptation type and redundancy value.
                if run_config_adaptation.get_name() == adaptation.get_name():
                    if simulator == "simsnn":
                        results[adaptation.get_name()] = graphs_dict[
                            graph_name
                        ].network.graph.graph["results"]
                    elif simulator == "nx":
                        results[adaptation.get_name()] = graphs_dict[
                            graph_name
                        ]["graph"]["results"]
        elif graph_name != "input_graph":
            # This are:
            # - snn_algo_graphs: 100% score
            # - snn_adapted_graphs: 100% score
            # - rad_snn_algo_graphs: xx% score, but it all graphs of this
            # category are put into 1 column, because there aren't any
            # different types of adaptation.
            x_labels.append(graph_name)
            results[x_labels[-1]] = graphs_dict[
                graph_name
            ].network.graph.graph["results"]
    return x_labels, results


@typechecked
def get_mdsa_boxplot_data(
    *,
    adaptations: List[Adaptation],
    graph_names: List[str],
    seeds: List[int],
) -> Dict[str, Dict[int, "Boxplot_x_val"]]:
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
    # This are:
    # - snn_algo_graphs: 100% score
    # - snn_adapted_graphs: 100% score
    # - rad_snn_algo_graphs: xx% score, but it all graphs of this
    # category are put into 1 column, because there aren't any
    # different types of adaptation.
    boxplot_data: Dict[str, Dict[int, Boxplot_x_val]] = {}
    for graph_name in graph_names:
        if graph_name == "rad_adapted_snn_graph":
            for adaptation in adaptations:
                # Create multiple columns in boxplot, to show the
                # adaptation effectivity, show 1 column with score per
                # adaptation type.
                boxplot_data[adaptation.get_name()] = copy.deepcopy(
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
    *, boxplot_data: Dict[str, Dict[int, "Boxplot_x_val"]]
) -> Dict[str, List[float]]:
    """Converts boxplot_data into x_labels and y_series for boxplot plotting.

    Per column, per seed, it computes a score in range[0,1].
    """

    # Initialise dataseries.
    columns: Dict[str, List[float]] = {}
    for name in boxplot_data.keys():
        columns[name] = []

    for name, seed_and_y_vals in boxplot_data.items():
        for y_score in seed_and_y_vals.values():
            columns[name].append(
                # Compute the score in range [0,1] and add it to the column
                # score list.
                float(y_score.correct_results)
                / float(y_score.correct_results + y_score.wrong_results)
            )
    return columns


@typechecked
def add_graph_scores(
    *,
    boxplot_data: Dict[str, Dict[int, "Boxplot_x_val"]],
    x_label: str,
    result: Dict,
    seed: int,
) -> None:
    """Per column, per seed it creates a score, which can become a
    fraction/score in range [0,1].

    Then the e.g. 20 seeds, yield 20 scores in range [0,1] which can
    result in an avg score in range [0,1] per column.
    """
    if result["passed"]:
        boxplot_data[x_label][seed].correct_results += 1
    else:
        boxplot_data[x_label][seed].wrong_results += 1
