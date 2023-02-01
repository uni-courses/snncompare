"""Contains helper functions that are used throughout this repository."""
import copy
import random
from pathlib import Path
from pprint import pprint
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import networkx as nx
import pylab as plt
from networkx.classes.graph import Graph

# from snncompare.export_results.load_json_to_nx_graph import dicts_are_equal
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_plots.plot_graphs import export_plot
from snncompare.run_config.Run_config import Run_config
from snncompare.run_config.Supported_run_settings import Supported_run_settings
from snncompare.run_config.verify_run_settings import verify_run_config

if TYPE_CHECKING:
    pass


@typechecked
def generate_list_of_n_random_nrs(
    *, G: Graph, max_val: Optional[int] = None, seed: Optional[int] = None
) -> List[int]:
    """Generates list of numbers in range of 1 to (and including) len(G), or:

    Generates list of numbers in range of 1 to (and including) max, or:
    TODO: Verify list does not contain duplicates, throw error if it does.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param max_val:  (Default value = None)
    :param seed: The value of the random seed used for this test.  (Default
    value = None)
    """
    if max_val is None:
        return list(range(0, len(G)))
    if max_val == len(G) - 1:
        return list(range(0, len(G)))
    if max_val >= len(G):
        large_list = list(range(0, max_val))
        if seed is not None:
            random.seed(seed)
        return random.sample(large_list, len(G))
    raise Exception(
        f"The max_val={max_val} is smaller than the graph size:{len(G)}."
    )


@typechecked
def compute_mark(*, input_graph: nx.Graph, rand_ceil: float) -> None:
    """Computes the mark at the counter neurons after the simulation is
    completed.

    TODO: move into algorithms module.
    """
    # Compute the mark based on degree+randomness=weight
    for node in input_graph.nodes:
        max_weight = max(
            input_graph.nodes[n]["weight"]
            for n in nx.all_neighbors(input_graph, node)
        )

        nr_of_max_weights = 0
        for n in nx.all_neighbors(input_graph, node):
            if (
                input_graph.nodes[n]["weight"] == max_weight
            ):  # should all max weight neurons be marked or only one of them?

                # Read of the score from countermarks, not marks.
                input_graph.nodes[n]["marks"] += rand_ceil + 1
                input_graph.nodes[n]["countermarks"] += 1
                nr_of_max_weights = nr_of_max_weights + 1

                # Verify there is only one max weight neuron.
                if nr_of_max_weights > 1:
                    raise Exception("Two numbers with identical max weight.")


@typechecked
def plot_alipour(
    *,
    configuration: str,
    seed: int,
    size: int,
    m: int,
    G: nx.DiGraph,
    export: bool = True,
    show: bool = False,
) -> None:
    """

    :param configuration:
    param iteration: The initialisation iteration that is used.
    :param seed: The value of the random seed used for this test.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    :param G: The original graph on which the MDSA algorithm is ran.
    :param export:  (Default value = True)
    :param show:  (Default value = False)

    """
    # pylint: disable=R0913
    # TODO: reduce 8/5 input arguments to at most 5/5.
    the_labels = get_alipour_labels(G=G, configuration=configuration)
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, labels=the_labels, with_labels=True)
    if show:
        plt.show()
    if export:
        export_plot(
            plt,
            f"alipour_{seed}_size{size}_m{m}_iter_combined_"
            + f"{configuration}",
            extensions=["png"],  # TODO: include run_config extensions.
        )

    plt.clf()
    plt.close()


@typechecked
def get_alipour_labels(*, G: nx.DiGraph, configuration: str) -> Dict[str, str]:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param configuration:

    """
    labels = {}
    for node_name in G.nodes:
        if configuration == "0rand_mark":
            labels[node_name] = (
                f'{node_name},R:{G.nodes[node_name]["random_number"]}, M:'
                + f'{G.nodes[node_name]["marks"]}'
            )
        elif configuration == "1weight":
            labels[
                node_name
            ] = f'{node_name}, W:{G.nodes[node_name]["weight"]}'
        else:
            raise Exception("Unsupported configuration.")

    return labels


# checks if file exists
@typechecked
def file_exists(*, filepath: str) -> bool:
    """

    :param string:

    """
    # TODO: Execute Path(string).is_file() directly instead of calling this
    # function.
    my_file = Path(filepath)
    return my_file.is_file()


@typechecked
def compute_marks_for_m_larger_than_one(
    *,
    input_graph: nx.Graph,
    m: int,
    seed: int,
    size: int,
    rand_ceil: int,
    export: bool = False,
    show: bool = False,
) -> None:
    """Stores the marks in the counter nodes of the graph.."""
    # pylint: disable=R0913
    # TODO: reduce 10/5 arguments to at most 5/5.
    # Don't compute for m=0
    for loop in range(1, m + 1):
        for node in input_graph.nodes:

            # Compute the weights for this round of m.
            input_graph.nodes[node]["weight"] = (
                input_graph.nodes[node]["marks"]
                + input_graph.nodes[node]["random_number"]
            )

            # Reset marks.
            input_graph.nodes[node]["marks"] = 0
            input_graph.nodes[node]["countermarks"] = 0

        for node in input_graph.nodes:
            max_weight = max(
                input_graph.nodes[n]["weight"]
                for n in nx.all_neighbors(input_graph, node)
            )
            for n in nx.all_neighbors(input_graph, node):
                if input_graph.nodes[n]["weight"] == max_weight:

                    # Always raise mark always by (rand_ceil + 1) * delta
                    # (not by 1).
                    input_graph.nodes[n]["marks"] += rand_ceil + 1
                    input_graph.nodes[n]["countermarks"] += 1

        if show or export:
            plot_alipour(
                configuration="0rand_mark",
                seed=seed,
                size=size,
                m=loop,
                G=input_graph,
                show=show,
            )
            plot_alipour(
                configuration="1weight",
                seed=seed,
                size=size,
                m=loop,
                G=input_graph,
                show=show,
            )
            plot_alipour(
                configuration="2inhib_weight",
                seed=seed,
                size=size,
                m=loop,
                G=input_graph,
                show=show,
            )


@typechecked
def set_node_default_values(
    *,
    input_graph: nx.Graph,
    node: int,
    rand_ceil: float,
    uninhibited_spread_rand_nrs: List[float],
) -> None:
    """Initialises the starting values of the node attributes."""
    # pylint: disable=R0913
    # TODO: reduce 6/5 arguments to at most 5/5.
    # Initialise values.
    # G.nodes[node]["marks"] = 0
    input_graph.nodes[node]["marks"] = input_graph.degree(node) * (
        rand_ceil + 1
    )
    input_graph.nodes[node]["countermarks"] = 0
    input_graph.nodes[node]["random_number"] = (
        1 * uninhibited_spread_rand_nrs[node]
    )
    input_graph.nodes[node]["weight"] = (
        input_graph.degree(node) * (rand_ceil + 1)
        + input_graph.nodes[node]["random_number"]
    )


@typechecked
def add_stage_completion_to_graph(
    *, input_graph: nx.Graph, stage_index: int
) -> None:
    """Adds the completed stage to the list of completed stages for the
    incoming graph."""
    # Initialise the completed_stages key.
    if stage_index == 1:
        if "completed_stages" in input_graph.graph:
            raise Exception(
                "Error, the completed_stages parameter is"
                + f"already created for stage 1{input_graph.graph}:"
            )
        input_graph.graph["completed_stages"] = []

    # After stage 1, the completed_stages key should already be a list.
    elif not isinstance(input_graph.graph["completed_stages"], list):
        raise Exception(
            "Error, the completed_stages parameter is not of type"
            + "list. instead, it is of type:"
            + f'{type(input_graph.graph["completed_stages"])}'
        )

    # At this point, the completed_stages key should not contain the current
    # stage index already..
    if stage_index in input_graph.graph["completed_stages"]:
        raise Exception(
            f"Error, the stage:{stage_index} is already in the completed_stage"
            f's: {input_graph.graph["completed_stages"]}'
        )

    # Add the completed stages key to the snn graph.
    input_graph.graph["completed_stages"].append(stage_index)


@typechecked
def get_max_sim_duration(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
) -> int:
    """Compute the simulation duration for a given algorithm and graph."""
    for algo_name, algo_settings in run_config.algorithm.items():
        if algo_name == "MDSA":

            # TODO: Move into stage_1 get input graphs.
            sim_time: int = int(
                len(input_graph)
                * (len(input_graph) + 1)
                * ((algo_settings["m_val"]) + 1)  # +_6 for delay
            )
            return sim_time
        raise Exception(
            f"Error, algo_name:{algo_name} is not (yet) supported."
        )
    raise Exception("Error, the simulation time was not found.")


@typechecked
def get_actual_duration(*, snn_graph: nx.DiGraph) -> int:
    """Compute the simulation duration for a given algorithm and graph."""
    return snn_graph.graph["sim_duration"]


@typechecked
def get_expected_stages(
    *,
    stage_index: int,
) -> List[int]:
    """Computes which stages should be expected at this stage of the
    experiment."""
    expected_stages = list(range(1, stage_index + 1))
    # stage 3 is checked on completeness by looking if image files exist.
    if 3 in expected_stages:
        expected_stages.remove(3)

    # Sort and remove dupes.
    return list(set(sorted(expected_stages)))


@typechecked
def generate_run_configs(
    *,
    exp_config: Exp_config,
    specific_run_config: Optional[Run_config] = None,
) -> List[Run_config]:
    """Generates the run configs belonging to an experiment config, and then
    removes all run configs except for the desired run config.

    Throws an error if the desired run config is not within the expected
    run configs.
    """
    found_run_config = False
    pprint(exp_config.__dict__)
    # Generate run configurations.
    run_configs: List[Run_config] = exp_config_to_run_configs(
        exp_config=exp_config
    )
    if specific_run_config is not None:
        if specific_run_config.unique_id is None:
            pprint(specific_run_config.__dict__)
            # Append unique_id to run_config
            Supported_run_settings().append_unique_run_config_id(
                specific_run_config, allow_optional=True
            )
        for gen_run_config in run_configs:
            if dicts_are_equal(
                left=gen_run_config.__dict__,
                right=specific_run_config.__dict__,
                without_unique_id=True,
            ):
                found_run_config = True
                if gen_run_config.unique_id != specific_run_config.unique_id:
                    raise Exception(
                        "Error, equal dict but unequal unique_ids."
                    )
                break

        if not found_run_config:
            pprint(run_configs)
            raise Exception(
                f"The expected run config:{specific_run_config} was not"
                "found."
            )
        run_configs = [specific_run_config]

    return run_configs


@typechecked
def exp_config_to_run_configs(
    *,
    exp_config: Exp_config,
) -> List[Run_config]:
    """Generates all the run_config dictionaries of a single experiment
    configuration. Then verifies whether each run_config is valid.

    TODO: Ensure this can be done modular, and lower the depth of the loop.
    """
    # pylint: disable=R0914

    run_configs: List[Run_config] = []

    # pylint: disable=R1702
    # TODO: make it loop through a list of keys.
    # for algorithm in exp_config.algorithms:
    for algorithm_name, algo_specs in exp_config.algorithms.items():
        for algo_config in algo_specs:
            algorithm = {algorithm_name: algo_config}

            for adaptation, radiation in get_adaptation_and_radiations(
                exp_config=exp_config
            ):
                fill_remaining_run_config_settings(
                    adaptation=adaptation,
                    algorithm=algorithm,
                    exp_config=exp_config,
                    radiation=radiation,
                    run_configs=run_configs,
                )

    set_run_config_export_settings(
        exp_config=exp_config,
        run_configs=run_configs,
    )
    return list(reversed(run_configs))


@typechecked
def fill_remaining_run_config_settings(
    *,
    adaptation: Union[None, Dict],
    algorithm: Dict,
    exp_config: Exp_config,
    radiation: Union[None, Dict],
    run_configs: List[Run_config],
) -> None:
    """Generate basic settings for a run config."""
    for seed in exp_config.seeds:
        for size_and_max_graph in exp_config.size_and_max_graphs:
            for simulator in exp_config.simulators:
                for graph_nr in range(0, size_and_max_graph[1]):
                    run_config: Run_config = run_parameters_to_dict(
                        adaptation=adaptation,
                        algorithm=algorithm,
                        seed=seed,
                        size_and_max_graph=size_and_max_graph,
                        graph_nr=graph_nr,
                        radiation=radiation,
                        exp_config=exp_config,
                        simulator=simulator,
                    )
                    run_configs.append(run_config)


@typechecked
def set_run_config_export_settings(
    *,
    exp_config: Exp_config,
    run_configs: List[Run_config],
) -> None:
    """Sets the export settings for run configs that are created based on an
    experiment config."""
    supp_run_setts = Supported_run_settings()
    for run_config in run_configs:
        if exp_config.export_images:
            run_config.export_types = exp_config.export_types
            run_config.gif = exp_config.gif
            run_config.overwrite_images_only = exp_config.overwrite_images_only
        run_config.recreate_s1 = exp_config.recreate_s1
        run_config.recreate_s2 = exp_config.recreate_s2
        run_config.recreate_s4 = exp_config.recreate_s4
        verify_run_config(
            supp_run_setts=supp_run_setts,
            run_config=run_config,
            has_unique_id=False,
            allow_optional=True,
        )

        # Append unique_id to run_config
        supp_run_setts.append_unique_run_config_id(
            run_config, allow_optional=True
        )

        # Append show_snns and export_images to run config.
        supp_run_setts.assert_has_key(
            exp_config.__dict__, "export_images", bool
        )
        run_config.export_images = exp_config.export_images


# pylint: disable=R0913
@typechecked
def run_parameters_to_dict(
    *,
    adaptation: Union[None, Dict[str, int]],
    algorithm: Dict[str, Dict[str, int]],
    seed: int,
    size_and_max_graph: Tuple[int, int],
    graph_nr: int,
    radiation: Union[None, Dict],
    exp_config: Exp_config,
    simulator: str,
) -> Run_config:
    """Stores selected parameters into a dictionary.

    Written as separate argument to keep code width under 80 lines. #
    TODO: verify typing.
    """
    run_config: Run_config = Run_config(
        adaptation=adaptation,
        algorithm=algorithm,
        seed=seed,
        graph_size=size_and_max_graph[0],
        graph_nr=graph_nr,
        radiation=radiation,
        recreate_s4=exp_config.recreate_s4,
        overwrite_images_only=exp_config.overwrite_images_only,
        simulator=simulator,
    )

    return run_config


def get_adaptation_and_radiations(
    *,
    exp_config: Exp_config,
) -> List[tuple]:
    """Returns a list of adaptations and radiations that will be used for the
    experiment."""

    adaptations_radiations: List[tuple] = []
    if exp_config.adaptations is None:
        adaptation = None
        adaptations_radiations.extend(
            get_radiations(exp_config=exp_config, adaptation=adaptation)
        )
    else:
        for (
            adaptation_name,
            adaptation_setts_list,
        ) in exp_config.adaptations.items():
            for adaptation_config in adaptation_setts_list:
                adaptation = {adaptation_name: adaptation_config}
                adaptations_radiations.extend(
                    get_radiations(
                        exp_config=exp_config, adaptation=adaptation
                    )
                )

    # Make sure it contains at least 1 empty entry:
    if not adaptations_radiations:  # Empty list evaluates to False.
        return [(None, None)]
    return adaptations_radiations


def get_radiations(
    *, exp_config: Exp_config, adaptation: Union[None, Dict[str, int]]
) -> List[Tuple[Union[None, Dict], Union[None, Dict]]]:
    """Returns the radiations."""
    adaptation_and_radiations: List[
        Tuple[Union[None, Dict], Union[None, Dict]]
    ] = []
    if exp_config.radiations is None:
        adaptation_and_radiations.append((adaptation, None))
    else:
        for (
            radiation_name,
            radiation_setts_list,
        ) in exp_config.radiations.items():
            # TODO: verify it is of type list.
            for rad_config in radiation_setts_list:
                radiation = {radiation_name: rad_config}
                adaptation_and_radiations.append((adaptation, radiation))
    return adaptation_and_radiations


@typechecked
def dicts_are_equal(
    *, left: Dict, right: Dict, without_unique_id: bool
) -> bool:
    """Determines whether two run configurations are equal or not."""
    if without_unique_id:
        left_copy = copy.deepcopy(left)
        right_copy = copy.deepcopy(right)
        if "unique_id" in left_copy:
            left_copy.pop("unique_id")
        if "unique_id" in right_copy:
            right_copy.pop("unique_id")
        return left_copy == right_copy
    return left == right
