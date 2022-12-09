"""Contains helper functions that are used throughout this repository."""
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import pylab as plt
from networkx.classes.graph import Graph
from typeguard import typechecked

from snncompare.export_plots.Plot_to_tex import Plot_to_tex


@typechecked
def sort_neurons(neurons: List, neuron_dict: dict) -> List:
    """

    :param neurons:
    :param neuron_dict:

    """
    sorted_neurons: List = []
    # Sort by value.
    sorted_dict: dict = dict(
        sorted(neuron_dict.items(), key=lambda item: item[1])
    )
    for neuron, _ in sorted_dict.items():
        if neuron in neurons:
            sorted_neurons.append(neuron)
    return sorted_neurons


@typechecked
def generate_list_of_n_random_nrs(
    G: Graph, max_val: int = None, seed: int = None
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
def get_y_position(G: nx.Graph, node: int, neighbour: int, d: float) -> float:
    """Ensures the degree receiver nodes per node are aligned with continuous
    interval.

    for example for node 1, the positions 0,2,3 are mapped to positions:
    0,1,2 by subtracting 1.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param node:
    :param neighbour:
    :param d: Unit length of the spacing used in the positions of the nodes for
    plotting.
    """
    if neighbour > node:
        return float((node + (neighbour - 1) / len(G)) * 4 * d)
    return float((node + neighbour / len(G)) * 4 * d)


@typechecked
def get_neuron_from_dict(
    neuron_dict: dict, neurons: List, neuron_name: str
) -> Any:
    """

    :param neuron_dict:
    :param neurons:
    :param neuron_name:

    """
    for neuron in neurons:
        if neuron_dict[neuron] == neuron_name:
            return neuron
    raise Exception(f"Did not find neuron:{neuron_name} in dict:{neuron_dict}")


@typechecked
def print_time(status: Any, previous_millis: int) -> Tuple[Any, int]:
    """

    :param status:

    :param previous_millis:

    """
    # TODO: remove unused function.
    now = datetime.now()
    now_millis = int(round(time.time() * 1000))

    duration_millis = now_millis - previous_millis
    print(
        f"{str(now.time())[:8]}, Duration:{duration_millis} [ms], "
        + f"status:{status}"
    )
    return now, now_millis


@typechecked
def compute_mark(
    delta: float, input_graph: nx.Graph, rand_ceil: float
) -> None:
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

                # Always raise mark always by (rand_ceil + 1) * delta
                # (not by 1).
                # Read of the score from countermarks, not marks.
                input_graph.nodes[n]["marks"] += (rand_ceil + 1) * delta
                input_graph.nodes[n]["countermarks"] += 1
                nr_of_max_weights = nr_of_max_weights + 1

                # Verify there is only one max weight neuron.
                if nr_of_max_weights > 1:
                    raise Exception("Two numbers with identical max weight.")


@typechecked
def plot_alipour(
    configuration: str,
    iteration: int,
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
    the_labels = get_alipour_labels(G, configuration=configuration)
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, labels=the_labels, with_labels=True)
    if show:
        plt.show()
    if export:
        plot_export = Plot_to_tex()
        plot_export.export_plot(
            plt,
            f"alipour_{seed}_size{size}_m{m}_iter{iteration}_combined_"
            + f"{configuration}",
        )

    plt.clf()
    plt.close()


@typechecked
def get_alipour_labels(G: nx.DiGraph, configuration: str) -> Dict[str, str]:
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
        elif configuration == "2inhib_weight":
            labels[
                node_name
            ] = f'{node_name}, W:{G.nodes[node_name]["inhibited_weight"]}'

    return labels


# checks if file exists
@typechecked
def file_exists(filepath: str) -> bool:
    """

    :param string:

    """
    # TODO: Execute Path(string).is_file() directly instead of calling this
    # function.
    my_file = Path(filepath)
    return my_file.is_file()


@typechecked
def compute_marks_for_m_larger_than_one(
    delta: float,
    input_graph: nx.Graph,
    inhibition: float,
    iteration: int,
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
            input_graph.nodes[node]["weight"] = (
                input_graph.nodes[node]["marks"]
                + input_graph.nodes[node]["random_number"]
            )
            input_graph.nodes[node]["inhibited_weight"] = (
                input_graph.nodes[node]["weight"] - inhibition
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
                    input_graph.nodes[n]["marks"] += (rand_ceil + 1) * delta
                    input_graph.nodes[n]["countermarks"] += 1

        if show or export:
            plot_alipour(
                "0rand_mark",
                iteration,
                seed,
                size,
                loop,
                input_graph,
                show=show,
            )
            plot_alipour(
                "1weight", iteration, seed, size, loop, input_graph, show=show
            )
            plot_alipour(
                "2inhib_weight",
                iteration,
                seed,
                size,
                loop,
                input_graph,
                show=show,
            )


@typechecked
def set_node_default_values(
    delta: float,
    input_graph: nx.Graph,
    inhibition: int,
    node: int,
    rand_ceil: float,
    uninhibited_spread_rand_nrs: List[float],
) -> None:
    """Initialises the starting values of the node attributes."""
    # pylint: disable=R0913
    # TODO: reduce 6/5 arguments to at most 5/5.
    # Initialise values.
    # G.nodes[node]["marks"] = 0
    input_graph.nodes[node]["marks"] = (
        input_graph.degree(node) * (rand_ceil + 1) * delta
    )
    input_graph.nodes[node]["countermarks"] = 0
    input_graph.nodes[node]["random_number"] = (
        1 * uninhibited_spread_rand_nrs[node]
    )
    input_graph.nodes[node]["weight"] = (
        input_graph.degree(node) * (rand_ceil + 1) * delta
        + input_graph.nodes[node]["random_number"]
    )
    input_graph.nodes[node]["inhibited_weight"] = (
        input_graph.nodes[node]["weight"] - inhibition
    )


@typechecked
def get_extensions_list(run_config: dict, stage_index: int) -> List:
    """

    :param run_config: param stage_index:
    :param stage_index:

    extensions = list(get_extensions_dict(run_config, stage_index).values())
    """
    return list(get_extensions_dict(run_config, stage_index).values())


@typechecked
def get_extensions_dict(run_config: dict, stage_index: int) -> dict:
    """Returns the file extensions of the output types. The dictionary key
    describes the content of the file, and the extension is given as the value.
    Config_and_graphs means that the experiment or run config is included in
    the file. Graphs means that the networkx graphs have been encoded.

    :param run_config: param stage_index:
    :param stage_index:
    """
    if stage_index == 1:
        return {"config_and_graphs": ".json"}
    if stage_index == 2:
        if run_config["simulator"] == "lava":
            return {"config": ".json"}
        # The networkx simulator is used:
        return {"config_and_graphs": ".json"}
    if stage_index == 3:
        # TODO: support .eps and/or .pdf.
        # TODO: verify graphs, or graphs_dict
        return {"graphs": ".png"}
    if stage_index == 4:
        return {"config_graphs_and_results": ".json"}
    raise Exception("Unsupported experiment stage.")


@typechecked
def add_stage_completion_to_graph(
    input_graph: nx.Graph, stage_index: int
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
def get_sim_duration(
    input_graph: nx.Graph,
    run_config: dict,
) -> int:
    """Compute the simulation duration for a given algorithm and graph."""
    for algo_name, algo_settings in run_config["algorithm"].items():
        if algo_name == "MDSA":

            # TODO: determine why +10 is required.
            # TODO: Move into stage_1 get input graphs.

            sim_time: int = (
                input_graph.graph["alg_props"]["inhibition"]
                * (algo_settings["m_val"] + 2)
                + 30
            )

            if not isinstance(sim_time, int):
                raise Exception(
                    "Error, sim_time is not an int."
                    + 'snn_graph.graph["alg_props"]["inhibition"]='
                    + f'{input_graph.graph["alg_props"]["inhibition"]}'
                    + '(algo_settings["m_val"] + 2)='
                    + f'{(algo_settings["m_val"] + 2)}'
                )
            return sim_time
        raise Exception("Error, algo_name:{algo_name} is not (yet) supported.")
    raise Exception("Error, the simulation time was not found.")


@typechecked
def get_actual_duration(snn_graph: nx.DiGraph) -> int:
    """Compute the simulation duration for a given algorithm and graph."""
    return snn_graph.graph["sim_duration"]


@typechecked
def get_expected_stages(
    export_images: bool, stage_index: int, to_run: dict
) -> List[int]:
    """Computes which stages should be expected at this stage of the
    experiment."""
    expected_stages = list(range(1, stage_index + 1))

    if not to_run["stage_3"] or not export_images:
        if 3 in expected_stages:
            expected_stages.remove(3)
    if export_images and stage_index > 2:

        if 3 not in expected_stages:
            expected_stages.append(3)
    # Sort and remove dupes.
    return list(set(sorted(expected_stages)))
