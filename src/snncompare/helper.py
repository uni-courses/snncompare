"""Contains helper functions that are used throughout this repository."""
import os
import pickle  # nosec - User is trusted not to load malicious pickle files.
import random
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import pylab as plt
from lava.proc.monitor.process import Monitor
from networkx.classes.graph import Graph

from src.snncompare.export_results.export_json_results import get_unique_hash
from src.snncompare.export_results.plot_graphs import (
    create_root_dir_if_not_exists,
)
from src.snncompare.export_results.Plot_to_tex import Plot_to_tex
from src.snncompare.graph_generation.radiation.Radiation_damage import (
    store_dead_neuron_names_in_graph,
)
from src.snncompare.simulation.LIF_neuron import LIF_neuron
from src.snncompare.simulation.verify_graph_is_snn import (
    verify_networkx_snn_spec,
)


def fill_dictionary(
    neuron_dict: dict,
    neurons: List,
    previous_us: dict,
    previous_vs: dict,
    previous_selector: dict = None,
    previous_has_spiked: dict = None,
) -> Tuple[dict, dict, Any, Any]:
    """

    :param neuron_dict:
    :param neurons:
    :param previous_us:
    :param previous_vs:
    :param previous_selector:  (Default value = None)
    :param previous_has_spiked:  (Default value = None)

    """
    # pylint: disable=R0913
    # TODO: reduce 6/5 arguments to at most 5/5.
    sorted_neurons = sort_neurons(neurons, neuron_dict)
    for neuron in sorted_neurons:
        neuron_name = neuron_dict[neuron]
        previous_us[neuron_name] = 0
        previous_vs[neuron_name] = 0
        if previous_selector is not None:
            previous_selector[neuron_name] = 0
        if previous_has_spiked is not None:
            previous_has_spiked[neuron_name] = False

    if previous_selector is not None:
        if previous_has_spiked is not None:
            return (
                previous_us,
                previous_vs,
                previous_selector,
                previous_has_spiked,
            )
        return previous_us, previous_vs, previous_selector, None
    if previous_has_spiked is not None:
        return previous_us, previous_vs, previous_has_spiked, None
    return previous_us, previous_vs, None, None


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
        return list(range(1, len(G) + 1))
    if max_val == len(G):
        return list(range(1, len(G) + 1))
    if max_val > len(G):
        large_list = list(range(1, max_val + 1))
        if seed is not None:
            random.seed(seed)
        return random.sample(large_list, len(G))
    raise Exception(
        "The max_val={max_val} is smaller than the graph size:{len(G)}."
    )


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


def delete_files_in_folder(folder: str) -> None:
    """

    :param folder:

    """
    os.makedirs(folder, exist_ok=True)
    try:
        shutil.rmtree(folder)
    except OSError:
        print(traceback.format_exc())
    os.makedirs(folder, exist_ok=False)


def delete_file_if_exists(filepath: str) -> None:
    """Deletes a file if it exists."""
    try:
        os.remove(filepath)
    except OSError:
        pass


def export_get_degree_graph(
    has_adaptation: bool,
    has_radiation: bool,
    G: nx.DiGraph,
    get_degree: dict,
    iteration: int,
    m: int,
    neuron_death_probability: float,
    rand_props: Dict[str, Any],
    seed: int,
    sim_time: int,
    size: int,
    test_object: Any,
) -> None:
    """

    :param has_adaptation:
    :param has_radiation: Indicates whether the experiment simulates
    radiation or not.
    :param G: The original graph on which the MDSA algorithm is ran.
    :param get_degree: Graph with the MDSA SNN approximation solution.
    :param iteration: The initialisation iteration that is used.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    :param neuron_death_probability:
    :param rand_props:
    :param run_result:
    :param seed: The value of the random seed used for this test.
    :param sim_time: Nr. of timesteps for which the experiment is ran.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param test_object: Object containing test settings.

    """
    # pylint: disable=R0913
    # TODO: reduce 12/5 arguments.
    # TODO: remove unused function.
    remove_monitors_from_get_degree(get_degree)
    # pylint: disable=R0801
    unique_hash = get_unique_hash(
        test_object.final_dead_neuron_names,
        has_adaptation,
        has_radiation,
        iteration,
        m,
        neuron_death_probability,
        seed,
        sim_time,
    )

    if test_object.rad_damaged_graph is not None:
        store_dead_neuron_names_in_graph(
            test_object.rad_damaged_graph,
            test_object.final_dead_neuron_names,
        )
    create_root_dir_if_not_exists("pickles")
    with open(
        f"pickles/probability_{neuron_death_probability}"
        + f"adapt_{has_adaptation}_{seed}_size{size}_m{m}_iter"
        + f"{iteration}_{unique_hash}.pkl",
        "wb",
    ) as fh:
        pickle.dump(
            [
                has_adaptation,
                G,
                has_radiation,
                iteration,
                m,
                neuron_death_probability,
                rand_props,
                seed,
                sim_time,
                test_object.mdsa_graph,
                test_object.brain_adaptation_graph,
                test_object.rad_damaged_graph,
                test_object.final_dead_neuron_names,
                unique_hash,
            ],
            fh,
        )


def remove_monitors_from_get_degree(get_degree: nx.DiGraph) -> None:
    """

    :param get_degree: Graph with the MDSA SNN approximation solution.

    """
    for node_name in get_degree.nodes:
        get_degree.nodes[node_name]["neuron"] = None
        get_degree.nodes[node_name]["spike_monitor"] = None
        get_degree.nodes[node_name]["spike_monitor_id"] = None


def get_counter_neurons_from_dict(
    expected_nr_of_neurons: int, neuron_dict: dict, m: int
) -> List:
    """

        :param expected_nr_of_neurons:
        :param neuron_dict:
        :param m: The amount of approximation iterations used in the MDSA
    approximation.

    """
    counter_neurons = []
    neurons = list(neuron_dict.keys())
    neuron_names = list(neuron_dict.values())

    # Get sorted counter neurons.
    for node_index in range(expected_nr_of_neurons):
        for neuron_name in neuron_names:
            if neuron_name == f"counter_{node_index}_{m}":
                counter_neurons.append(
                    get_neuron_from_dict(neuron_dict, neurons, neuron_name)
                )

    if expected_nr_of_neurons != len(counter_neurons):
        raise Exception(
            f"Error, expected {expected_nr_of_neurons} neurons, yet found"
            + f" {len(counter_neurons)} neurons"
        )
    return counter_neurons


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


def write_results_to_file(
    has_passed: bool,
    m: int,
    G: nx.DiGraph,
    iteration: int,
    G_alipour: nx.DiGraph,
    counter_neurons: List,
) -> None:
    """

        :param has_passed:
        :param m: The amount of approximation iterations used in the MDSA
    approximation.
        :param G: The original graph on which the MDSA algorithm is ran.
        param iteration: The initialisation iteration that is used.
        :param G_alipour:
        :param counter_neurons: Neuron objects at the counter position. Type
        unknown.

    """
    # pylint: disable=R0913
    # TODO: reduce 6/5 arguments to at most 5/5.

    # Append-adds at last
    with open("results.txt", "a", encoding="utf-8") as file1:  # append mode

        now = datetime.now()
        file1.write(now.strftime("%Y-%m-%d %H:%M:%S\n"))
        file1.write(f"m={m}\n")
        file1.write(f"len(G)={len(G)}\n")
        file1.write(f"has_passed={has_passed,}\n")
        file1.write("edges\n")
        for edge in G.edges:
            file1.write(f"{str(edge)}\n")
        file1.write(f"iteration={iteration}\n")
        file1.write("G_alipour countermarks-SNN counter current\n")
        for node in G.nodes:
            file1.write(
                f'{G_alipour.nodes[node]["countermarks"]}-"'
                + f"{counter_neurons[node].u.get()}\n"
            )
        file1.write("\n\n")
        file1.close()


def create_neuron_monitors(test_object: Any, sim_time: int) -> None:
    """

    :param test_object: Object containing test settings.
    :param sim_time: Nr. of timesteps for which the experiment is ran.

    """
    get_degree = test_object.get_degree
    for node_name in get_degree.nodes:
        # The connecting node does not interact with the snn, it serves merely
        # to connect the snn for simulation purposes.
        if node_name != "connecting_node":
            neuron = get_neuron_from_dict(
                test_object.neuron_dict, test_object.neurons, node_name
            )

            if neuron is None:
                raise Exception(
                    "Error, was not able to find the neuron for "
                    + f"node:{node_name}"
                )

            # Create monitor
            monitor = Monitor()

            # Specify what the monitor monitors, and for how long.
            monitor.probe(neuron.s_out, sim_time)

            # Get monitor process id
            monitor_process_id = list(monitor.get_data())[0]

            # Read out the boolean spike (at time t, with 1=1 being after 1
            # second of running.) or no spike with:
            # s_out=monitor.get_data()[monitor_process_id]["s_out"][t]
            # You need to call this again after every timestep.

            # Store monitor in node attribute.
            get_degree.nodes[node_name]["neuron"] = neuron
            get_degree.nodes[node_name]["spike_monitor"] = monitor
            get_degree.nodes[node_name][
                "spike_monitor_id"
            ] = monitor_process_id


def store_spike_values_in_neurons(get_degree: nx.DiGraph, t: int) -> None:
    """

    :param get_degree: Graph with the MDSA SNN approximation solution.
    :param t:

    """
    for node_name in get_degree.nodes:
        if node_name != "connecting_node":
            # Add neuron as attribute of node.

            monitor = get_degree.nodes[node_name]["spike_monitor"]
            monitor_process_id = get_degree.nodes[node_name][
                "spike_monitor_id"
            ]

            # TODO: doubt, or t=1?
            # simulation starts at time t=1, then 1 timestep is simulated,
            # after that time step, this monitoring function is called, which
            # has the first spike value stored in list index 0, hence t-1.
            # Spike value for t=2 is then stored in list index 1. (The first 0
            # is because it is a list of lists.)
            s_out = monitor.get_data()[monitor_process_id]["s_out"][t - 1][0]
            if s_out == 1:
                get_degree.nodes[node_name]["spike"][t] = True
            elif s_out == 0:
                get_degree.nodes[node_name]["spike"][t] = False
            else:
                raise Exception(
                    f"Was not able to if node:{node_name} spikes or not."
                )


def full_alipour(
    iteration: int,
    G: nx.DiGraph,
    m: int,
    rand_props: Any,
    seed: int,
    size: int,
    show: bool = False,
    export: bool = False,
) -> nx.DiGraph:
    """TODO: remove return, the results are stored in the graph, even without
    returning the graph.

        param iteration: The initialisation iteration that is used.

        :param G: The original graph on which the MDSA algorithm is ran.
        :param m: The amount of approximation iterations used in the MDSA
    approximation.
        :param rand_props:
        :param seed: The value of the random seed used for this test.
        :param size: Nr of nodes in the original graph on which test is ran.
        :param show:  (Default value = False)
        :param export:  (Default value = False)
    """
    # pylint: disable=R0913
    # TODO: reduce 8/5 input arguments to at most 5/5.
    # pylint: disable=R0914
    # TODO: reduce 18/15 local variables to at most 15/15.
    # pylint: disable=R0912
    # TODO: reduce 8/5 input branches to at most 5/5.

    delta = rand_props.delta
    inhibition = rand_props.inhibition
    rand_ceil = rand_props.rand_ceil
    # TODO: resolve this naming discrepancy
    rand_nrs = rand_props.initial_rand_current

    # Reverse engineer actual rand nrs:
    uninhibited_rand_nrs = [(x + inhibition) for x in rand_nrs]
    for node in G.nodes:
        set_node_default_values(
            delta, G, inhibition, node, rand_ceil, uninhibited_rand_nrs
        )

    if show or export:
        plot_alipour("0rand_mark", iteration, seed, size, 0, G, show=show)
        plot_alipour("1weight", iteration, seed, size, 0, G, show=show)
        plot_alipour("2inhib_weight", iteration, seed, size, 0, G, show=show)

    compute_mark(delta, G, rand_ceil)

    compute_marks_for_m_larger_than_one(
        delta=delta,
        G=G,
        inhibition=inhibition,
        iteration=iteration,
        m=m,
        seed=seed,
        size=size,
        rand_ceil=rand_ceil,
        export=False,
        show=False,
    )

    for node in G.nodes:
        print(f'node:{node}, ali-mark:{G.nodes[node]["countermarks"]}')
    return G


def compute_mark(delta: float, G: nx.DiGraph, rand_ceil: float) -> None:
    """Computes the mark at the counter neurons after the simulation is
    completed."""
    # Compute the mark based on degree+randomness=weight
    for node in G.nodes:
        max_weight = max(
            G.nodes[n]["weight"] for n in nx.all_neighbors(G, node)
        )

        nr_of_max_weights = 0
        for n in nx.all_neighbors(G, node):
            if (
                G.nodes[n]["weight"] == max_weight
            ):  # should all max weight neurons be marked or only one of them?

                # Always raise mark always by (rand_ceil + 1) * delta
                # (not by 1).
                # Read of the score from countermarks, not marks.
                G.nodes[n]["marks"] += (rand_ceil + 1) * delta
                G.nodes[n]["countermarks"] += 1
                nr_of_max_weights = nr_of_max_weights + 1

                # Verify there is only one max weight neuron.
                if nr_of_max_weights > 1:
                    raise Exception("Two numbers with identical max weight.")


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
def file_exists(filepath: str) -> bool:
    """

    :param string:

    """
    # TODO: Execute Path(string).is_file() directly instead of calling this
    # function.
    my_file = Path(filepath)
    return my_file.is_file()


def compute_marks_for_m_larger_than_one(
    delta: float,
    G: nx.DiGraph,
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
        for node in G.nodes:
            G.nodes[node]["weight"] = (
                G.nodes[node]["marks"] + G.nodes[node]["random_number"]
            )
            G.nodes[node]["inhibited_weight"] = (
                G.nodes[node]["weight"] - inhibition
            )
            # Reset marks.
            G.nodes[node]["marks"] = 0
            G.nodes[node]["countermarks"] = 0

        for node in G.nodes:
            max_weight = max(
                G.nodes[n]["weight"] for n in nx.all_neighbors(G, node)
            )
            for n in nx.all_neighbors(G, node):
                if G.nodes[n]["weight"] == max_weight:

                    # Always raise mark always by (rand_ceil + 1) * delta
                    # (not by 1).
                    G.nodes[n]["marks"] += (rand_ceil + 1) * delta
                    G.nodes[n]["countermarks"] += 1

        if show or export:
            plot_alipour(
                "0rand_mark", iteration, seed, size, loop, G, show=show
            )
            plot_alipour("1weight", iteration, seed, size, loop, G, show=show)
            plot_alipour(
                "2inhib_weight", iteration, seed, size, loop, G, show=show
            )


def set_node_default_values(
    delta: float,
    G: nx.DiGraph,
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
    G.nodes[node]["marks"] = G.degree(node) * (rand_ceil + 1) * delta
    G.nodes[node]["countermarks"] = 0
    G.nodes[node]["random_number"] = 1 * uninhibited_spread_rand_nrs[node]
    G.nodes[node]["weight"] = (
        G.degree(node) * (rand_ceil + 1) * delta
        + G.nodes[node]["random_number"]
    )
    G.nodes[node]["inhibited_weight"] = G.nodes[node]["weight"] - inhibition


def is_identical(
    original: dict, other: dict, excluded_keys: List[str]
) -> bool:
    """Compares dictionaries whether the left dict contains the same keys, as
    the right keys, for each key verifies the values are identical.

    The keys and values in excluded_keys do not need to be similar.
    TODO: specify whether the keys need to be at least in the dict or not.
    """

    # Check whether all values in original dict are in the excluded keys
    for key in original.keys():
        if key not in other.keys():
            if key not in excluded_keys:
                return False

        # Check if the values are identical for the given key.
        else:
            if isinstance(other[key], type(original[key])):
                if other[key] != original[key]:
                    if key not in excluded_keys:
                        return False
            else:
                return False
    return True


def get_extensions_list(run_config: dict, stage_index: int) -> List:
    """

    :param run_config: param stage_index:
    :param stage_index:

    extensions = list(get_extensions_dict(run_config, stage_index).values())
    """
    return list(get_extensions_dict(run_config, stage_index).values())


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


def add_stage_completion_to_graph(
    some_graph: nx.DiGraph, stage_index: int
) -> None:
    """Adds the completed stage to the list of completed stages for the
    incoming graph."""
    # Initialise the completed_stages key.
    if stage_index == 1:
        if "completed_stages" in some_graph.graph:
            raise Exception(
                "Error, the completed_stages parameter is"
                + f"already created for stage 1{some_graph.graph}:"
            )
        some_graph.graph["completed_stages"] = []

    # After stage 1, the completed_stages key should already be a list.
    elif not isinstance(some_graph.graph["completed_stages"], list):
        raise Exception(
            "Error, the completed_stages parameter is not of type"
            + "list. instead, it is of type:"
            + f'{type(some_graph.graph["completed_stages"])}'
        )

    # At this point, the completed_stages key should not contain the current
    # stage index already..
    if stage_index in some_graph.graph["completed_stages"]:
        raise Exception(
            f"Error, the stage:{stage_index} is already in the completed_stage"
            f's: {some_graph.graph["completed_stages"]}'
        )

    # Add the completed stages key to the snn graph.
    some_graph.graph["completed_stages"].append(stage_index)


def get_sim_duration(
    input_graph: nx.DiGraph,
    run_config: dict,
) -> int:
    """Compute the simulation duration for a given algorithm and graph."""
    for algo_name, algo_settings in run_config["algorithm"].items():
        if algo_name == "MDSA":

            # TODO: determine why +10 is required.
            # TODO: Move into stage_1 get input graphs.

            sim_time: int = (
                input_graph.graph["alg_props"]["inhibition"]
                * (algo_settings["m_val"] + 1)
                + 10
            )

            if not isinstance(sim_time, int):
                raise Exception(
                    "Error, sim_time is not an int."
                    + 'snn_graph.graph["alg_props"]["inhibition"]='
                    + f'{input_graph.graph["alg_props"]["inhibition"]}'
                    + '(algo_settings["m_val"] + 1)='
                    + f'{(algo_settings["m_val"] + 1)}'
                )
            return sim_time
        raise Exception("Error, algo_name:{algo_name} is not (yet) supported.")
    raise Exception("Error, the simulation time was not found.")


def old_graph_to_new_graph_properties(G: nx.DiGraph) -> None:
    """Converts the old graph properties of the first template neuron into the
    new template neuron.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    for nodename in G.nodes:
        G.nodes[nodename]["nx_LIF"] = [
            LIF_neuron(
                name=nodename,
                bias=float(G.nodes[nodename]["bias"]),
                du=float(G.nodes[nodename]["du"]),
                dv=float(G.nodes[nodename]["dv"]),
                vth=float(G.nodes[nodename]["vth"]),
            )
        ]
    verify_networkx_snn_spec(G, t=0, backend="nx")


def get_expected_stages(
    export_images: bool, stage_index: int, to_run: dict
) -> List[int]:
    """Computes which stages should be expected at this stage of the
    experiment."""
    expected_stages = list(range(1, stage_index + 1))

    if not to_run["stage_3"]:
        if 3 in expected_stages:
            expected_stages.remove(3)
    if export_images:
        if 3 not in expected_stages:
            expected_stages.append(3)
    # Sort and remove dupes.
    return list(set(sorted(expected_stages)))
