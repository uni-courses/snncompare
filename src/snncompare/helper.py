"""Contains helper functions that are used throughout this repository."""
import copy
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import networkx as nx
from networkx.classes.graph import Graph
from simsnn.core.simulators import Simulator
from typeguard import typechecked

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
        # Return the consecutive numbers in len(G) by default.
        return list(range(0, len(G)))
    if max_val == len(G) - 1:
        # TODO: remove this artifact.
        return list(range(0, len(G)))
    if max_val >= len(G):
        # Generate a too large list with numbers in range 0 to max_val.
        large_list = list(range(0, max_val))
        if seed is not None:
            random.seed(seed)
        # Return a random (with seed) sampling of len (G) from the previously
        # created large range within [0, max_val).
        return random.sample(large_list, len(G))
    raise ValueError(
        f"The max_val={max_val} is smaller than the graph size:{len(G)}."
    )


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
def add_stage_completion_to_graph(
    *, snn: Union[nx.Graph, Simulator], stage_index: int
) -> None:
    """Adds the completed stage to the list of completed stages for the
    incoming graph."""
    if isinstance(snn, Simulator):
        graph = snn.network.graph
    else:
        graph = snn
    # Initialise the completed_stages key.
    if stage_index == 1:
        if "completed_stages" in graph.graph:
            raise ValueError(
                "Error, the completed_stages parameter is"
                + f"already created for stage 1{graph.graph}:"
            )
        graph.graph["completed_stages"] = []
    # After stage 1, the completed_stages key should already be a list.
    elif not isinstance(graph.graph["completed_stages"], list):
        raise TypeError(
            "Error, the completed_stages parameter is not of type"
            + "list. instead, it is of type:"
            + f'{type(graph.graph["completed_stages"])}'
        )
    # At this point, the completed_stages key should not contain the current
    # stage index already..
    if stage_index in graph.graph["completed_stages"]:
        raise ValueError(
            f"Error, the stage:{stage_index} is already in the completed_stage"
            f's: {graph.graph["completed_stages"]}'
        )

    # Add the completed stages key to the snn graph.
    graph.graph["completed_stages"].append(stage_index)


@typechecked
def get_max_sim_duration(  # type:ignore[misc]
    *,
    input_graph: nx.Graph,
    run_config: Any,
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
        raise NotImplementedError(
            f"Error, algo_name:{algo_name} is not (yet) supported."
        )
    raise ValueError("Error, the simulation time was not found.")


@typechecked
def get_some_duration(
    *,
    simulator: str,
    snn_graph: Union[nx.DiGraph, Simulator],
    duration_name: str,
) -> int:
    """Compute the simulation duration for a given algorithm and graph."""
    if simulator == "simsnn":
        if duration_name not in snn_graph.network.graph.graph:
            if duration_name == "actual_duration":
                return len(snn_graph.multimeter.V)
            raise ValueError(
                f"Error, {duration_name} not found in simsnn graph."
            )
        return snn_graph.network.graph.graph[duration_name]
    if simulator == "nx":
        return snn_graph.graph[duration_name]
    raise NotImplementedError(f"Error, simulator:{simulator} not implemented.")


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


@typechecked
def get_with_adaptation_bool(*, graph_name: str) -> bool:
    """Returns True if the graph name belongs to a graph that has adaptation,
    returns False otherwise."""
    if graph_name in ["adapted_snn_graph", "rad_adapted_snn_graph"]:
        return True
    if graph_name in ["snn_algo_graph", "rad_snn_algo_graph"]:
        return False
    raise NotImplementedError(f"Error, {graph_name} is not supported.")


@typechecked
def get_with_radiation_bool(*, graph_name: str) -> bool:
    """Returns True if the graph name belongs to a graph that has radiation,
    returns False otherwise."""
    if graph_name in ["rad_snn_algo_graph", "rad_adapted_snn_graph"]:
        return True
    if graph_name in ["snn_algo_graph", "adapted_snn_graph"]:
        return False
    raise NotImplementedError(f"Error, {graph_name} is not supported.")


def get_snn_graph_from_graphs_dict(
    with_adaptation: bool,
    with_radiation: bool,
    graphs_dict: Dict[str, Union[nx.DiGraph, Simulator]],
) -> Union[nx.DiGraph, Simulator]:
    """Returns the snn graph corresponding to the adaptation and radiation
    configuration."""
    graph_name: str = get_snn_graph_name(
        with_adaptation=with_adaptation, with_radiation=with_radiation
    )
    return graphs_dict[graph_name]


def get_snn_graph_name(
    with_adaptation: bool, with_radiation: bool
) -> Union[nx.DiGraph, Simulator]:
    """Returns the snn graph name corresponding to the adaptation and radiation
    configuration."""
    if with_adaptation:
        if with_radiation:
            return "rad_adapted_snn_graph"
        return "adapted_snn_graph"

    if with_radiation:
        return "rad_snn_algo_graph"
    return "snn_algo_graph"


def get_snn_graph_names() -> List[str]:
    """Returns the 4 graph names: rad_adapted_snn_graph adapted_snn_graph
    rad_snn_algo_graph snn_algo_graph.

    in some order.
    """
    graph_names: List[str] = []

    for with_radiation in [False, True]:
        for with_adaptation in [False, True]:
            graph_name: str = get_snn_graph_name(
                with_adaptation=with_adaptation, with_radiation=with_radiation
            )
            graph_names.append(graph_name)
    return graph_names
