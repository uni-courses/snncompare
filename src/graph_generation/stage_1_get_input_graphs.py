"""Gets the input graphs that may be adapted in later stages.

Takes run config of an experiment config as input, and returns a
networkx Graph.
"""
import copy
from pprint import pprint
from typing import List

import networkx as nx

from src.graph_generation.Used_graphs import Used_graphs


def get_used_graphs(run_config: dict) -> List[nx.DiGraph]:
    """First gets the input graph.

    Then generates a graph with adaptation if it is required. Then
    generates a graph with radiation if it is required. Then returns
    this list of graphs.
    """
    input_graph = get_input_graph(run_config)
    get_adapted_graph(input_graph, run_config)
    return [input_graph]


def get_input_graph(run_config: dict) -> nx.DiGraph:
    """TODO: support retrieving graph sizes larger than size 5.
    TODO: ensure those graphs have valid properties, e.g. triangle-free and
    non-planar."""

    # Get the graph of the right size.
    # TODO: Pass random seed.
    input_graph = get_the_input_graphs(run_config)

    # TODO: Verify the graphs are valid.

    return input_graph


def get_the_input_graphs(run_config) -> nx.DiGraph:
    """Removes graphs that are not used, because of a maximum nr of graphs that
    is to be evaluated."""
    used_graphs = Used_graphs()
    input_graphs = used_graphs.get_graphs(run_config["graph_size"])
    if len(input_graphs) > run_config["graph_nr"]:
        return input_graphs[run_config["graph_nr"]]
    raise Exception(
        f"For input_graph of size:{run_config['graph_size']}, I found:"
        + f"{len(input_graphs)} graphs, yet expected graph_nr:"
        + f"{run_config['graph_nr']}. Please lower the max_graphs setting in:"
        + "size_and_max_graphs in the experiment configuration."
    )


def get_adapted_graph(input_graph: nx.DiGraph, run_config: dict) -> nx.DiGraph:
    """Converts an input graph of stage 1 and applies a form of brain-inspired
    adaptation to it."""
    pprint(run_config)
    for adapatation_name, adaptation_setting in run_config[
        "adaptation"
    ].items():
        print("adapatation")
        pprint(adapatation_name)

        if adapatation_name is None:
            pass
        elif adapatation_name == "redundancy":
            if not isinstance(adaptation_setting, float):
                raise Exception(
                    f"Error, adaptation_setting={adaptation_setting},"
                    + "which is not an int."
                )
            get_redundant_graph(input_graph, adaptation_setting)
        else:
            raise Exception(
                f"Error, adapatation_name:{adapatation_name} is not"
                + " supported."
            )


def get_redundant_graph(input_graph: nx.DiGraph, red_lev: float):
    """Returns a networkx graph that has a form of adaptation added."""
    if red_lev == 0:
        raise Exception(
            "Redundancy level 0 not supported if adaptation is" + " required."
        )
    if red_lev == 1:
        copy.deepcopy(input_graph)
    else:
        raise Exception(
            "Error, redundancy level above 1 is currently not" + " supported."
        )
