"""Gets the input graphs that may be adapted in later stages.

Takes run config of an experiment config as input, and returns a
networkx Graph.
"""
import networkx as nx

from src.graph_generation.Used_graphs import Used_graphs


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
        + f"{run_config['graph_nr']}"
    )
