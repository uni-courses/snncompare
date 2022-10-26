"""Code to load and parse the simulation results dict consisting of the
experiment config, run config and json graphs, from a json dict.

Appears to also be used to partially convert json graphs back into nx
graphs.
"""
import json
from pathlib import Path

import networkx as nx

from src.export_results.verify_nx_graphs import verify_results_nx_graphs


def load_results_from_json(json_filepath: str, run_config: dict) -> dict:
    """Loads the results from a json file, and then converts the graph dicts
    back into a nx.Digraph object."""
    # Load the json dictionary of results.
    results_json_graphs: dict = load_json_file_into_dict(json_filepath)

    # Verify the dict contains a key for the graph dict.
    if "graphs_dict" not in results_json_graphs:
        raise Exception(
            "Error, the graphs dict key was not in the stage_1_dict:"
            + f"{results_json_graphs}"
        )

    # Verify the graphs dict is of type dict.
    if results_json_graphs["graphs_dict"] == {}:
        raise Exception("Error, the graphs dict was an empty dict.")

    set_graph_attributes(results_json_graphs["graphs_dict"])
    verify_results_nx_graphs(results_json_graphs, run_config)
    return results_json_graphs


def set_graph_attributes(graphs_dict: dict) -> nx.DiGraph:
    """First loads the graph attributes from a graph dict and stores them as a
    dict.

    Then converts the nx.DiGraph that is encoded as a dict, back into a
    nx.DiGraph object.
    """

    # For each graph in the graphs dict, restore the graph attributes.
    for graph_name in graphs_dict.keys():

        # First load the graph attributes from the dict.
        graph_attributes = graphs_dict[graph_name]["graph"]

        # Convert the graph dict back into an nx.DiGraph object.
        graphs_dict[graph_name] = nx.DiGraph(graphs_dict[graph_name])

        # Add the graph attributes back to the nx.DiGraph object.
        for graph_attribute_name, value in graph_attributes.items():
            graphs_dict[graph_name].graph[graph_attribute_name] = value


def load_json_file_into_dict(json_filepath):
    """TODO: make this into a private function that cannot be called by
    any other object than some results loader.
    Loads a json file into dict from a filepath."""
    if not Path(json_filepath).is_file():
        raise Exception("Error, filepath does not exist:{filepath}")
    # TODO: verify extension.
    # TODO: verify json formatting is valid.
    with open(json_filepath, encoding="utf-8") as json_file:
        the_dict = json.load(json_file)
    return the_dict
