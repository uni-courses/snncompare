"""Parses the graph json files to recreate the graphs."""

import json
from pathlib import Path

import networkx as nx
from networkx.readwrite import json_graph

from src.export_results.helper import run_config_to_filename
from src.export_results.verify_stage_1_graphs import assert_graphs_are_in_dict
from src.helper import get_extensions_list, is_identical


def load_json_file_into_dict(json_filepath):
    """Loads a json file into dict from a filepath."""
    if not Path(json_filepath).is_file():
        raise Exception("Error, filepath does not exist:{filepath}")
    # TODO: verify extension.
    # TODO: verify json formatting is valid.
    with open(json_filepath, encoding="utf-8") as json_file:
        the_dict = json.load(json_file)
    return the_dict


def load_results_stage_1(run_config: dict) -> dict:
    """Loads the experiment config, run config and graphs from the json file.

    # TODO: ensure it only loads the graphs of stage 1. OR: make all
    dict loading the same.
    """
    stage_index = 1

    # Get the json filename.
    filename = run_config_to_filename(run_config)
    relative_output_dir = "results/"
    extensions = get_extensions_list(run_config, stage_index)
    for extension in extensions:
        if extension == ".json":
            filepath = relative_output_dir + filename + extension

    stage_1_dict = load_results_from_json(filepath)

    # Split the dictionary into three separate dicts.
    loaded_run_config = stage_1_dict["run_config"]

    # Verify the run_dict is valid.
    # TODO: determine why the unique ID is different for the same dict.
    # TODO: Verify passing the same dict to get hash with popped unique id
    # returns the same id.
    if not is_identical(run_config, loaded_run_config, ["unique_id"]):
        print("run_config")
        # pprint(run_config)
        print("Yet loaded_run_config is:")
        # pprint(loaded_run_config)
        raise Exception("Error, wrong run config was loaded.")

    # Verify the graph names are as expected for the graph name.
    assert_graphs_are_in_dict(run_config, stage_1_dict["graphs_dict"], 1)

    stage_1_graphs = {}
    # Converting back into graphs
    for graph_name, some_graph in stage_1_dict["graphs_dict"][
        "stage_1"
    ].items():
        print(f"graph_name={graph_name}")
        print(f"some_graph={type(some_graph)}")
        stage_1_graphs[graph_name] = json_to_digraph(some_graph)
    return stage_1_graphs


def load_results_from_json(json_filepath) -> dict:
    """Loads the results from a json file, and then converts the graph dicts
    back into a nx.Digraph object."""
    # Load the json dictionary of results.
    stage_1_dict: dict = load_json_file_into_dict(json_filepath)

    # Verify the dict contains a key for the graph dict.
    if "graphs_dict" not in stage_1_dict:
        raise Exception(
            "Error, the graphs dict key was not in the stage_1_dict:"
            + f"{stage_1_dict}"
        )

    # Verify the graphs dict is of type dict.
    if stage_1_dict["graphs_dict"] == {}:
        raise Exception("Error, the graphs dict was an empty dict.")

    set_graph_attributes(stage_1_dict["graphs_dict"])
    return stage_1_dict


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


def json_to_digraph(json_data):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    TODO: remove if not used.

    """
    if json_data is not None:
        return json_graph.node_link_graph(json_data)
    raise Exception("Error, did not find json_data.")
