"""Parses the graph json files to recreate the graphs."""

import json
from pathlib import Path

from networkx.readwrite import json_graph

from src.export_results.helper import run_config_to_filename
from src.export_results.Output import get_extensions_list
from src.export_results.verify_stage_1_graphs import assert_graphs_are_in_dict
from src.helper import is_identical


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
    """Loads the experiment config, run config and graphs from the json
    file."""
    stage_index = 1

    # Get the json filename.
    filename = run_config_to_filename(run_config)
    relative_output_dir = "results/"
    extensions = get_extensions_list(run_config, stage_index)
    for extension in extensions:
        if extension == ".json":
            json_filepath = relative_output_dir + filename + extension

    # Load the json dictionary of results.
    stage_1_dict: dict = load_json_file_into_dict(json_filepath)
    if "graphs_dict" not in stage_1_dict:
        raise Exception(
            "Error, the graphs dict key was not in the stage_1_dict:"
            + f"{stage_1_dict}"
        )
    if stage_1_dict["graphs_dict"] == {}:
        raise Exception("Error, the graphs dict was an empty dict.")

    for key in stage_1_dict.keys():
        print(f"stage_1_dict.key={key}")

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

    for graph_name, some_graph in stage_1_dict["graphs_dict"].items():
        print(f"graph_name={graph_name}")
        print(f"some_graph={type(some_graph)}")
        stage_1_dict["graphs_dict"][graph_name] = json_to_digraph(some_graph)

    # TODO: convert dict back into graph.
    return stage_1_dict["graphs_dict"]


def json_to_digraph(json_data):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    TODO: remove if not used.

    """
    if json_data is not None:
        return json_graph.node_link_graph(json_data)
    raise Exception("Error, did not find json_data.")
