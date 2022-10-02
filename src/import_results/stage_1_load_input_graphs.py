"""Parses the graph json files to recreate the graphs."""

import json
from pathlib import Path
from typing import List

import networkx as nx
from networkx.readwrite import json_graph

from src.export_results.helper import run_config_to_filename
from src.export_results.verify_stage_1_graphs import (
    assert_graphs_are_in_dict,
    get_expected_stage_1_graph_names,
)
from src.helper import get_extensions_list, is_identical


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


def performed_stage(run_config, stage_index: int) -> bool:
    """Verifies the required output files exist for a given simulation.

    :param run_config: param stage_index:
    :param stage_index:
    """
    expected_filepaths = []

    filename = run_config_to_filename(run_config)

    relative_output_dir = "results/"
    extensions = get_extensions_list(run_config, stage_index)
    for extension in extensions:
        if stage_index in [1, 2, 4]:

            expected_filepaths.append(
                relative_output_dir + filename + extension
            )
            # TODO: append expected_filepath to run_config per stage.
            print(f"expected_filepaths={expected_filepaths}")

        if stage_index == 3:

            # Check if output file(s) of stage 2 exist, otherwise return False.
            if not Path(relative_output_dir + filename + extension).is_file():
                return False

            # TODO: get graph length from graph object.
            nr_of_simulation_steps = 100000
            for t in range(0, nr_of_simulation_steps):
                # Generate graph filenames
                expected_filepaths.append(
                    relative_output_dir + filename + f"t_{t}" + extension
                )

    # Check if the expected output files already exist.
    for filepath in expected_filepaths:
        if not Path(filepath).is_file():
            print("Result file not found.")
            return False
        if filepath[-5:] == ".json":
            the_dict = load_results_from_json(filepath)
            print(f"type{the_dict}={type(the_dict)}")
            # Check if the graphs are in the files and included correctly.
            if "graphs_dict" not in the_dict:
                print("Results dont contain graphs_dict")
                return False
            if not graph_dict_completed_stage(
                run_config, the_dict, stage_index
            ):
                print(f"Did not complete stage:{stage_index}")
                return False
        else:
            print(f"filepath does not end in json:{filepath}")
            print(f"filepath[-5:]:{filepath[-5:]}")
    return True


def graph_dict_completed_stage(
    run_config: dict, the_dict: dict, stage_index: int
) -> bool:
    """Checks whether all expected graphs have been completed for the stages:

    <stage_index>. This check is performed by loading the graph dict
    from the graph dict, and checking whether the graph dict contains a
    list with the completed stages, and checking this list whether it
    contains the required stage number.
    """

    # Loop through expected graph names for this run_config.
    for graph_name in get_expected_stage_1_graph_names(run_config):
        graph = the_dict["graphs_dict"][graph_name]
        if graph_name not in the_dict["graphs_dict"]:
            print(f"graph_name:{graph_name} not in dict")
            return False
        if ("completed_stages") not in graph.graph:
            return False
        if not isinstance(
            graph.graph["completed_stages"],
            List,
        ):
            raise Exception(
                "Error, completed stages parameter type is not a list."
            )
        if stage_index not in graph.graph["completed_stages"]:
            return False
    return True


def load_stage_2_output_dict(relative_output_dir, filename) -> dict:
    """Loads the stage_2 output dictionary from a file.

    # TODO: Determine why the file does not yet exist at this positinoc.
    # TODO: Output dict to json format.

    :param relative_output_dir: param filename:
    :param filename:
    """
    stage_2_output_dict_filepath = relative_output_dir + filename
    with open(stage_2_output_dict_filepath, encoding="utf-8") as json_file:
        stage_2_output_dict = json.load(json_file)
    return stage_2_output_dict
