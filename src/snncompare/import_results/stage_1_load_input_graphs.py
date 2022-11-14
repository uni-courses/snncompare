"""Parses the graph json files to recreate the graphs."""
import json
from pprint import pprint

from typeguard import typechecked

from ..export_results.helper import run_config_to_filename
from ..export_results.load_json_to_nx_graph import (
    dicts_are_equal,
    json_to_digraph,
)
from ..export_results.verify_stage_1_graphs import assert_graphs_are_in_dict
from ..helper import get_extensions_list
from .read_json import load_results_from_json


@typechecked
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

    stage_1_dict = load_results_from_json(filepath, run_config)

    # Split the dictionary into three separate dicts.
    loaded_run_config = stage_1_dict["run_config"]

    # Verify the run_dict is valid.
    if not dicts_are_equal(
        run_config, loaded_run_config, without_unique_id=True
    ):
        print("Current run_config:")
        pprint(run_config)
        print("Loaded run_config:")
        pprint(loaded_run_config)
        raise Exception("Error, difference in run configs, see above.")

    # Verify the graph names are as expected for the graph name.
    assert_graphs_are_in_dict(run_config, stage_1_dict["graphs_dict"], 1)

    stage_1_graphs = {}
    # Converting back into graphs
    for graph_name, some_graph in stage_1_dict["graphs_dict"][
        "stage_1"
    ].items():
        # TODO: update typing and name of "stage1 graphs"
        stage_1_graphs[graph_name] = json_to_digraph(some_graph)
    return stage_1_graphs


@typechecked
def load_stage_2_output_dict(relative_output_dir: str, filename: str) -> dict:
    """Loads the stage_2 output dictionary from a file.

    # TODO: Determine why the file does not yet exist at this positinoc.
    # TODO: Output dict to json format.

    :param relative_output_dir: param filename:
    :param filename:
    """
    stage_2_output_dict_filepath = relative_output_dir + filename
    with open(stage_2_output_dict_filepath, encoding="utf-8") as json_file:
        stage_2_output_dict = json.load(json_file)
        json_file.close()
    return stage_2_output_dict
