"""Converts the json graphs back into nx graphs."""

import json
from pprint import pprint
from typing import List

import networkx as nx
from networkx.readwrite import json_graph

from src.snncompare.export_results.check_nx_graphs import (
    json_graphs_contain_expected_stages,
)
from src.snncompare.export_results.helper import run_config_to_filename
from src.snncompare.export_results.verify_json_graphs import (
    verify_results_json_graphs_contain_correct_stages,
)
from src.snncompare.export_results.verify_nx_graphs import (
    verify_results_nx_graphs_contain_expected_stages,
)
from src.snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from src.snncompare.helper import file_exists, get_expected_stages


def json_to_digraph(json_data: dict) -> nx.DiGraph:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    TODO: remove if not used.

    """
    if json_data is not None:
        return json_graph.node_link_graph(json_data)
    raise Exception("Error, did not find json_data.")


def json_dicts_of_graph_results_exist(
    run_config: dict, expected_stages: List[int]
) -> bool:
    """Checks whether there is a json file with the graphs dict of stage 1 or 2
    respectively,"""
    # Check if the results json exists.
    filename: str = run_config_to_filename(run_config)
    json_filepath = f"results/{filename}.json"

    if file_exists(json_filepath):
        expected_graph_names = get_expected_stage_1_graph_names(run_config)
        loaded_json_graphs = load_json_graphs_from_json(run_config)
        # Check if the json graph dicts are of the right stage.
        return json_graphs_contain_expected_stages(
            loaded_json_graphs, expected_graph_names, expected_stages
        )
    return False


def load_json_to_nx_graph_from_file(
    run_config: dict, stage_index: int, to_run: dict
) -> dict:
    """Assumes a json file with the graphs dict of stage 1 or 2 respectively
    exists, and then loads them back as json dicts.

    Then converts those json dicts back into networkx graphs if that is
    the backend type. Then merges those nx_graphs into the results dict
    with the experiment_config and run_config.
    """
    nx_graphs_dict = {}
    # Load existing graph dict if it already exists, and if overwrite is off.
    json_graphs_dict: dict = load_pre_existing_graph_dict(
        run_config, stage_index, to_run
    )
    for graph_name, graph in json_graphs_dict.items():
        nx_graph = json_graph.node_link_graph(graph)
        nx_graphs_dict[graph_name] = nx_graph
    verify_results_nx_graphs_contain_expected_stages(
        {"graphs_dict": nx_graphs_dict}, stage_index, to_run
    )
    return nx_graphs_dict


def load_pre_existing_graph_dict(
    run_config: dict, stage_index: int, to_run: dict
) -> dict:
    """Returns the pre-existing graphs that were generated during earlier
    stages of the experiment.

    TODO: write tests to verify it returns the
    correct data.
    """
    if stage_index == 1:  # you should always return an empty dict.
        # TODO: fix.
        return {}
    if stage_index == 2:
        if not run_config["overwrite_sim_results"]:
            # Load graphs stages 1, 2, 3, 4
            return load_verified_json_graphs_from_json(run_config, [1, 2])
        return load_verified_json_graphs_from_json(run_config, [1])
    if stage_index == 3:
        return load_verified_json_graphs_from_json(run_config, [1, 2, 3])
    if stage_index == 4:
        return load_verified_json_graphs_from_json(
            run_config,
            get_expected_stages(
                run_config["export_images"], stage_index, to_run
            ),
        )
    raise Exception("Error, unexpected stage_index.")


def load_verified_json_graphs_from_json(
    run_config: dict, expected_stages: List[int]
) -> dict:
    """Loads the json dict and returns the graphs of the relevant stages."""
    results_json_graphs = {}

    filename: str = run_config_to_filename(run_config)
    json_filepath = f"results/{filename}.json"

    # Read output JSON file into dict.
    with open(json_filepath, encoding="utf-8") as json_file:
        results_json_graphs = json.load(json_file)
    print("contains correct stages.")
    verify_results_json_graphs_contain_correct_stages(
        results_json_graphs, expected_stages
    )

    if results_json_graphs["run_config"] != run_config:
        print("Current run_config:")
        pprint(run_config)
        print("Loaded run_config:")
        pprint(results_json_graphs["run_config"])
        raise Exception("Error, difference in experiment configs, see above.")

    return results_json_graphs["graphs_dict"]


def load_json_graphs_from_json(run_config: dict) -> dict:
    """TODO: make private.
    Loads the json dict and returns the graphs of the relevant stages."""
    results_json_graphs = {}

    filename: str = run_config_to_filename(run_config)
    json_filepath = f"results/{filename}.json"

    # Read output JSON file into dict.
    with open(json_filepath, encoding="utf-8") as json_file:
        results_json_graphs = json.load(json_file)
    return results_json_graphs["graphs_dict"]
