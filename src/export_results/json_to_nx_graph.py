"""Converts the json graphs back into nx graphs."""

import json
from pprint import pprint
from typing import List

import networkx as nx
from networkx.readwrite import json_graph

from src.export_results.helper import run_config_to_filename
from src.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from src.helper import file_exists


def json_to_digraph(json_data):
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
    run_config: dict, stage_index: int
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
        run_config, stage_index
    )
    for graph_name, graph in json_graphs_dict.items():
        expected_stages = list(range(1, stage_index + 1))
        nx_graph = json_graph.node_link_graph(graph)
        verify_nx_graph_contains_correct_stages(
            graph_name, nx_graph, expected_stages
        )
        nx_graphs_dict[graph_name] = nx_graph
    return nx_graphs_dict


def load_pre_existing_graph_dict(run_config, stage_index) -> dict:
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
        return load_verified_json_graphs_from_json(run_config, [1, 2, 3, 4])
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


def results_with_json_graphs_contain_correct_stages(
    results_json_graphs: dict, expected_stages: List[int]
):
    """Returns True if the json graphs are valid, False otherwise."""
    try:
        verify_results_json_graphs_contain_correct_stages(
            results_json_graphs, expected_stages
        )
        return True
    # pylint: disable=W0702
    except KeyError:
        return False
    except ValueError:
        return False


def json_graphs_contain_correct_stages(
    json_graphs: dict, expected_stages: List[int], run_config: dict
):
    """Returns True if the json graphs are valid, False otherwise."""
    try:
        verify_json_graphs_dict_contain_correct_stages(
            json_graphs, expected_stages, run_config
        )
        return True
    # pylint: disable=W0702
    except KeyError:
        return False
    except ValueError:
        return False


def verify_results_json_graphs_contain_correct_stages(
    results_json_graphs: dict, expected_stages: List[int]
) -> None:
    """Checks whether the loaded graphs from json contain at least the expected
    stages for this stage of the experiment."""

    if "experiment_config" not in results_json_graphs:
        raise KeyError("Error, key: experiment_config not in output_dict.")
    if "run_config" not in results_json_graphs:
        raise KeyError("Error, key: run_config not in output_dict.")
    if "graphs_dict" not in results_json_graphs:
        raise KeyError("Error, key: graphs_dict not in output_dict.")
    verify_json_graphs_dict_contain_correct_stages(
        results_json_graphs["graphs_dict"],
        expected_stages,
        results_json_graphs["run_config"],
    )


def verify_json_graphs_dict_contain_correct_stages(
    json_graphs: dict, expected_stages: List[int], run_config: dict
) -> None:
    """Verifies the json graphs dict contains the expected stages in each
    graph."""
    for expected_stage in expected_stages:
        for graph_name, graph in json_graphs.items():
            #            print(f'graph={graph}')
            print(f"{graph_name}, type={type(graph)}")
            # for elem in graph:
            #    print(elem)
            if graph["graph"]["completed_stages"]:
                if expected_stage not in graph["graph"]["completed_stages"]:
                    raise ValueError(
                        "Error, for run_config: "
                        + f"{run_config}, the expected "
                        + f"stage:{expected_stage}, was not found in "
                        + "the completed stages:"
                        + f'{graph["graph"]["completed_stages"]} '
                        + f"that were loaded from graph: {graph_name}."
                    )


def verify_nx_graph_contains_correct_stages(
    graph_name: str, nx_graph: nx.DiGraph, expected_stages: List[int]
) -> None:
    """Verifies the networkx graph object contains the correct completed
    stages."""
    if "completed_stages" in nx_graph.graph.keys():
        for expected_stage in expected_stages:
            if expected_stage not in nx_graph.graph["completed_stages"]:
                raise ValueError(
                    f"Error, {graph_name} did not contain the expected "
                    f"stages:{expected_stages}."
                )


def json_graphs_contain_expected_stages(
    json_graphs: dict,
    expected_graph_names: List[str],
    expected_stages: List[int],
) -> bool:
    """Verifies for each of the expected graph names is in the json_graphs
    dict, and then verifies each of those graphs contain the completed
    stages."""
    for expected_graph_name in expected_graph_names:
        if expected_graph_name not in json_graphs.keys():
            return False
        for expected_stage in expected_stages:
            # More completed stages than expected is ok.
            if (
                expected_stage
                not in json_graphs[expected_graph_name]["completed_stages"]
            ):
                return False
    return True
