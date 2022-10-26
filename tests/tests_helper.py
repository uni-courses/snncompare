"""Contains functions used to help the tests."""
import copy
import pathlib
import random
from typing import List

import jsons
import networkx as nx

from src.export_results.export_json_results import write_dict_to_json
from src.export_results.helper import get_expected_image_paths_stage_3
from src.graph_generation.snn_algo.mdsa_snn_algo import Alipour_properties


def get_n_random_run_configs(run_configs, n: int, seed: int = None):
    """Returns n random experiment configurations."""
    if seed is not None:
        random.seed(seed)
    if n > len(run_configs):
        n = len(run_configs)
    return random.sample(run_configs, n)


def assertIsFile(path):
    """Asserts a file exists.

    Throws error if a file does not exist.
    """
    if not pathlib.Path(path).resolve().is_file():
        # pylint: disable=C0209
        raise AssertionError("File does not exist: %s" % str(path))


def assertIsNotFile(path):
    """Asserts a file does not exists.

    Throws error if the file does exist.
    """
    if pathlib.Path(path).resolve().is_file():
        # pylint: disable=C0209
        raise AssertionError("File exist: %s" % str(path))


def create_result_file_for_testing(
    json_filepath: str,
    graph_names: List[str],
    completed_stages: List[str],
    input_graph: nx.DiGraph,
    run_config: dict,
):
    """Creates a dummy .json result file that can be used to test functions
    that recognise which stages have been computed already or not.

    In particular, the has_outputted_stage() function is tested with
    this.
    """
    dummy_result = {}
    # TODO: create the output results file with the respective graphs.
    if max(completed_stages) == 1:
        dummy_result = create_results_dict_for_testing_stage_1(
            graph_names, completed_stages, input_graph, run_config
        )
    elif max(completed_stages) in [2, 3]:
        dummy_result = create_results_dict_for_testing_stage_2(
            graph_names, completed_stages, input_graph, run_config
        )
    # TODO: support stage 4 dummy creation.

    # TODO: Optional: ensure output files exists.
    write_dict_to_json(json_filepath, jsons.dump(dummy_result))

    # Verify output JSON file exists.
    filepath = pathlib.Path(json_filepath)
    assertIsFile(filepath)


def create_results_dict_for_testing_stage_1(
    graph_names: List[str],
    completed_stages: List[str],
    input_graph: nx.DiGraph,
    run_config: dict,
) -> dict:
    """Generates a dictionary with the the experiment_config, run_config and
    graphs."""
    graphs_dict = {}

    for graph_name in graph_names:
        if graph_name == "input_graph":
            # Add MDSA algorithm properties to input graph.
            graphs_dict["input_graph"] = input_graph
            graphs_dict["input_graph"].graph["alg_props"] = Alipour_properties(
                graphs_dict["input_graph"], run_config["seed"]
            ).__dict__
        else:
            # Get random nx.DiGraph graph.
            graphs_dict[graph_name] = input_graph

        # Add the completed stages as graph attribute.
        graphs_dict[graph_name].graph["completed_stages"] = completed_stages

        # Convert the nx.DiGraph object to dict.
        graphs_dict[graph_name] = graphs_dict[graph_name].__dict__

    # Merge graph and experiment and run config into a single result dict.
    dummy_result = {
        "experiment_config": None,
        "run_config": run_config,
        "graphs_dict": graphs_dict,
    }
    return dummy_result


def create_results_dict_for_testing_stage_2(
    graph_names: List[str],
    completed_stages: List[str],
    input_graph: nx.DiGraph,
    run_config: dict,
) -> dict:
    """Generates a dictionary with the the experiment_config, run_config and
    graphs."""
    graphs_dict = {}

    for graph_name in graph_names:
        if graph_name == "input_graph":
            # Add MDSA algorithm properties to input graph.
            graphs_dict["input_graph"] = [input_graph]
            graphs_dict["input_graph"][-1].graph[
                "alg_props"
            ] = Alipour_properties(
                graphs_dict["input_graph"][-1], run_config["seed"]
            ).__dict__
        else:
            # Get random nx.DiGraph graph.
            graphs_dict[graph_name] = [copy.deepcopy(input_graph)]

        graphs_dict[graph_name][-1].graph[
            "completed_stages"
        ] = completed_stages

    # Convert the nx.DiGraph object to dict.
    for graph_name in graph_names:
        if isinstance(graphs_dict[graph_name], List):
            for i, _ in enumerate(graphs_dict[graph_name]):
                graphs_dict[graph_name][i] = graphs_dict[graph_name][
                    i
                ].__dict__
        else:
            raise Exception("Error, graph for stage 2 is a list.")

    # Merge graph and experiment and run config into a single result dict.
    dummy_result = {
        "experiment_config": None,
        "run_config": run_config,
        "graphs_dict": graphs_dict,
    }
    return dummy_result

    # if completed_stages[-1] == 4:
    #     # Include dummy results in graph.
    #     # graphs_dict[graph_name].graph["results"] = {}
    #     graphs_dict[graph_name].graph["results"] = "Filler"


def create_dummy_output_images_stage_3(
    graph_names: List[str],
    input_graph: nx.DiGraph,
    run_config: dict,
    extensions,
) -> None:
    """Creates the dummy output images that would be created as output for
    stage 3, if exporting is on."""

    image_filepaths = get_expected_image_paths_stage_3(
        graph_names, input_graph, run_config, extensions
    )
    for image_filepath in image_filepaths:
        # ensure output images exist.
        with open(image_filepath, "w", encoding="utf-8"):
            pass

        # Verify output JSON file exists.
        filepath = pathlib.Path(image_filepath)
        assertIsFile(filepath)
