"""Method used to perform checks on whether the input is loaded correctly."""
from pathlib import Path
from typing import List

from src.export_results.helper import (
    get_expected_image_paths_stage_3,
    run_config_to_filename,
)
from src.export_results.verify_graphs import verify_completed_stages_list
from src.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from src.helper import file_exists, get_extensions_list
from src.import_results.read_json import load_results_from_json


# pylint: disable=R0912
def has_outputted_stage(run_config, stage_index: int) -> bool:
    """Verifies the required output files exist for a given simulation.

    :param run_config: param stage_index:
    :param stage_index:
    """
    expected_filepaths = []

    filename = run_config_to_filename(run_config)

    relative_output_dir = "results/"
    extensions = get_extensions_list(run_config, stage_index)
    for extension in extensions:
        if stage_index in [1, 2, 4]:  # json is checked in: stage_index == 3.

            expected_filepaths.append(
                relative_output_dir + filename + extension
            )
            # TODO: append expected_filepath to run_config per stage.

        if stage_index in [3, 4]:
            json_filepath = f"results/{filename}.json"
            if file_exists(json_filepath):
                json_filepath = (
                    f"results/{run_config_to_filename(run_config)}.json"
                )
                run_results = load_results_from_json(json_filepath, run_config)
            else:
                return False
        if stage_index == 3:

            expected_filepaths.extend(
                get_expected_image_paths_stage_3(
                    run_results["graphs_dict"].keys(),
                    run_results["graphs_dict"]["input_graph"],
                    run_config,
                    extensions,
                )
            )

        if stage_index == 4:
            if not stage_4_results_exist(run_results):
                return False

    # Check if the expected output files already exist.
    for filepath in expected_filepaths:
        if not Path(filepath).is_file():
            return False
        if filepath[-5:] == ".json":
            the_dict = load_results_from_json(filepath, run_config)
            # Check if the graphs are in the files and included correctly.
            if "graphs_dict" not in the_dict:
                return False
            if not graph_dict_completed_stage(
                run_config, the_dict, stage_index
            ):
                return False
    return True


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


def nx_graphs_contain_expected_stages(
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


def graph_dict_completed_stage(
    run_config: dict, results_nx_graphs: dict, stage_index: int
) -> bool:
    """Checks whether all expected graphs have been completed for the stages:

    <stage_index>. This check is performed by loading the graph dict
    from the graph dict, and checking whether the graph dict contains a
    list with the completed stages, and checking this list whether it
    contains the required stage number.
    """

    # Loop through expected graph names for this run_config.
    for graph_name in get_expected_stage_1_graph_names(run_config):
        graph = results_nx_graphs["graphs_dict"][graph_name]
        if graph_name not in results_nx_graphs["graphs_dict"]:
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
        verify_completed_stages_list(graph.graph["completed_stages"])
    return True


def stage_4_results_exist(run_results: dict) -> bool:
    """Verifies the stage 4 results are stored in the expected graph
    objects."""
    for graph_name, graph in run_results["graphs_dict"].items():
        if graph_name != "input_graph":
            if "results" not in graph.graph.keys():
                return False
            if not isinstance(graph.graph["results"], dict):
                raise Exception(
                    "Error, unexpected result type in graph. "
                    + f'Expected dict, yet got:{type(graph.graph["results"])}'
                )
    return True
