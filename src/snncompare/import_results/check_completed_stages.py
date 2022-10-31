"""Method used to perform checks on whether the input is loaded correctly."""
from pathlib import Path
from typing import List

from src.snncompare.export_results.helper import (
    get_expected_image_paths_stage_3,
    run_config_to_filename,
)
from src.snncompare.export_results.load_json_to_nx_graph import (
    load_pre_existing_graph_dict,
)
from src.snncompare.export_results.verify_nx_graphs import (
    verify_completed_stages_list,
)
from src.snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from src.snncompare.graph_generation.stage_1_get_input_graphs import (
    get_input_graph,
)
from src.snncompare.helper import get_expected_stages, get_extensions_list


# pylint: disable=R0912
def has_outputted_stage(
    run_config: dict, stage_index: int, to_run: dict
) -> bool:
    """Checks whether the the required output files exist, for a given
    simulation and whether their content is valid. Returns True if the file
    exists, and its content is valid, False otherwise.

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

        if stage_index == 3:
            expected_filepaths.extend(
                get_expected_image_paths_stage_3(
                    get_expected_stage_1_graph_names(run_config),
                    get_input_graph(run_config),
                    run_config,
                    extensions,
                )
            )

    # Check if the expected output files already exist.
    for filepath in expected_filepaths:
        if not Path(filepath).is_file():
            return False
        if filepath[-5:] == ".json":
            # Load the json graphs from json file to see if they exist.
            # TODO: separate loading and checking if it can be loaded.
            try:
                json_graphs = load_pre_existing_graph_dict(
                    run_config, stage_index, to_run
                )
            # pylint: disable=R0801
            except KeyError:
                return False
            except ValueError:
                return False
            except TypeError:
                return False
            if stage_index == 4:
                return has_valid_json_results(json_graphs, run_config, to_run)
    return True


def nx_graphs_have_completed_stage(
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


def stage_4_results_exist(nx_graphs: dict) -> bool:
    """Verifies the stage 4 results are stored in the expected graph
    objects."""
    for graph_name, graph in nx_graphs.items():
        if graph_name != "input_graph":
            if "results" not in graph.graph.keys():
                return False
            if not isinstance(graph.graph["results"], dict):
                raise Exception(
                    "Error, unexpected result type in graph. "
                    + f'Expected dict, yet got:{type(graph.graph["results"])}'
                )
    return True


# pylint: disable=R1702
def has_valid_json_results(
    json_graphs: dict, run_config: dict, to_run: dict
) -> bool:
    """Checks if the json_graphs contain the expected results.

    TODO: support different algorithms.
    """
    for algo_name, algo_settings in run_config["algorithm"].items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                graphnames_with_results = [
                    "snn_algo_graph",
                    "adapted_snn_graph",
                    "rad_snn_algo_graph",
                    "rad_adapted_snn_graph",
                ]
                if not set(graphnames_with_results).issubset(
                    json_graphs.keys()
                ):
                    print("Graph name not set")
                    return False

                expected_stages = get_expected_stages(
                    run_config["export_images"], 4, to_run
                )

                for graph_name, json_graph in json_graphs.items():
                    if graph_name in graphnames_with_results:

                        if expected_stages[-1] == 1:
                            graph_properties = json_graph["graph"]

                        elif expected_stages[-1] in [2, 4]:
                            # TODO: determine why this is a list of graphs,
                            # instead of a graph with list of nodes.
                            # Completed stages are only stored in the last
                            # timestep of the graph.
                            graph_properties = json_graph[-1]["graph"]
                        else:
                            raise Exception(
                                "Error, stage:{expected_stages[-1]} is "
                                "not yet supported in this check."
                            )
                        if "results" not in graph_properties.keys():
                            print(graph_properties.keys())
                            return False
                return True
            raise Exception(
                "Error, m_val setting is not of type int:"
                f'{type(algo_settings["m_val"])}'
                f'm_val={algo_settings["m_val"]}'
            )

        raise Exception(
            f"Error, algo_name:{algo_name} is not (yet) supported."
        )
    return True
