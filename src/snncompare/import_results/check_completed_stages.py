"""Method used to perform checks on whether the input is loaded correctly."""
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional

from snnbackends.verify_nx_graphs import verify_completed_stages_list
from typeguard import typechecked

from snncompare.exp_config.run_config.Run_config import Run_config
from snncompare.graph_generation.stage_1_create_graphs import (
    get_input_graph_of_run_config,
)

from ..export_results.helper import (
    get_expected_image_paths_stage_3,
    run_config_to_filename,
)
from ..export_results.load_json_to_nx_graph import (
    load_verified_json_graphs_from_json,
)
from ..export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from ..helper import get_expected_stages


@typechecked
def has_outputted_stage_jsons(
    *,
    expected_stages: List[int],
    run_config: Run_config,
    stage_index: int,
) -> bool:
    """Checks if a stage has been outputted or not."""
    if stage_index not in [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Error, stage_index:{stage_index} was not in range:{[1,2,3,4,5]}"
        )
    expected_filepaths = get_expected_files(run_config=run_config)
    # print(f'expected_filepaths=')
    # pprint(expected_filepaths)

    if not expected_files_exist(expected_filepaths=expected_filepaths):
        print(f"files did not exist={expected_filepaths}")
        return False

    if not expected_jsons_are_valid(
        expected_filepaths=expected_filepaths,
        expected_stages=expected_stages,
        run_config=run_config,
        stage_index=stage_index,
    ):
        print(f"Json not valid.={expected_filepaths}")
        return False
    return True


@typechecked
def has_outputted_images(
    *,
    results_nx_graphs: Dict,
    run_config: Run_config,
) -> bool:
    """Verifies whether the images of stage 3 have been outputted."""

    expected_image_paths: List[str] = get_expected_image_paths_stage_3(
        nx_graphs_dict=results_nx_graphs["graphs_dict"],
        input_graph=get_input_graph_of_run_config(run_config=run_config),
        run_config=run_config,
        extensions=["png"],
    )

    if not expected_files_exist(expected_filepaths=expected_image_paths):
        print(f"files did not exist={expected_image_paths}")
        return False
    return True


@typechecked
def get_expected_files(*, run_config: Run_config) -> List[str]:
    """Returns the list of expected files for a run configuration."""
    expected_filepaths = []
    filename = run_config_to_filename(run_config=run_config)
    relative_output_dir = "results/"

    output_file_extensions = [".json"]

    for extension in output_file_extensions:
        expected_filepaths.append(relative_output_dir + filename + extension)
    return expected_filepaths


@typechecked
def expected_files_exist(
    *, expected_filepaths: List[str], verbose: Optional[bool] = False
) -> bool:
    """Returns True if a file exists, False otherwise."""

    # Check if the expected output files already exist.
    for filepath in expected_filepaths:
        if not Path(filepath).is_file():
            if verbose:
                print(f"File={filepath} missing.")
            return False
    return True


@typechecked
def expected_jsons_are_valid(
    *,
    expected_filepaths: List[str],
    expected_stages: List[int],
    run_config: Run_config,
    stage_index: int,
) -> bool:
    """Checks for all expected json files whether they can successfully be
    loaded into this experiment."""
    for filepath in expected_filepaths:
        if filepath[-5:] == ".json":
            if not expected_json_content_is_valid(
                expected_stages=expected_stages,
                filepath=filepath,
                run_config=run_config,
                stage_index=stage_index,
                verbose=False,
            ):
                return False
    return True


@typechecked
def expected_json_content_is_valid(
    *,
    expected_stages: List[int],
    filepath: str,
    run_config: Run_config,
    stage_index: int,
    verbose: Optional[bool] = False,
) -> bool:
    """Safely checks if a json file can successfully be loaded into this
    experiment."""

    if filepath[-5:] == ".json":
        # Load the json graphs from json file to see if they exist.
        # TODO: separate loading and checking if it can be loaded.
        try:
            json_graphs = load_verified_json_graphs_from_json(
                run_config=run_config, expected_stages=expected_stages
            )
        # pylint: disable=R0801
        except KeyError as k:
            if verbose:
                print(f"KeyError for: {filepath}: {repr(k)}")
            return False
        except ValueError as v:
            if verbose:
                print(f"ValueError for: {filepath}:")
                pprint(repr(v))
            return False
        except TypeError as t:
            if verbose:
                print(f"TypeError for: {filepath}: {repr(t)}")
            return False
        if stage_index == 4:
            return has_valid_json_results(
                json_graphs=json_graphs, run_config=run_config
            )
    else:
        raise FileNotFoundError(f"Error, the file:{filepath} is not a json.")
    return True


@typechecked
def nx_graphs_have_completed_stage(
    *,
    run_config: Run_config,
    results_nx_graphs: Dict,
    stage_index: int,
) -> bool:
    """Checks whether all expected graphs have been completed for the stages:

    <stage_index>. This check is performed by loading the graph dict
    from the graph Dict, and checking whether the graph dict contains a
    list with the completed stages, and checking this list whether it
    contains the required stage number.
    """

    # Loop through expected graph names for this run_config.
    for graph_name in get_expected_stage_1_graph_names(run_config=run_config):
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
        verify_completed_stages_list(
            completed_stages=graph.graph["completed_stages"]
        )
    return True


# pylint: disable=R1702
@typechecked
def has_valid_json_results(
    *,
    json_graphs: Dict,
    run_config: Run_config,
) -> bool:
    """Checks if the json_graphs contain the expected results.

    TODO: support different algorithms.
    """
    for algo_name, algo_settings in run_config.algorithm.items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                graphnames_with_results = get_expected_stage_1_graph_names(
                    run_config=run_config
                )
                graphnames_with_results.remove("input_graph")
                if not set(graphnames_with_results).issubset(
                    json_graphs.keys()
                ):
                    return False

                expected_stages = get_expected_stages(
                    stage_index=4,
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
                            graph_properties = json_graph["graph"]
                        else:
                            raise Exception(
                                "Error, stage:{expected_stages[-1]} is "
                                "not yet supported in this check."
                            )
                        if "results" not in graph_properties.keys():
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
