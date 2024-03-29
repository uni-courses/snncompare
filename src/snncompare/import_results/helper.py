"""Helps importing and exporting."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import networkx as nx
from typeguard import typechecked

# if TYPE_CHECKING:
from snncompare.run_config.Run_config import Run_config


@typechecked
def prepare_target_file_output(
    *,
    output_dir: str,
    some_graph: Union[nx.Graph, nx.DiGraph],
    rad_affected_neurons_hash: Optional[str] = None,
    rand_nrs_hash: Optional[str] = None,
) -> Tuple[bool, str]:
    """Creates the relative filepath if it does not exist.

    Returns True if the target file already exists, False otherwise.
    """

    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=some_graph)
    additional_hashes: str = ""
    if rand_nrs_hash is not None:
        additional_hashes = f"{additional_hashes}_rand_{rand_nrs_hash}"
    if rad_affected_neurons_hash is not None:
        additional_hashes = (
            f"{additional_hashes}_rad_{rad_affected_neurons_hash}"
        )

    output_filepath: str = (
        f"{output_dir}{isomorphic_hash}{additional_hashes}.json"
    )

    if not Path(output_filepath).is_file():
        create_relative_path(some_path=output_dir)
    return Path(output_filepath).is_file(), output_filepath


@typechecked
def get_isomorphic_graph_hash(*, some_graph: nx.Graph) -> str:
    """Returns the hash of the isomorphic graph. Meaning all graphs that have
    the same shape, return the same hash string.

    An isomorphic graph is one that looks the same as another/has the
    same shape as another, (if you are blind to the node numbers).
    """
    isomorphic_hash: str = (
        nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(some_graph)
    )
    return isomorphic_hash


@typechecked
def create_relative_path(*, some_path: str) -> None:
    """Exports Run_config to a json file."""
    absolute_path: str = f"{os.getcwd()}/{some_path}"
    # Create subdirectory in results dir.
    if not os.path.exists(absolute_path):
        # Path(absolute_path).mkdir(parents=True, exist_ok=True)
        os.makedirs(absolute_path)

    if not os.path.exists(absolute_path):
        raise NotADirectoryError(f"{absolute_path} does not exist.")


@typechecked
def seed_rand_nrs_hash_file_exists(
    *,
    output_category: str,
    run_config: Run_config,
) -> Tuple[bool, str]:
    """Returns two tuples which contain: graph file exists, and the graph
    filepath.

    First tuple for the unadapted snn, the second tuple for the adapted
    tuple.
    """
    algorithm_name, algorithm_parameter = get_algorithm_description(
        run_config=run_config
    )

    output_path: str = (
        f"results/stage1/{algorithm_name}_"
        + f"{algorithm_parameter}/no_adaptation/{output_category}/"
        + f"{run_config.seed}.txt"
    )
    return Path(output_path).is_file(), output_path


@typechecked
def seed_rad_neurons_hash_file_exists(
    *,
    output_category: str,
    run_config: Run_config,
    with_adaptation: bool,
) -> Tuple[bool, str]:
    """Returns two tuples which contain: graph file exists, and the graph
    filepath.

    First tuple for the unadapted snn, the second tuple for the adapted
    tuple.
    """
    algorithm_name, algorithm_parameter = get_algorithm_description(
        run_config=run_config
    )

    if with_adaptation:
        output_path: str = (
            f"results/stage1/{algorithm_name}_"
            + f"{algorithm_parameter}/{run_config.adaptation.adaptation_type}_"
            + f"{run_config.adaptation.redundancy}/{output_category}/"
            + f"{run_config.seed}.txt"
        )
    else:
        output_path = (
            f"results/stage1/{algorithm_name}_"
            + f"{algorithm_parameter}/no_adaptation/{output_category}/"
            + f"{run_config.seed}.txt"
        )
    return Path(output_path).is_file(), output_path


@typechecked
def simsnn_files_exists_and_get_path(
    *,
    output_category: str,
    run_config: Run_config,
    input_graph: nx.Graph,
    with_adaptation: bool,
    stage_index: int,
    rad_affected_neurons_hash: Optional[str] = None,
    rand_nrs_hash: Optional[str] = None,
) -> Tuple[bool, str]:
    """Returns two tuples which contain: graph file exists, and the graph
    filepath.

    First tuple for the unadapted snn, the second tuple for the adapted
    tuple.
    """
    algorithm_name, algorithm_parameter = get_algorithm_description(
        run_config=run_config
    )

    if algorithm_name == "MDSA":
        if with_adaptation:
            # Import adapted snn.
            output_dir: str = (
                f"results/stage{stage_index}/{algorithm_name}_"
                + f"{algorithm_parameter}/"
                + f"{run_config.adaptation.adaptation_type}_"
                + f"{run_config.adaptation.redundancy}/{output_category}/"
            )
            (
                snn_algo_graph_exists,
                snn_algo_graph_filepath,
            ) = prepare_target_file_output(
                output_dir=output_dir,
                some_graph=input_graph,
                rad_affected_neurons_hash=rad_affected_neurons_hash,
                rand_nrs_hash=rand_nrs_hash,
            )
            # print("With adaptation=True")
            # print(snn_algo_graph_filepath)
        else:
            # Import default snn.

            output_dir = (
                f"results/stage{stage_index}/{algorithm_name}_"
                + f"{algorithm_parameter}/no_adaptation/{output_category}/"
            )
            (
                snn_algo_graph_exists,
                snn_algo_graph_filepath,
            ) = prepare_target_file_output(
                output_dir=output_dir,
                some_graph=input_graph,
                rad_affected_neurons_hash=rad_affected_neurons_hash,
                rand_nrs_hash=rand_nrs_hash,
            )
            # print("With adaptation=False")
            # print(snn_algo_graph_filepath)
            # print("Does the snn filepath include the rand_nrs hash?")
        return (snn_algo_graph_exists, snn_algo_graph_filepath)

    raise NotImplementedError(f"Error:{algorithm_name} is not yet supported.")


@typechecked
def get_algorithm_description(*, run_config: Run_config) -> Tuple[str, int]:
    """Returns the algorithm name and value as a single string."""
    algorithm_name: str = get_single_element(
        some_list=list(run_config.algorithm.keys())
    )

    algorithm_parameter: int = get_single_element(
        some_list=list(run_config.algorithm[algorithm_name].values())
    )
    return algorithm_name, algorithm_parameter


@typechecked
def get_single_element(*, some_list: List) -> Union[str, int]:
    """Asserts a list has only one element and returns that element."""
    assert_has_one_element(some_list=some_list)
    return some_list[0]


@typechecked
def assert_has_one_element(*, some_list: List) -> None:
    """Asserts a list contains only 1 element."""
    if len(some_list) != 1:
        raise ValueError(
            "Error the number of algorithms in a single run_config was not 1:"
            + f"{some_list}"
        )


@typechecked
def file_contains_line(*, filepath: str, expected_line: str) -> bool:
    """Returns True if a file exists and contains a line at least once, False
    otherwise."""
    with open(filepath, encoding="utf-8") as txt_file:
        for _, line in enumerate(txt_file, 1):
            if expected_line in line:
                return True
        txt_file.close()
    return False
