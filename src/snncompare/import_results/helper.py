"""Helps importing and exporting."""

import os
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union

import networkx as nx
from typeguard import typechecked

# if TYPE_CHECKING:
from snncompare.run_config.Run_config import Run_config


@typechecked
def prepare_target_file_output(
    *, output_dir: str, some_graph: Union[nx.Graph, nx.DiGraph]
) -> Tuple[bool, str]:
    """Creates the relative filepath if it does not exist.

    Returns True if the target file already exists, False otherwise.
    """

    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=some_graph)
    output_filepath: str = f"{output_dir}{isomorphic_hash}.json"

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
def simsnn_files_exists_and_get_path(
    *,
    output_category: str,
    run_config: Run_config,
    input_graph: nx.Graph,
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

    adaptation_name, adaptation_parameter = get_adaptation_description(
        run_config=run_config
    )
    if algorithm_name == "MDSA":
        if with_adaptation:
            # Import adapted snn.
            output_dir: str = (
                f"results/{algorithm_name}_{algorithm_parameter}/"
                + f"{adaptation_name}_{adaptation_parameter}/{output_category}"
                + "/"
            )
            (
                snn_algo_graph_exists,
                snn_algo_graph_filepath,
            ) = prepare_target_file_output(
                output_dir=output_dir, some_graph=input_graph
            )
        else:
            # Import default snn.
            output_dir = (
                f"results/{algorithm_name}_{algorithm_parameter}"
                + f"/no_adaptation/{output_category}/"
            )
            (
                snn_algo_graph_exists,
                snn_algo_graph_filepath,
            ) = prepare_target_file_output(
                output_dir=output_dir, some_graph=input_graph
            )
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
def get_adaptation_description(*, run_config: Run_config) -> Tuple[str, int]:
    """Returns the adaptation name and value as a single string."""
    adaptation_name: str = get_single_element(
        some_list=list(run_config.adaptation.keys())
    )

    adaptation_parameter: int = run_config.adaptation[adaptation_name]

    if adaptation_parameter == 0:
        pprint(run_config.__dict__)
        raise ValueError(
            "Error, redundancy=0 is a duplicate of original graph."
        )
    return adaptation_name, adaptation_parameter


def get_radiation_description(*, run_config: Run_config) -> Tuple[str, int]:
    """Returns the radiation name and value as a single string."""
    radiation_name: str = get_single_element(
        some_list=list(run_config.radiation.keys())
    )

    radiation_parameter: int = run_config.radiation[radiation_name]

    if radiation_parameter == 0:
        pprint(run_config.__dict__)
        raise ValueError(
            "Error, redundancy=0 is a duplicate of original graph."
        )
    return radiation_name, radiation_parameter


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
