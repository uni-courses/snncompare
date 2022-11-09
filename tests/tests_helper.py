"""Contains functions used to help the tests."""
from __future__ import annotations

import copy
import pathlib
import random
from pathlib import PosixPath
from typing import TYPE_CHECKING, Any, List

import jsons
import networkx as nx
from snnalgorithms.sparse.MDSA.create_snns import Alipour_properties
from typeguard import typechecked

from snncompare.export_results.export_json_results import write_dict_to_json
from snncompare.export_results.helper import get_expected_image_paths_stage_3

if TYPE_CHECKING:
    from tests.simulation.test_cyclic_graph_propagation import (
        Test_cyclic_propagation_with_recurrent_edges,
    )
    from tests.simulation.test_rand_network_propagation import (
        Test_propagation_with_recurrent_edges,
    )


@typechecked
def get_n_random_run_configs(
    run_configs: list[dict], n: int, seed: int = None
) -> Any:
    """Returns n random experiment configurations.

    TODO: specify what to do if seed is None.
    """
    if seed is not None:
        random.seed(seed)
    if n > len(run_configs):
        n = len(run_configs)

    return random.sample(run_configs, n)


@typechecked
def assertIsFile(path: PosixPath) -> None:
    """Asserts a file exists.

    Throws error if a file does not exist.
    """
    if not pathlib.Path(path).resolve().is_file():
        # pylint: disable=C0209
        raise AssertionError("File does not exist: %s" % str(path))


@typechecked
def assertIsNotFile(path: str) -> None:
    """Asserts a file does not exists.

    Throws error if the file does exist.
    """
    if pathlib.Path(path).resolve().is_file():
        # pylint: disable=C0209
        raise AssertionError("File exist: %s" % str(path))


@typechecked
def create_result_file_for_testing(
    json_filepath: str,
    graph_names: list[str],
    completed_stages: list[int],
    input_graph: nx.Graph,
    run_config: dict,
) -> None:
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
    elif max(completed_stages) in [2, 3, 4]:
        dummy_result = create_results_dict_for_testing_stage_2(
            graph_names, completed_stages, input_graph, run_config
        )
    if max(completed_stages) == 4:
        add_results_to_stage_4(dummy_result)

    # TODO: support stage 4 dummy creation.

    # TODO: Optional: ensure output files exists.
    write_dict_to_json(json_filepath, jsons.dump(dummy_result))

    # Verify output JSON file exists.
    filepath = pathlib.PosixPath(json_filepath)
    assertIsFile(filepath)


@typechecked
def create_results_dict_for_testing_stage_1(
    graph_names: list[str],
    completed_stages: list[int],
    input_graph: nx.Graph,
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


@typechecked
def create_results_dict_for_testing_stage_2(
    graph_names: list[str],
    completed_stages: list[int],
    input_graph: nx.Graph,
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


@typechecked
def add_results_to_stage_4(dummy_nx_results: dict) -> None:
    """Creates dummy results in the last timestep/list element of the graph for
    stage 4."""
    for graph_name, nx_graph_list in dummy_nx_results["graphs_dict"].items():
        if graph_name != "input_graph":
            nx_graph_list[-1]["graph"]["results"] = {}
            nx_graph_list[-1]["graph"]["results"] = "Filler"


@typechecked
def create_dummy_output_images_stage_3(
    graph_names: list[str],
    input_graph: nx.Graph,
    run_config: dict,
    extensions: list[str],
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
        filepath = pathlib.PosixPath(image_filepath)
        assertIsFile(filepath)


@typechecked
def get_cyclic_graph_without_directed_path() -> nx.DiGraph:
    """Gets a cyclic graph with nodes that cannot be reached following the
    directed edges, to test if the Lava simulation imposes some requirements on
    the graph properties."""
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        color="w",
    )
    graph.add_edges_from(
        [
            (1, 0),
            (1, 2),
            (3, 2),
            (4, 3),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 5),
            (8, 7),
        ],
        weight=float(10),
    )
    return graph


def compare_static_snn_properties(
    test_object: (
        Test_propagation_with_recurrent_edges
        | Test_cyclic_propagation_with_recurrent_edges
    ),
    G: nx.DiGraph,
    t: int = 0,
) -> None:
    """Performs comparison of static neuron properties at each timestep.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    for node in G.nodes:
        lava_neuron = G.nodes[node]["lava_LIF"]
        nx_neuron = G.nodes[node]["nx_LIF"][t]

        # Assert bias is equal.
        test_object.assertEqual(
            lava_neuron.bias_mant.get(), nx_neuron.bias.get()
        )

        # dicts
        # print(f"lava_neuron.__dict__={lava_neuron.__dict__}")
        # print(f"lava_neuron.__dict__={nx_neuron.__dict__}")

        # Assert du is equal.
        test_object.assertEqual(lava_neuron.du.get(), nx_neuron.du.get())
        #

        # Assert dv is equal.
        test_object.assertEqual(lava_neuron.dv.get(), nx_neuron.dv.get())

        # print(f"lava_neuron.name.get()={lava_neuron.name.get()}")
        # print(f"lava_neuron.name.get()={nx_neuron.name.get()}")
        # Assert name is equal.
        # self.assertEqual(lava_neuron.name, nx_neuron.name)

        # Assert vth is equal.
        test_object.assertEqual(lava_neuron.vth.get(), nx_neuron.vth.get())

        # Assert v_reset is equal. (Not yet implemented in Lava.)
        # self.assertEqual(
        #    lava_neuron.v_reset.get(), nx_neuron.v_reset.get()
        # )
