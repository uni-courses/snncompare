"""Contains functions used to help the tests."""
import pathlib
import random
from typing import List

import jsons

from src.export_results.export_json_results import write_dict_to_json
from src.graph_generation.get_graph import get_networkx_graph_of_2_neurons
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
    run_config: dict,
):
    """Creates a dummy result file that can be used to test functions that
    recognise which stages have been computed already or not.

    In particular, the performed_stage() function is tested with this.
    """

    # TODO: create the output results file with the respective graphs.
    dummy_result: dict = create_results_dict_for_testing(
        graph_names, completed_stages, run_config
    )

    # TODO: Optional: ensure output files exists.
    write_dict_to_json(json_filepath, jsons.dump(dummy_result))

    # Verify output JSON file exists.
    filepath = pathlib.Path(json_filepath)
    assertIsFile(filepath)


def create_results_dict_for_testing(
    graph_names: List[str], completed_stages: List[str], run_config: dict
) -> dict:
    """Generates a dictionary with the the experiment_config, run_config and
    graphs."""
    graphs_dict = {}

    for graph_name in graph_names:
        if graph_name == "input_graph":
            # Add MDSA algorithm properties to input graph.
            graphs_dict["input_graph"] = get_networkx_graph_of_2_neurons()
            graphs_dict["input_graph"].graph["alg_props"] = Alipour_properties(
                graphs_dict["input_graph"], run_config["seed"]
            ).__dict__
        else:
            # Get random nx.DiGraph graph.
            graphs_dict[graph_name] = get_networkx_graph_of_2_neurons()
            # Add the completed stages as graph attribute.
        graphs_dict[graph_name].graph["completed_stages"] = completed_stages

        # Convert the nx.DiGraph object to dict.
        graphs_dict[graph_name] = graphs_dict[graph_name].__dict__

    # Merge graph and experiment and run config into a single result dict.
    dummy_result = {
        "experiment_config": None,
        "run_config": None,
        "graphs_dict": graphs_dict,
    }
    return dummy_result
