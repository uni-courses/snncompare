"""Converts the json graphs back into nx graphs."""
import json
from pprint import pprint
from typing import Dict, List

from networkx.readwrite import json_graph
from snnbackends.verify_nx_graphs import (
    verify_results_nx_graphs_contain_expected_stages,
)
from typeguard import typechecked

from snncompare.helper import dicts_are_equal, file_exists
from snncompare.run_config.Run_config import Run_config

from .helper import run_config_to_filename
from .verify_json_graphs import (
    verify_results_safely_check_json_graphs_contain_expected_stages,
)


@typechecked
def load_json_to_nx_graph_from_file(
    *,
    run_config: Run_config,
    stage_index: int,
    expected_stages: List[int],
) -> Dict:
    """Assumes a json file with the graphs dict of stage 1 or 2 respectively
    exists, and then loads them back as json dicts.

    Then converts those json dicts back into networkx graphs if that is
    the backend type. Then merges those nx_graphs into the results dict
    with the exp_config and run_config.
    """
    nx_graphs_dict = {}
    json_graphs_dict: Dict = load_verified_json_graphs_from_json(
        run_config=run_config, expected_stages=expected_stages
    )

    for graph_name, graph in json_graphs_dict.items():
        nx_graph = json_graph.node_link_graph(graph)
        nx_graphs_dict[graph_name] = nx_graph
    results_nx_graphs = {
        "run_config": run_config,
        "graphs_dict": nx_graphs_dict,
    }
    verify_results_nx_graphs_contain_expected_stages(
        results_nx_graphs=results_nx_graphs,
        stage_index=stage_index,
        expected_stages=expected_stages,
    )
    return nx_graphs_dict


@typechecked
def load_verified_json_graphs_from_json(
    *,
    run_config: Run_config,
    expected_stages: List[int],
) -> Dict:
    """Loads the json dict and returns the graphs of the relevant stages."""
    results_json_graphs = load_json_results(
        run_config=run_config,
        expected_stages=expected_stages,
    )

    if run_config.unique_id != results_json_graphs[
        "run_config"
    ].unique_id or not dicts_are_equal(
        left=results_json_graphs["run_config"].__dict__,
        right=run_config.__dict__,
        without_unique_id=True,
    ):
        print("Current run_config:")
        pprint(run_config.__dict__)
        print("Loaded run_config:")
        pprint(results_json_graphs["run_config"].__dict__)
        raise TabError("Error, difference in run configs, see above.")

    return results_json_graphs["graphs_dict"]


@typechecked
def load_json_results(
    *,
    run_config: Run_config,
    expected_stages: List[int],
) -> Dict:
    """Loads results from json file."""
    results_json_graphs = {}
    filename: str = run_config_to_filename(run_config_dict=run_config.__dict__)
    json_filepath = f"results/{filename}.json"

    if not file_exists(filepath=json_filepath):
        raise FileNotFoundError(f"Error, {json_filepath} was not found.")

    # Read output JSON file into dict.
    with open(json_filepath, encoding="utf-8") as json_file:
        results_json_graphs = json.load(json_file)
        json_file.close()

    verify_results_safely_check_json_graphs_contain_expected_stages(
        results_json_graphs=results_json_graphs,
        expected_stages=expected_stages,
    )

    return results_json_graphs
