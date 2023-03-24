"""Converts the generated snns into json dicts that can be exported."""
from typing import Dict

from snnbackends.simsnn.export import Json_dict_simsnn
from snnbackends.verify_nx_graphs import (
    verify_results_nx_graphs,
    verify_results_nx_graphs_contain_expected_stages,
)
from typeguard import typechecked

from .export_nx_graph_to_json import convert_digraphs_to_json


@typechecked
def prepare_stage_1_and_2_nx_lif_output(
    *, results_nx_graphs: Dict, stage_index: int
) -> Dict:
    """Converts the generated nx snns into json dicts that can be exported."""
    verify_results_nx_graphs(
        results_nx_graphs=results_nx_graphs,
        run_config=results_nx_graphs["run_config"],
    )

    verify_results_nx_graphs_contain_expected_stages(
        results_nx_graphs=results_nx_graphs, stage_index=stage_index
    )

    results_json_graphs = convert_digraphs_to_json(
        results_nx_graphs=results_nx_graphs, stage_index=stage_index
    )
    return results_json_graphs


@typechecked
def prepare_stage_1_and_2_simsnn_output(
    *,
    graphs_dict: Dict,
) -> Dict:
    """Converts the Simulator object of the simsnn into a json dict.

    TODO: Create verification for simsnn.
    """

    json_simsnn_graphs: Dict = {}
    for graph_name, simulator in graphs_dict.items():
        if graph_name != "input_graph":
            json_simsnn_graphs[graph_name] = Json_dict_simsnn(simulator)
    return json_simsnn_graphs
