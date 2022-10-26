"""Methods used to verify the graphs."""
from typing import List

import networkx as nx

from src.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from src.verification_generic import verify_completed_stages_list


def verify_results_nx_graphs_contain_expected_stages(
    results_nx_graphs: dict, stage_index: int
):
    """Verifies that the nx_graphs dict contains the expected completed stages
    in each nxgraph.graph dict.

    Throws an error otherwise.
    """
    for graph_name, nx_graph in results_nx_graphs["graphs_dict"].items():
        expected_stages = list(range(1, stage_index + 1))
        verify_nx_graph_contains_correct_stages(
            graph_name, nx_graph, expected_stages
        )


def verify_results_nx_graphs(results_nx_graphs: dict, run_config: dict):
    """Verifies the results that are loaded from json file are of the expected
    format.

    Does not verify whether any expected stages have been completed.
    """
    stage_1_graph_names = get_expected_stage_1_graph_names(run_config)
    # Verify the 3 dicts are in the result dict.
    if "experiment_config" not in results_nx_graphs.keys():
        raise Exception(
            "Error, experiment_config not in run_result keys:"
            + f"{results_nx_graphs}"
        )

    if "run_config" not in results_nx_graphs.keys():
        raise Exception(
            "Error, run_config not in results_nx_graphs keys:"
            + f"{results_nx_graphs}"
        )
    if "graphs_dict" not in results_nx_graphs.keys():
        raise Exception(
            "Error, graphs_dict not in results_nx_graphs keys:"
            + f"{results_nx_graphs}"
        )

    # Verify the right graphs are within the graphs_dict.
    for graph_name in stage_1_graph_names:
        if graph_name not in results_nx_graphs["graphs_dict"].keys():
            raise Exception(
                f"Error, {graph_name} not in results_nx_graphs keys:"
                + f"{results_nx_graphs}"
            )

    # Verify each graph is of the networkx type.
    for graph_name, graph in results_nx_graphs["graphs_dict"].items():
        if graph_name == "input_graph":
            if not isinstance(graph, nx.Graph):
                raise Exception(
                    f"Error, input graph changed to type:{type(graph)}"
                )
        else:
            if not isinstance(graph, nx.DiGraph):
                raise ValueError(
                    "Error, the results_nx_graphs object contains a "
                    + f"graph:{graph_name} that is not of type: nx.DiGraph:"
                    + f"{type(graph)}"
                )

        # Verify each graph has the right completed stages attribute.
        verify_completed_stages_list(graph.graph["completed_stages"])


def verify_nx_graph_contains_correct_stages(
    graph_name: str, nx_graph: nx.DiGraph, expected_stages: List[int]
) -> None:
    """Verifies the networkx graph object contains the correct completed
    stages."""
    if "completed_stages" in nx_graph.graph.keys():
        for expected_stage in expected_stages:
            if expected_stage not in nx_graph.graph["completed_stages"]:
                raise ValueError(
                    f"Error, {graph_name} did not contain the expected "
                    f"stages:{expected_stages}."
                )
