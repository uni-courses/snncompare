"""Methods used to verify the graphs.

TODO: merge with from src.import_results.check_completed_stages
"""
from typing import List

import networkx as nx

from src.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
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


def verify_completed_stages_list(completed_stages: List) -> None:
    """Verifies the completed stages list is a list of consecutive positive
    integers.

    TODO: test this function.
    """
    start_stage = completed_stages[0]
    for next_stage in completed_stages[1:]:
        if start_stage != next_stage - 1:
            raise Exception(
                f"Stage indices are not consecutive:{completed_stages}."
            )
        start_stage = next_stage
    for stage in completed_stages:
        if stage < 1:
            raise Exception(
                "completed_stages contained non positive integer:"
                + f"{completed_stages}"
            )
