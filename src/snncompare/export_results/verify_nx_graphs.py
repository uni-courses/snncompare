"""Methods used to verify the graphs."""
from typing import List

import networkx as nx

from src.snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from src.snncompare.helper import get_expected_stages
from src.snncompare.verification_generic import verify_completed_stages_list


def verify_results_nx_graphs_contain_expected_stages(
    results_nx_graphs: dict, stage_index: int, to_run: dict
) -> None:
    """Verifies that the nx_graphs dict contains the expected completed stages
    in each nxgraph.graph dict.

    Throws an error otherwise.
    """
    for graph_name, nx_graph in results_nx_graphs["graphs_dict"].items():
        expected_stages = get_expected_stages(
            results_nx_graphs["run_config"]["export_images"],
            stage_index,
            to_run,
        )
        verify_nx_graph_contains_correct_stages(
            graph_name, nx_graph, expected_stages
        )


# pylint: disable=R0912
def verify_results_nx_graphs(
    results_nx_graphs: dict, run_config: dict
) -> None:
    """Verifies the results that are loaded from json file are of the expected
    format.

    Does not verify whether any expected stages have been completed. #
    TODO: include check on graph types based on the completed stages. If
    a graph contains completed_stages==[1], then the type should be:
    nx.Graph for graph_name=="input_graph", and nx.DiGraph otherwise. If
    completed_stages of a graph is larger than 1, the value of the
    results["graphs_dict"] should be of type list, with nx.Graph and
    nx.DiGraphs respectively. # TODO: break this check into separate
    functions.
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
    for graph_name, nx_graph in results_nx_graphs["graphs_dict"].items():
        if graph_name == "input_graph":
            if isinstance(nx_graph, List):
                for nx_graph_frame in nx_graph:
                    if not isinstance(nx_graph_frame, nx.Graph):
                        raise Exception(
                            "Error, input nx_graph_frame changed to type:"
                            + f"{type(nx_graph_frame)}"
                        )
            elif not isinstance(nx_graph, nx.Graph):
                raise Exception(
                    f"Error, input nx_graph changed to type:{type(nx_graph)}"
                )
        else:
            if isinstance(nx_graph, List):
                for nx_graph_frame in nx_graph:
                    if not isinstance(nx_graph_frame, nx.DiGraph):
                        raise ValueError(
                            "Error, the results_nx_graphs object contains a "
                            + f"graph:{graph_name} that is not of type: nx.DiG"
                            + f"raph,instead, it is:{type(nx_graph_frame)}"
                        )

                    # Verify each graph has the right completed stages
                    # attribute.
                    verify_completed_stages_list(
                        nx_graph_frame.graph["completed_stages"]
                    )
            elif isinstance(nx_graph, nx.DiGraph):
                # Verify each graph has the right completed stages attribute.
                verify_completed_stages_list(
                    nx_graph.graph["completed_stages"]
                )
            elif not isinstance(nx_graph, nx.DiGraph):
                raise ValueError(
                    "Error, the results_nx_graphs object contains a "
                    + f"graph:{graph_name} that is not of type: nx.DiGraph:"
                    + f"instead, it is of type:{type(nx_graph)}"
                )


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
                    f"stages:{expected_stages}. Instead, it contained:"
                    f'{nx_graph.graph["completed_stages"]}'
                )
