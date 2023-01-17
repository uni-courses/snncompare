"""Determines which algorithm(s) are ran, then determines for which graphs the
results are to be computed, and then computes the results per graph, against
the expected results. A different algorithm may return a different dictionary.

The MDSA algorithm results will consist of a list of nodes per used
graph that have been selected according to Alipour, and according to the
respective SNN graph.
"""
import copy
from typing import Dict

import networkx as nx
from snnalgorithms.sparse.MDSA.apply_results_to_graphs import (
    set_mdsa_snn_results,
)
from snnbackends.verify_nx_graphs import (
    verify_nx_graph_contains_correct_stages,
)
from typeguard import typechecked

from snncompare.exp_setts.run_config.Run_config import Run_config

from ..export_results.Output_stage_34 import output_stage_files_3_and_4
from ..helper import add_stage_completion_to_graph, get_expected_stages
from ..import_results.check_completed_stages import (
    nx_graphs_have_completed_stage,
)


@typechecked
def set_results(
    run_config: Run_config,
    stage_2_graphs: Dict,
) -> None:
    """Gets the results for the algorithms that have been ran."""
    for algo_name, algo_settings in run_config.algorithm.items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                perform_mdsa_results_computation_if_needed(
                    m_val=algo_settings["m_val"],
                    run_config=run_config,
                    stage_2_graphs=stage_2_graphs,
                )
            else:
                raise Exception(
                    "Error, m_val setting is not of type int:"
                    f'{type(algo_settings["m_val"])}'
                    f'm_val={algo_settings["m_val"]}'
                )
        else:
            raise Exception(
                f"Error, algo_name:{algo_name} is not (yet) supported."
            )


@typechecked
def perform_mdsa_results_computation_if_needed(
    m_val: int,
    run_config: Run_config,
    stage_2_graphs: Dict,
) -> None:
    """Performs result computation if the results are not in the graph yet."""
    for nx_graph in stage_2_graphs.values():
        if (
            4 not in nx_graph.graph["completed_stages"]
            or run_config.overwrite_sim_results
        ):
            set_mdsa_snn_results(m_val, run_config, stage_2_graphs)

            # Indicate the graphs have completed stage 1.
            for nx_graph in stage_2_graphs.values():
                add_stage_completion_to_graph(nx_graph, 4)


@typechecked
def export_results_to_json(
    export_images: bool,
    results_nx_graphs: Dict,
    stage_index: int,
    to_run: Dict,
) -> None:
    """Integrates the results per graph type into the graph, then export the
    results dictionary (again) into the stage 4 folder."""
    # Create new independent graphs dict to include the results.
    # TODO: determine why/don't duplicate.
    stage_4_graphs = copy.deepcopy(results_nx_graphs["graphs_dict"])
    # Embed results into snn graphs
    for graph_name in results_nx_graphs["graphs_dict"].keys():
        if graph_name == "snn_algo_graph":
            # stage_4_graphs[graph_name]["results"] =results["snn_algo_result"]
            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"][graph_name].graph["results"],
            )
        elif graph_name == "adapted_snn_graph":
            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"][graph_name].graph["results"],
            )
        elif graph_name == "rad_snn_algo_graph":
            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"][graph_name].graph["results"],
            )
        elif graph_name == "rad_adapted_snn_graph":

            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"][graph_name].graph["results"],
            )

    # overwrite nx_graphs with stage_4_graphs
    results_nx_graphs["graphs_dict"] = stage_4_graphs

    # Verify the results_nx_graphs are valid.
    nx_graphs_have_completed_stage(
        results_nx_graphs["run_config"], results_nx_graphs, 4
    )

    # Export graphs with embedded results to json.
    for graph_name, nx_graph in stage_4_graphs.items():
        verify_nx_graph_contains_correct_stages(
            graph_name,
            nx_graph,
            get_expected_stages(export_images, stage_index, to_run),
        )

    output_stage_files_3_and_4(results_nx_graphs, 4, to_run)


@typechecked
def add_result_to_last_graph(
    snn_graphs: nx.DiGraph, result_per_type: Dict
) -> None:
    """Checks whether the incoming snn_graph is a list of graphs or single
    graph.

    If it is a graph, add the results as key of the graph. If it is a
    list of graphs, add the results as a key of the last graph in the
    list.
    """
    if isinstance(snn_graphs, nx.DiGraph):
        snn_graphs.graph["results"] = result_per_type
    else:
        raise Exception(
            "Error, unsupported snn graph type:" + f"{type(snn_graphs)}"
        )
