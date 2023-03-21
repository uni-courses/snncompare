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
from simsnn.core.simulators import Simulator
from snnalgorithms.sparse.MDSA.apply_results_to_graphs import (
    set_mdsa_snn_results,
)
from snnbackends.verify_nx_graphs import verify_snn_contains_correct_stages
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config

from ..helper import add_stage_completion_to_graph, get_expected_stages
from ..import_results.check_completed_stages import (
    nx_graphs_have_completed_stage,
)


@typechecked
def set_results(
    *,
    exp_config: Exp_config,
    output_config: Output_config,
    run_config: Run_config,
    stage_2_graphs: Dict,
) -> bool:
    """Gets the results for the algorithms that have been ran."""
    for algo_name, algo_settings in run_config.algorithm.items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                return perform_mdsa_results_computation_if_needed(
                    exp_config=exp_config,
                    m_val=algo_settings["m_val"],
                    output_config=output_config,
                    run_config=run_config,
                    stage_2_graphs=stage_2_graphs,
                )
            # pylint: disable=R0801
            raise TypeError(
                "Error, m_val setting is not of type int:"
                f'{type(algo_settings["m_val"])}'
                f'm_val={algo_settings["m_val"]}'
            )
        raise NotImplementedError(
            f"Error, algo_name:{algo_name} is not (yet) supported."
        )
    return False


@typechecked
def perform_mdsa_results_computation_if_needed(
    *,
    exp_config: Exp_config,
    m_val: int,
    output_config: Output_config,
    run_config: Run_config,
    stage_2_graphs: Dict,
) -> bool:
    """Performs result computation if the results are not in the graph yet."""
    set_new_results: bool = False

    for snn in stage_2_graphs.values():
        if isinstance(snn, Simulator):
            graph = snn.network.graph
        else:
            graph = snn

        if (
            4 not in graph.graph["completed_stages"]
            or 4 in output_config.recreate_stages
        ):
            set_new_results = True
            set_mdsa_snn_results(
                exp_config=exp_config,
                m_val=m_val,
                output_config=output_config,
                run_config=run_config,
                stage_2_graphs=stage_2_graphs,
            )

            # Indicate the graphs have completed stage 1.
            add_stage_completion_to_graph(snn=graph, stage_index=4)
    return set_new_results


@typechecked
def compute_results(
    *,
    results_nx_graphs: Dict,
    stage_index: int,
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
                snn_graphs=stage_4_graphs[graph_name],
                result_per_type=results_nx_graphs["graphs_dict"][
                    graph_name
                ].graph["results"],
            )
        elif graph_name == "adapted_snn_graph":
            add_result_to_last_graph(
                snn_graphs=stage_4_graphs[graph_name],
                result_per_type=results_nx_graphs["graphs_dict"][
                    graph_name
                ].graph["results"],
            )
        elif graph_name == "rad_snn_algo_graph":
            add_result_to_last_graph(
                snn_graphs=stage_4_graphs[graph_name],
                result_per_type=results_nx_graphs["graphs_dict"][
                    graph_name
                ].graph["results"],
            )
        elif graph_name == "rad_adapted_snn_graph":
            add_result_to_last_graph(
                snn_graphs=stage_4_graphs[graph_name],
                result_per_type=results_nx_graphs["graphs_dict"][
                    graph_name
                ].graph["results"],
            )

    # overwrite nx_graphs with stage_4_graphs
    results_nx_graphs["graphs_dict"] = stage_4_graphs

    # Verify the results_nx_graphs are valid.
    nx_graphs_have_completed_stage(
        run_config=results_nx_graphs["run_config"],
        results_nx_graphs=results_nx_graphs,
        stage_index=4,
    )

    # Export graphs with embedded results to json.
    # TODO: move export into separate function.
    for graph_name, nx_graph in stage_4_graphs.items():
        verify_snn_contains_correct_stages(
            graph_name=graph_name,
            snn=nx_graph,
            expected_stages=get_expected_stages(
                stage_index=stage_index,
            ),
        )


@typechecked
def add_result_to_last_graph(
    *, snn_graphs: nx.DiGraph, result_per_type: Dict
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
        raise TypeError(
            "Error, unsupported snn graph type:" + f"{type(snn_graphs)}"
        )
