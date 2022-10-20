"""Determines which algorithm(s) are ran, then determines for which graphs the
results are to be computed, and then computes the results per graph, against
the expected results. A different algorithm may return a different dictionary.

The MDSA algorithm results will consist of a list of nodes per used
graph that have been selected according to Alipour, and according to the
respective SNN graph.
"""

import copy

import networkx as nx

from src.export_results.Output_stage_34 import output_stage_files_3_and_4
from src.process_results.get_mdsa_results import get_mdsa_snn_results


def get_results(run_config: dict, stage_2_graphs: dict) -> dict:
    """Gets the results for the algorithms that have been ran."""
    for algo_name, algo_settings in run_config["algorithm"].items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                return get_mdsa_snn_results(
                    algo_settings["m_val"], run_config, stage_2_graphs
                )
            raise Exception("Error, m_val setting is not of type int.")
        raise Exception(
            f"Error, algo_name:{algo_name} is not (yet) supported."
        )
    raise Exception("No supported algorithm found.")


def export_results(results_nx_graphs: dict, stage_index: int):
    """Integrates the results per graph type into the graph, then export the
    results dictionary (again) into the stage 4 folder."""
    print(f"stage_index={stage_index}")
    # Create new independent graphs dict to include the results.
    # TODO: determine why/don't duplicate.
    stage_4_graphs = copy.deepcopy(results_nx_graphs["graphs_dict"])

    # Embed results into snn graphs
    for graph_name in results_nx_graphs["graphs_dict"].keys():
        if graph_name == "snn_algo_graph":
            # stage_4_graphs[graph_name]["results"] =results["snn_algo_result"]
            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"]["snn_algo_result"],
            )
        elif graph_name == "adapted_snn_graph":
            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"]["adapted_snn_algo_result"],
            )
        elif graph_name == "rad_snn_algo_graph":
            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"]["rad_snn_algo_graph"],
            )
        elif graph_name == "rad_adapted_snn_graph":

            add_result_to_last_graph(
                stage_4_graphs[graph_name],
                results_nx_graphs["graphs_dict"]["rad_adapted_snn_graph"],
            )

    # Export graphs with embedded results to json.
    output_stage_files_3_and_4(results_nx_graphs, 4)


def add_result_to_last_graph(snn_graphs, result_per_type: dict):
    """Checks whether the incoming snn_graph is a list of graphs or single
    graph.

    If it is a graph, add the results as key of the graph. If it is a
    list of graphs, add the results as a key of the last graph in the
    list.
    """
    if isinstance(snn_graphs, nx.DiGraph):
        snn_graphs.graph["result"] = result_per_type
    # elif isinstance(snn_graphs, List):
    #    raise Exception("Unexpected snn graphs type: list.")
    #    snn_graphs[-1].graph["result"] = result_per_type
    else:
        raise Exception(
            "Error, unsupported snn graph type:" + f"{type(snn_graphs)}"
        )


# def query_results(graph_type: str, passed: bool, stage_4_graphs: dict):
#    """Returns all graphs of type: graph_type that have failed/passed the
#    test."""
#    # TODO: implement
