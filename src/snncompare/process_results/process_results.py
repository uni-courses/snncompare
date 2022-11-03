"""Determines which algorithm(s) are ran, then determines for which graphs the
results are to be computed, and then computes the results per graph, against
the expected results. A different algorithm may return a different dictionary.

The MDSA algorithm results will consist of a list of nodes per used
graph that have been selected according to Alipour, and according to the
respective SNN graph.
"""

import copy

import networkx as nx

from src.snncompare.export_results.Output_stage_34 import (
    output_stage_files_3_and_4,
)
from src.snncompare.export_results.verify_nx_graphs import (
    verify_nx_graph_contains_correct_stages,
)
from src.snncompare.helper import (
    add_stage_completion_to_graph,
    get_expected_stages,
)
from src.snncompare.import_results.check_completed_stages import (
    nx_graphs_have_completed_stage,
)
from src.snncompare.process_results.get_mdsa_results import (
    set_mdsa_snn_results,
)


def set_results(run_config: dict, stage_2_graphs: dict) -> None:
    """Gets the results for the algorithms that have been ran."""
    for algo_name, algo_settings in run_config["algorithm"].items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                set_mdsa_snn_results(
                    algo_settings["m_val"], run_config, stage_2_graphs
                )
                # Set completed stage.
                # Indicate the graphs have completed stage 1.
                for nx_graph in stage_2_graphs.values():
                    add_stage_completion_to_graph(nx_graph, 4)
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


def export_results_to_json(
    export_images: bool,
    results_nx_graphs: dict,
    stage_index: int,
    to_run: dict,
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


def add_result_to_last_graph(snn_graphs: dict, result_per_type: dict) -> None:
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
