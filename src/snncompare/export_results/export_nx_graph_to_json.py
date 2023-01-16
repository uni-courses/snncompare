"""Converts the nx graphs into json objects."""
import copy
from typing import Dict, Union

import networkx as nx
from networkx.readwrite import json_graph
from typeguard import typechecked


@typechecked
def digraph_to_json(G: nx.Graph) -> dict:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    TODO: remove if not used.

    """
    if G is not None:
        some_json_graph: Dict = json_graph.node_link_data(G)
        if not isinstance(some_json_graph, dict):
            raise TypeError(
                "Error, json graph type is not dict, it is:"
                + f"{type(some_json_graph)}"
            )
        return some_json_graph
        # return json_graph.dumps(G)
    raise Exception("Error, incoming graph was None.")


@typechecked
def convert_digraphs_to_json(
    results_nx_graphs: Dict, stage_index: int
) -> dict:
    """Converts the digraph networkx objects to json dicts."""
    results_json_graphs = copy.deepcopy(results_nx_graphs)
    # Convert incoming graphs to dictionary.
    if stage_index == 1:
        results_json_graphs["graphs_dict"] = convert_stage_1_digraphs_to_json(
            results_nx_graphs["graphs_dict"]
        )
    elif stage_index == 2:
        results_json_graphs["graphs_dict"] = convert_stage_2_digraphs_to_json(
            results_nx_graphs["graphs_dict"]
        )
    if stage_index == 3:
        pass
    if stage_index == 4:
        results_json_graphs["graphs_dict"] = convert_stage_2_digraphs_to_json(
            results_nx_graphs["graphs_dict"]
        )
    return results_json_graphs


@typechecked
def convert_stage_1_digraphs_to_json(graphs: Dict) -> Dict:
    """Puts all the graphs of stage 1 into a single graph."""
    graphs_dict_stage_1 = {}
    for graph_name, graph_container in graphs.items():
        if not isinstance(graph_container, (nx.DiGraph, nx.Graph)):
            raise Exception(
                "stage_index=1, Error, for graph:"
                + f"{graph_name}, the graph is not a"
                + f"nx.DiGraph(). Instead, it is:{type(graph_container)}"
            )
        graphs_dict_stage_1[graph_name] = digraph_to_json(graph_container)
    if not graphs_dict_stage_1:  # checks if dict not empty like: {}
        raise Exception(
            f"Error, len(graphs)={len(graphs)} stage=1, graphs_dict_stage_1"
            + " is empty."
        )
    return graphs_dict_stage_1


@typechecked
def convert_stage_2_digraphs_to_json(
    graphs: Dict[str, Union[nx.Graph, nx.DiGraph]]
) -> Dict[str, dict]:
    """Puts all the graphs of stage 2 into a single graph dict."""
    graphs_dict_stage_2 = {}
    for graph_name, graph_container in graphs.items():
        if isinstance(graph_container, (nx.DiGraph, nx.Graph)):
            graphs_dict_stage_2[graph_name] = digraph_to_json(graph_container)
        else:
            raise Exception(f"Error, unsupported type:{type(graph_container)}")
    if not graphs_dict_stage_2:  # checks if dict not empty like: {}
        raise Exception(
            f"Error, len(graphs)={len(graphs)} stage=2, graphs_dict_stage_2"
            + " is empty."
        )
    return graphs_dict_stage_2
