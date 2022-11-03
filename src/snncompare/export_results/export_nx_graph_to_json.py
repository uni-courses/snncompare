"""Converts the nx graphs into json objects."""

import copy
from typing import Any, Dict, List, Union

import networkx as nx
from networkx.readwrite import json_graph


def digraph_to_json(G: nx.DiGraph) -> Any:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    TODO: remove if not used.

    """
    if G is not None:
        return json_graph.node_link_data(G)
        # return json_graph.dumps(G)
    raise Exception("Error, incoming graph was None.")


def convert_digraphs_to_json(
    results_nx_graphs: dict, stage_index: int
) -> dict:
    """Converts the digraph networkx objects to json dicts."""
    print(f"stage_index={stage_index}")
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


def convert_stage_1_digraphs_to_json(
    graphs: Union[nx.Graph, nx.DiGraph]
) -> Dict[str, Any]:
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


def convert_stage_2_digraphs_to_json(
    graphs: Dict[str, Union[Union[nx.Graph, nx.DiGraph], List]]
) -> Dict[str, Any]:
    """Puts all the graphs of stage 2 into a single graph."""
    graphs_dict_stage_2 = {}
    for graph_name, graph_container in graphs.items():
        graphs_per_type = []
        if isinstance(graph_container, (nx.DiGraph, nx.Graph)):
            graphs_per_type.append(digraph_to_json(graph_container))
        elif isinstance(graph_container, List):
            for graph in graph_container:
                graphs_per_type.append(digraph_to_json(graph))
        else:
            raise Exception(f"Error, unsupported type:{type(graph_container)}")
        graphs_dict_stage_2[graph_name] = graphs_per_type
    if not graphs_dict_stage_2:  # checks if dict not empty like: {}
        raise Exception(
            f"Error, len(graphs)={len(graphs)} stage=2, graphs_dict_stage_2"
            + " is empty."
        )
    return graphs_dict_stage_2
