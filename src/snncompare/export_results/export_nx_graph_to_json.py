"""Converts the nx graphs into json objects."""
from typing import Dict

import networkx as nx
from networkx.readwrite import json_graph
from typeguard import typechecked


@typechecked
def digraph_to_json(*, G: nx.Graph) -> Dict:
    """:param G: The original graph on which the MDSA algorithm is ran.

    TODO: remove if not used.
    """
    if G is not None:
        some_json_graph: Dict = json_graph.node_link_data(G)
        if not isinstance(some_json_graph, Dict):
            raise TypeError(
                "Error, json graph type is not Dict, it is:"
                + f"{type(some_json_graph)}"
            )
        return some_json_graph
        # return json_graph.dumps(G)
    raise ValueError("Error, incoming graph was None.")
