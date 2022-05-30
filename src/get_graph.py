# -*- coding: utf-8 -*-
"""File used to generate graphs that are used for testing."""

import networkx as nx

from src.LIF_neuron import LIF_neuron


def get_standard_graph_4_nodes() -> nx.DiGraph:
    """Returns a Y-shaped graph with four nodes.

    :param ) -> nx.DiGraph(:
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [0, 1, 2, 3],
        color="w",
    )
    graph.add_edges_from(
        [
            (0, 2),
            (1, 2),
            (2, 3),
        ],
        weight=10,
    )
    return graph


def get_networkx_graph_of_2_neurons() -> nx.DiGraph:
    """Returns graph with 2 neurons with a synapse with weight of 4 from
    nodename 0 to nodename 1.

    :param ) -> nx.DiGraph(:
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [0, 1],
        color="w",
    )
    graph.add_edges_from(
        [
            (0, 1),
        ],
        weight=6,
    )

    # Create networkx neuron that simulates LIF neuron from lava.
    graph.nodes[0]["nx_LIF"] = LIF_neuron(
        name=0, bias=3.0, du=0.0, dv=0.0, vth=2.0
    )

    # Create networkx neuron that simulates LIF neuron from lava.
    graph.nodes[1]["nx_LIF"] = LIF_neuron(
        name=1, bias=0.0, du=0.0, dv=0.0, vth=10.0
    )
    return graph
