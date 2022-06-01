# -*- coding: utf-8 -*-
"""File used to generate graphs that are used for testing."""

import random
from itertools import combinations, groupby

import networkx as nx
import numpy as np

from src.LIF_neuron import LIF_neuron
from src.plot_graphs import plot_circular_graph


def get_standard_graph_4_nodes() -> nx.DiGraph:
    """Returns a Y-shaped graph with four nodes."""
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
    nodename 0 to nodename 1."""
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


def gnp_random_connected_graph(
    density,
    size,
    test_scope,
):
    """Generates a random undirected graph, similarly to an Erdős-Rényi graph,
    but enforcing that the resulting graph is conneted.

    :param density:
    :param size:
    :param test_scope:
    """
    random.seed(test_scope.seed)
    edges = combinations(range(size), 2)
    G = nx.DiGraph()
    G.add_nodes_from(range(size))
    if density <= 0:
        return G
    if density >= 1:
        return nx.complete_graph(size, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)

        random_edge = random.choice(node_edges)  # nosec - using a random seed.
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < density:  # nosec - no security application.
                G.add_edge(*e)

    set_random_edge_weights(
        G,
        test_scope.min_edge_weight,
        test_scope.max_edge_weight,
        test_scope.seed,
    )

    set_rand_neuron_properties(G, test_scope)
    plot_circular_graph(density, G, test_scope.seed, export=True, show=True)
    return G


def set_random_edge_weights(G, min_weight, max_weight, seed):
    """Creates random edge weights and assigns them to the edge objects in the
    graph.

    :param G:
    :param min_weight:
    :param max_weight:
    :param seed:
    """

    # Create filler for edge attributes.
    nx.set_edge_attributes(G, None, "weight")

    rand_edge_weights = get_list_with_rand_ints_in_range(
        min_weight, max_weight, G.number_of_edges(), seed
    )

    # Overwrite each individual edge weight with random edge weight.
    for i, edge in enumerate(G.edges):
        G.edges[edge]["weight"] = float(rand_edge_weights[i])


def set_rand_neuron_properties(
    G,
    test_scope,
):
    """Sets name: int, bias: float, du: float, dv: float, vth: float for each
    neuron with random value within predetermined ranges.

    :param G:
    :param test_scope:
    """
    biases = get_list_with_rand_floats_in_range(
        test_scope.min_bias, test_scope.max_bias, len(G), test_scope.seed
    )
    dus = get_list_with_rand_floats_in_range(0, 1, len(G), test_scope.seed)
    dvs = get_list_with_rand_floats_in_range(0, 1, len(G), test_scope.seed)
    v_thresholds = get_list_with_rand_floats_in_range(
        test_scope.min_vth, test_scope.max_vth, len(G), test_scope.seed
    )

    # Create a LIF neuron object.
    for node in G.nodes:
        G.nodes[node]["nx_LIF"] = LIF_neuron(
            name=node,
            bias=biases[node],
            du=dus[node],
            dv=dvs[node],
            vth=v_thresholds[node],
        )


def get_list_with_rand_ints_in_range(min_val, max_val, length, seed):
    """Generates and returns a list with random integers in range [min,max] of
    length length.

    :param min_val:
    :param max_val:
    :param length:
    :param seed:
    """
    # Specify random seed.
    random.seed(seed)

    # Get list with random edge weights.
    # The randomness needs to be deterministic for testing purposes, so
    # it is ok if it is not a real random number, this is not a security
    # application, hence the # nosec.
    rand_integers = random.choices(range(min_val, max_val), k=length)  # nosec
    return rand_integers


def get_list_with_rand_floats_in_range(min_val, max_val, length, seed):
    """Generates and returns a list with random integers in range [min,max] of
    length length.

    :param min_val:
    :param max_val:
    :param length:
    :param seed:
    """
    # Specify random seed.
    random.seed(seed)

    # Get list with random edge weights.
    rand_floats = np.random.uniform(low=min_val, high=max_val, size=length)
    print(f"rand_floats={rand_floats}")
    return rand_floats
