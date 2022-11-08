"""File used to generate graphs that are used for testing."""
from __future__ import annotations

import math
import random
from itertools import combinations, groupby
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from networkx.classes.digraph import DiGraph
from numpy import ndarray
from snnbackends.networkx.LIF_neuron import LIF_neuron
from snnbackends.plot_graphs import plot_circular_graph
from typeguard import typechecked

if TYPE_CHECKING:
    from tests.exp_setts.unsorted.test_scope import Long_scope_of_tests


@typechecked
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
    graph.nodes[0]["nx_LIF"] = [
        LIF_neuron(name=str(0), bias=3.0, du=0.0, dv=0.0, vth=2.0)
    ]

    # Create networkx neuron that simulates LIF neuron from lava.
    graph.nodes[1]["nx_LIF"] = [
        LIF_neuron(name=str(1), bias=0.0, du=0.0, dv=0.0, vth=10.0)
    ]
    return graph


@typechecked
def gnp_random_connected_graph(
    density: float,
    recurrent_density: int | float,
    size: int,
    test_scope: Long_scope_of_tests,
) -> DiGraph:
    """Generates a random undirected graph, similarly to an Erdős-Rényi graph,
    but enforcing that the resulting graph is conneted.

    :param density: param size:
    :param test_scope:
    :param recurrent_density:
    :param size: Nr of nodes in the original graph on which test is ran.
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
        listed_node_edges = list(node_edges)

        random_edge = random.choice(  # nosec - using a random seed.
            listed_node_edges
        )
        G.add_edge(*random_edge)
        for e in listed_node_edges:
            if random.random() < density:  # nosec - no security application.
                G.add_edge(*e)

    set_random_edge_weights(
        G,
        test_scope.min_edge_weight,
        test_scope.max_edge_weight,
        test_scope.seed,
    )

    add_random_recurrent_edges(G, recurrent_density, test_scope)

    set_rand_neuron_properties(G, test_scope)
    plot_circular_graph(
        density,
        G,
        recurrent_density,
        test_scope,
    )
    return G


@typechecked
def add_random_recurrent_edges(
    G: nx.DiGraph, recurrent_edge_density: float, test_scope: Any
) -> None:
    """Adds random recurrent edges.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param recurrent_edge_density:
    :param test_scope:
    """

    # Use the recurrent_edge_density to get amount of True values.
    # Use seed.
    # Get list of random booleans to decide which recurrent edges are created.
    rand_bools = get_list_with_rand_bools(
        len(G), recurrent_edge_density, test_scope.seed
    )

    # Get list of random edge values (un-used weights are ignored/skipped).
    rand_edge_weights = get_list_with_rand_ints_in_range(
        test_scope.min_edge_weight,
        test_scope.max_edge_weight,
        G.number_of_edges(),
        test_scope.seed,
    )

    for node in G.nodes:

        if rand_bools[node]:

            # Add the recurrent edge.
            G.add_edge(node, node)

            # TODO: verify whether attribute weight exists for the newly
            # created edge.
            # Create random weight in recurrent edge.
            G.edges[(node, node)]["weight"] = float(rand_edge_weights[node])


@typechecked
def set_random_edge_weights(
    G: DiGraph, min_weight: int, max_weight: int, seed: int
) -> None:
    """Creates random edge weights and assigns them to the edge objects in the
    graph.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param min_weight:
    :param max_weight:
    :param seed:
    :param min_weight:
    :param seed: The value of the random seed used for this test.
    """

    # Create filler for edge attributes.
    nx.set_edge_attributes(G, None, "weight")

    rand_edge_weights = get_list_with_rand_ints_in_range(
        min_weight, max_weight, G.number_of_edges(), seed
    )

    # Overwrite each individual edge weight with random edge weight.
    for i, edge in enumerate(G.edges):
        G.edges[edge]["weight"] = float(rand_edge_weights[i])


@typechecked
def set_rand_neuron_properties(
    G: DiGraph,
    test_scope: Long_scope_of_tests,
) -> None:
    """Sets name: int, bias: float, du: float, dv: float, vth: float for each
    neuron with random value within predetermined ranges.

    :param G: The original graph on which the MDSA algorithm is ran.
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
        G.nodes[node]["nx_LIF"] = [
            LIF_neuron(
                name=node,
                bias=biases[node],
                du=dus[node],
                dv=dvs[node],
                vth=v_thresholds[node],
            )
        ]


@typechecked
def get_list_with_rand_ints_in_range(
    min_val: int, max_val: int, length: int, seed: int
) -> Any:
    """Generates and returns a list with random integers in range [min,max] of
    length length.

    :param min_val: param max_val:
    :param length: param seed:
    :param max_val:
    :param seed: The value of the random seed used for this test.
    """
    # Specify random seed.
    random.seed(seed)

    # Get list with random edge weights.
    # The randomness needs to be deterministic for testing purposes, so
    # it is ok if it is not a real random number, this is not a security
    # application, hence the # nosec.
    rand_integers = random.choices(range(min_val, max_val), k=length)  # nosec
    return rand_integers


@typechecked
def get_list_with_rand_floats_in_range(
    min_val: int, max_val: int, length: int, seed: int
) -> ndarray:
    """Generates and returns a list with random integers in range [min,max] of
    length length.

    :param min_val: param max_val:
    :param length: param seed:
    :param max_val:
    :param seed: The value of the random seed used for this test.
    """
    # Specify random seed.
    np.random.seed(seed)

    # Get list with random edge weights.
    rand_floats = np.random.uniform(low=min_val, high=max_val, size=length)
    return rand_floats


@typechecked
def get_list_with_rand_bools(
    length: int, recurrent_edge_density: int | float, seed: int
) -> list[bool]:
    """Generates and returns a list with random booleans of length length. The
    amount of True values is determined by: recurrent_edge_density*length.

    :param min_val: param max_val:
    :param length: param seed:
    :param recurrent_edge_density:
    :param seed: The value of the random seed used for this test.
    """
    # Specify random seed.
    random.seed(seed)

    # Compute how many True and False values are expected.
    amount_of_true_vals = math.ceil(recurrent_edge_density * length)
    amount_of_false_vals = length - amount_of_true_vals

    # Generate the ordered true false list.
    rand_bools = [False] * amount_of_false_vals + [True] * amount_of_true_vals
    if len(rand_bools) != length:
        raise Exception(
            f"The list of random booleans has length:{len(rand_bools)}"
            + f" whereas it should have length:{length}."
        )

    # Implement random order of True and False values.
    # The randomness needs to be deterministic for testing purposes, so
    # it is ok if it is not a real random number, this is not a security
    # application, hence the # nosec.

    random.shuffle(rand_bools)  # nosec # Note this shuffles in place.

    return rand_bools
