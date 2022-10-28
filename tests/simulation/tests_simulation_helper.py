"""Contains function to assist the simulation tests."""

from src.graph_generation.get_graph import (
    get_cyclic_graph_without_directed_path,
    set_rand_neuron_properties,
)
from src.simulation.run_on_lava import add_lava_neurons_to_networkx_graph
from src.simulation.run_on_networkx import add_nx_neurons_to_networkx_graph


def get_graph_for_cyclic_propagation(test_scope):
    """Returns the graph for the test:

    test_random_networks_are_propagated_the_same_on_networkx_and_lava
    """
    G = get_cyclic_graph_without_directed_path()
    set_rand_neuron_properties(G, test_scope)

    # Generate networkx network.
    add_nx_neurons_to_networkx_graph(G, t=0)
    # Generate lava network.
    add_lava_neurons_to_networkx_graph(G, t=0)
    return G
