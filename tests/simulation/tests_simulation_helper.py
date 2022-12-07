"""Contains function to assist the simulation tests."""
from __future__ import annotations

from typing import TYPE_CHECKING

from networkx.classes.digraph import DiGraph
from snnalgorithms.get_graph import set_rand_neuron_properties
from snnbackends.lava.run_on_lava import add_lava_neurons_to_networkx_graph
from typeguard import typechecked

from tests.tests_helper import get_cyclic_graph_without_directed_path

if TYPE_CHECKING:
    from tests.exp_setts.unsorted.test_scope import Long_scope_of_tests


@typechecked
def get_graph_for_cyclic_propagation(
    test_scope: Long_scope_of_tests,
) -> DiGraph:
    """Returns the graph for the test:

    test_random_networks_are_propagated_the_same_on_networkx_and_lava
    """
    G = get_cyclic_graph_without_directed_path()
    set_rand_neuron_properties(G, test_scope)

    # TODO: Generate networkx network.

    # Generate lava network.
    add_lava_neurons_to_networkx_graph(G, t=0)
    return G
