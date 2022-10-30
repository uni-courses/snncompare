"""Verifies the graph represents a connected and valid SNN, with all required
neuron and synapse properties specified."""

# Import the networkx module.
import networkx as nx

from src.snncompare.simulation.verify_graph_is_lava_snn import (
    verify_lava_neuron_properties_are_specified,
)
from src.snncompare.simulation.verify_graph_is_networkx_snn import (
    assert_synapse_properties_are_specified,
    verify_nx_neuron_properties_are_specified,
)


def verify_networkx_snn_spec(G: nx.DiGraph, t: int, backend: str) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param G: The original graph on which the MDSA algorithm is ran.

    """
    for nodename in G.nodes:
        if nodename != "connecting_node":
            if backend in ["networkx", "generic"]:
                verify_nx_neuron_properties_are_specified(
                    G.nodes[nodename], t=t
                )
            if backend == "lava":
                verify_lava_neuron_properties_are_specified(G.nodes[nodename])

    # TODO: verify synapse properties
    for edge in G.edges:
        assert_synapse_properties_are_specified(G, edge)
