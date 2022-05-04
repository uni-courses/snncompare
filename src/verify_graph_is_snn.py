# Verifies the graph represents a connected and valid SNN, with all required
# neuron and synapse properties specified.

# Import the networkx module.
import networkx as nx


def verify_networkx_graph_is_valid_snn_specification(G: nx.Graph) -> bool:
    for node in G.nodes:
        verify_neuron_properties_are_specified(node)


def verify_neuron_properties_are_specified(node: nx.Graph.nodes):
    assert isinstance(node["bias"], int), "Bias is not an integer."
    assert isinstance(node["du"], int), "du is not an integer."
    assert isinstance(node["dv"], int), "dv is not an integer."
    assert isinstance(node["vth"], int), "vth is not an integer."


def verify_synapse_properties_are_specified(edge):
    assert isinstance(edge["w"], int), f"Weight of edge {edge} is not an"
    +" integer."
