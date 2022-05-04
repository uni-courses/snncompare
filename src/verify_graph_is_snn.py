# Verifies the graph represents a connected and valid SNN, with all required
# neuron and synapse properties specified.

# Import the networkx module.
import networkx as nx


def verify_networkx_snn_spec(G: nx.Graph) -> None:
    for node in G.nodes:
        verify_neuron_properties_are_specified(node)


def verify_neuron_properties_are_specified(node: nx.Graph.nodes) -> None:
    assert isinstance(node["bias"], int), "Bias is not an integer."
    assert isinstance(node["du"], int), "du is not an integer."
    assert isinstance(node["dv"], int), "dv is not an integer."
    assert isinstance(node["vth"], int), "vth is not an integer."


def verify_synapse_properties_are_specified(edge: nx.Graph.edges) -> None:
    assert isinstance(edge["w"], int), f"Weight of edge {edge} is not an"
    +" integer."


def assert_all_synapse_properties_are_specified(G, edge):
    if not all_synapse_properties_are_specified(G, edge):
        raise Exception(
            f"Not all synapse prpoerties of edge: {edge} are"
            + " specified. It only contains attributes:"
            + f"{get_synapse_property_names(G,edge)}"
        )


def all_synapse_properties_are_specified(G, edge):
    synapse_property_names = get_synapse_property_names(G, edge)
    if "weight" in synapse_property_names:
        # if 'delay' in synapse_property_names:
        # TODO: implement delay using a chain of neurons in series since this
        # is not yet supported by lava-nc.
        return True
    return False


def get_synapse_property_names(G, edge):
    return G.edges[edge].keys()
