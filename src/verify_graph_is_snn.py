# Verifies the graph represents a connected and valid SNN, with all required
# neuron and synapse properties specified.

# Import the networkx module.
import networkx as nx


def verify_networkx_snn_spec(G: nx.Graph) -> None:
    for node in G.nodes:
        print(f"node:{node}")
        print(f"node:{G.nodes[node]}")
        verify_neuron_properties_are_specified(G.nodes[node])


def verify_neuron_properties_are_specified(node: nx.Graph.nodes) -> None:
    assert isinstance(node["bias"], float), "Bias is not a float."
    assert isinstance(node["du"], float), "du is not a float."
    assert isinstance(node["dv"], float), "dv is not a float."
    assert isinstance(node["vth"], float), "vth is not a float."


def verify_synapse_properties_are_specified(edge: nx.Graph.edges) -> None:
    assert isinstance(edge["w"], float), f"Weight of edge {edge} is not a"
    +" float."


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
