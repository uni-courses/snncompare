# Runs a converted networkx graph on the Lava platform. First verifies the
# graph represents a connected and valid SNN, with all required neuron and
# synapse properties specified. Then it converts the incoming networkx object
# to an SNN network that can be ran by Lava, and retrieves a first/single
# neuron. The simulation is than ran for t timesteps on a Loihi emulation.

# Import the networkx module.
import networkx as nx


def verify_networkx_graph_is_valid_snn_specification(G: nx.Graph) -> bool:
    for node in G.nodes:
        node


def convert_networkx_graph_to_lava_snn():
    pass


def simulate_snn_on_lava_loihi_emulation():
    pass
