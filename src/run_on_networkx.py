# Runs a converted networkx graph without the Lava platform. First verifies the
# graph represents a connected and valid SNN, with all required neuron and
# synapse properties specified. Then loops through the network to simulate it,
# one neuron at a time.

# Import external libraries.
import networkx as nx

# Import local project functions and classes.
from .verify_graph_is_snn import verify_networkx_snn_spec


def simulate_network_on_networkx(G: nx.Graph(), t: int) -> None:
    # Verify the graph represents a connected and valid SNN, with all required
    # neuron and synapse properties specified.
    verify_networkx_snn_spec(G)

    # The simulation is ran for t timesteps on a Loihi emulation.
    run_simulation_with_networkx(G, t)


def run_simulation_with_networkx(G: nx.Graph(), t: int) -> None:
    pass
