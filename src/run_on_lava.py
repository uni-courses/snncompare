# -*- coding: utf-8 -*-
# Runs a converted networkx graph on the Lava platform. First verifies the
# graph represents a connected and valid SNN, with all required neuron and
# synapse properties specified. Then it converts the incoming networkx object
# to an SNN network that can be ran by Lava, and retrieves a first/single
# neuron. The simulation is than ran for t timesteps on a Loihi emulation.

import networkx as nx

# Instantiate Lava processes to build network.
from lava.proc.lif.process import LIF

# Import local project functions and classes.
from verify_graph_is_snn import verify_networkx_snn_spec


def simulate_network_on_lava(G: nx.Graph, t: int) -> None:
    """

    :param G: nx.Graph:
    :param t: int:

    """
    # Verify the graph represents a connected and valid SNN, with all required
    # neuron and synapse properties specified.
    verify_networkx_snn_spec(G)

    # Convert networkx graph to an SNN network that can be ran by Lava.
    starter_neuron = convert_networkx_graph_to_lava_snn(G)

    # The simulation is ran for t timesteps on a Loihi emulation.
    run_simulation_on_lava(t, starter_neuron)


def convert_networkx_graph_to_lava_snn(G: nx.Graph) -> LIF:
    """

    :param G: nx.Graph:

    """
    # TODO: Change return G to return a lif neuron.
    return G


def run_simulation_on_lava(t: int, starter_neuron: LIF) -> None:
    """

    :param t: int:
    :param starter_neuron: LIF:

    """
    # TODO: remove this line and write function.
    print(t)
    # TODO: remove this line and write function.
    print(f"starter_neuron={starter_neuron}")
