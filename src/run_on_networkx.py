# -*- coding: utf-8 -*-
# Runs a converted networkx graph without the Lava platform. First verifies the
# graph represents a connected and valid SNN, with all required neuron and
# synapse properties specified. Then loops through the network to simulate it,
# one neuron at a time.

# Import external libraries.
import networkx as nx

from .LIF_neuron import LIF_neuron

# Import local project functions and classes.
from .verify_graph_is_snn import verify_networkx_snn_spec


def simulate_snn_on_networkx(G: nx.DiGraph, duration: int) -> None:
    """

    :param G: nx.DiGraph:
    :param duration: int:

    """
    # Verify the graph represents a connected and valid SNN, with all required
    # neuron and synapse properties specified.
    verify_networkx_snn_spec(G)

    # Create LIF neurons in networkx graph.
    generate_lif_neurons(G)

    # Create synapses between neurons in networkx graph edges.
    generate_lif_synapses(G)

    # Initialise the nodes at time t=0 (with a_in=0).
    initialise_a_in_is_zero_at_t_is_1(G)

    # The simulation is ran for t timesteps on a Loihi emulation.
    for _ in range(duration):
        run_simulation_with_networkx(G)


def generate_lif_neurons(G: nx.DiGraph) -> None:
    """

    :param G: nx.DiGraph:

    """
    for node in G.nodes:
        G.nodes[node]["nx_LIF"] = LIF_neuron(
            G.nodes[node]["bias"],
            G.nodes[node]["du"],
            G.nodes[node]["dv"],
            G.nodes[node]["vth"],
        )


def generate_lif_synapses(G: nx.DiGraph) -> None:
    """

    :param G: nx.DiGraph:

    """
    return G


def run_simulation_with_networkx(G: nx.DiGraph) -> None:
    """

    :param G: nx.DiGraph:

    """

    # Compute for each node whether it spikes based on a_in, starting at t=1.
    for node in G.nodes:
        nx_lif = G.nodes[node]["nx_LIF"]
        spikes = nx_lif.simulate_neuron_one_timestep(nx_lif.a_in)
        # print(f'node={node}, u={nx_lif.u}, v={nx_lif.v} spikes={spikes}')
        if spikes:
            # Propagate the output spike to the connected receiving neurons.
            for neighbour in nx.all_neighbors(G, node):

                # Check if the outgoing edge is exists and is directed.
                if G.has_edge(node, neighbour):

                    # Compute synaptic weight.
                    weight = G.edges[(node, neighbour)]["weight"]

                    # Add input signal to connected receiving neuron.
                    G.nodes[neighbour]["nx_LIF"].a_in_next += 1 * weight

    # After all inputs have been computed, store a_in_next values for next
    # round into a_in of the current round to prepare for the nextsimulation
    # step.
    for node in G.nodes:
        nx_lif = G.nodes[node]["nx_LIF"]
        nx_lif.a_in = nx_lif.a_in_next


def initialise_a_in_is_zero_at_t_is_1(G: nx.DiGraph) -> None:
    """

    :param G: nx.DiGraph:

    """
    for node in G.nodes:
        G.nodes[node]["nx_LIF"].a_in = 0
        # G.nodes[node]["a_in"] = 0
