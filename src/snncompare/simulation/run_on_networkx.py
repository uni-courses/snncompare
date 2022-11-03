"""Runs a converted networkx graph without the Lava platform.

First verifies the graph represents a connected and valid SNN, with all
required neuron and synapse properties specified. Then loops through the
network to simulate it, one neuron at a time.
"""

# Import external libraries.

from typing import List

import networkx as nx

from src.snncompare.simulation.LIF_neuron import LIF_neuron
from src.snncompare.simulation.verify_graph_is_snn import (
    verify_networkx_snn_spec,
)

# Import local project functions and classes.


def add_nx_neurons_to_networkx_graph(G: nx.DiGraph, t: int) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param duration: int:

    """
    # Verify the graph represents a connected and valid SNN, with all required
    # neuron and synapse properties specified.
    verify_networkx_snn_spec(G, t, backend="nx")

    # Create LIF neurons in networkx graph.
    # generate_lif_neurons(G)

    # Create synapses between neurons in networkx graph edges.
    generate_lif_synapses(G)

    # Initialise the nodes at time t=0 (with a_in=0).
    initialise_a_in_is_zero_at_t_is_1(G, t)


def generate_lif_synapses(G: nx.DiGraph) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.

    """
    return G


def run_snn_on_networkx(G: nx.DiGraph, sim_duration: int) -> None:
    """Runs the simulation for t timesteps using networkx, not lava.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param t: int:
    """
    for t in range(sim_duration):
        # Verify the neurons of the previous timestep are valid.
        verify_networkx_snn_spec(G, t, backend="nx")

        # Copy the neurons into the new timestep.
        copy_old_neurons_into_new_neuron_element(G, t)

        verify_networkx_snn_spec(G, t + 1, backend="nx")
        run_simulation_with_networkx_for_1_timestep(G, t + 1)

    # Verify the network dimensions. (Ensure sufficient nodes are added.)
    verify_networkx_graph_dimensions(G, sim_duration)


def copy_old_neurons_into_new_neuron_element(G: nx.DiGraph, t: int) -> None:
    """Creates a new neuron for the next timestep, by copying the old neuron.

    TODO: determine what to do with the synapses.
    """
    for node in G.nodes:
        G.nodes[node]["nx_LIF"].append(G.nodes[node]["nx_LIF"][t])


def verify_networkx_graph_dimensions(G: nx.DiGraph, sim_duration: int) -> None:
    """Ensures the graph contains at least sim_duration SNN neurons of a single
    name. This is because each neuron, with a single name, needs to be
    simulated for sim_duration timesteps. This simulation is done by storing
    copies of the SNN neuron in a list, one for each simulated timestep.

    The graph is expected to adhere to the following structure:
    G.nodes[nodename] stores a single node, representing a neuron over
    time. Then that node should contain a list of nx_LIF neurons over
    time in: G.nodes[node]["nx_LIF"] which is of type: List. Then each
    element in that list must be of type: nx_LIF() neuron.
    """
    for nodename in G.nodes:
        # Assert node has nx_LIF neuron object of type list.
        if not isinstance(G.nodes[nodename]["nx_LIF"], List):
            raise Exception(
                "Error, {nodename} nx_LIF is not of type list."
                + f'instead, it is of type:{type(G.nodes[nodename]["nx_LIF"])}'
            )

        # TODO: remove the artifact last neuron, (remove the +1), it is not
        # needed.
        if not len(G.nodes[nodename]["nx_LIF"]) == sim_duration + 1:
            raise Exception(
                f"Error, neuron:{nodename} did not have len:"
                + f"{sim_duration+1}. Instead, it had len:"
                + f'{len(G.nodes[nodename]["nx_LIF"])}'
            )

        for t, neuron_at_time_t in enumerate(G.nodes[nodename]["nx_LIF"]):
            if not isinstance(neuron_at_time_t, LIF_neuron):
                raise Exception(
                    f"Error, {nodename} does not have a neuron of"
                    + f"type:{LIF_neuron} at t={t}. Instead, it is of type:"
                    + f"{type(neuron_at_time_t)}"
                )


def run_simulation_with_networkx_for_1_timestep(G: nx.DiGraph, t: int) -> None:
    """Runs the networkx simulation of the network for 1 timestep. The results
    of the simulation are stored in the G.nodes network.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    # Visited edges
    visited_edges = []

    # First reset all a_in_next values for a new round of simulation.
    reset_a_in_next_for_all_neurons(G, t)

    # Compute for each node whether it spikes based on a_in, starting at t=1.
    for node in G.nodes:
        nx_lif = G.nodes[node]["nx_LIF"][t]
        spikes = nx_lif.simulate_neuron_one_timestep(nx_lif.a_in)

        if spikes:

            # Propagate the output spike to the connected receiving neurons.
            for neighbour in nx.all_neighbors(G, node):

                if (node, neighbour) not in visited_edges:
                    visited_edges.append((node, neighbour))

                    # Check if the outgoing edge is exists and is directed.
                    if G.has_edge(node, neighbour):

                        # Compute synaptic weight.
                        weight = G.edges[(node, neighbour)]["weight"]

                        # Add input signal to connected receiving neuron.
                        G.nodes[neighbour]["nx_LIF"][t].a_in_next += 1 * weight

    # After all inputs have been computed, store a_in_next values for next
    # round into a_in of the current round to prepare for the nextsimulation
    # step.
    for node in G.nodes:
        nx_lif = G.nodes[node]["nx_LIF"][t]
        nx_lif.a_in = nx_lif.a_in_next


def reset_a_in_next_for_all_neurons(G: nx.DiGraph, t: int) -> None:
    """Resets the a_in_next for all neurons to 0.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    for node in G.nodes:
        G.nodes[node]["nx_LIF"][t].a_in_next = 0


def initialise_a_in_is_zero_at_t_is_1(G: nx.DiGraph, t: int) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.

    """
    for node in G.nodes:
        G.nodes[node]["nx_LIF"][t].a_in = 0
        # G.nodes[node]["a_in"] = 0
