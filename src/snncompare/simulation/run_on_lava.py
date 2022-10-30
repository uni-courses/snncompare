"""Runs a converted networkx graph on the Lava platform.

First verifies the graph represents a connected and valid SNN, with all
required neuron and synapse properties specified. Then it converts the
incoming networkx object to an SNN network that can be ran by Lava, and
retrieves a first/single neuron. The simulation is than ran for t
timesteps on a Loihi emulation.
"""


import networkx as nx
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

# Instantiate Lava processes to build network.
from lava.proc.lif.process import LIF

from src.snncompare.graph_generation.convert_networkx_to_lava import (
    initialise_networkx_to_snn_conversion,
)
from src.snncompare.simulation.verify_graph_is_snn import (
    verify_networkx_snn_spec,
)


def simulate_snn_on_lava(G: nx.Graph, starter_nodename: int, t: int) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran. nx.Graph:
    :param t: int:
    :param G: The original graph on which the MDSA algorithm is ran. nx.Graph:
    :param starter_nodename: int:
    :param t: int:

    """
    # Verify the graph represents a connected and valid SNN, with all required
    # neuron and synapse properties specified.
    verify_networkx_snn_spec(G, t, backend="lava")

    # The simulation is ran for t timesteps on a Loihi emulation.
    run_simulation_on_lava(t, G.nodes[starter_nodename]["lava_LIF"])


def run_simulation_on_lava(t: int, starter_neuron: LIF) -> None:
    """

    :param t: int:
    :param starter_neuron: LIF:
    :param t: int:
    :param starter_neuron: LIF:

    """
    # Run the simulation for t timesteps.
    starter_neuron.run(condition=RunSteps(num_steps=t), run_cfg=Loihi1SimCfg())


def add_lava_neurons_to_networkx_graph(G: nx.Graph, t: int) -> None:
    """Generates a lava SNN and adds the neurons to the networkx Graph nodes.

    :param G: The original graph on which the MDSA algorithm is ran. nx.Graph:
    """
    # Verify the graph represents a connected and valid SNN, with all required
    # neuron and synapse properties specified.
    verify_networkx_snn_spec(G, t, backend="generic")

    # Convert networkx graph to an SNN network that can be ran by Lava.
    # starter_neuron = convert_networkx_graph_to_lava_snn(G)
    # pylint: disable=R0801
    # TODO: remove old_conversion and then remove the necessitiy for this
    # pylint disable command.
    (
        converted_nodes,
        _,
        neurons,
        _,
        neuron_dict,
    ) = initialise_networkx_to_snn_conversion(G)

    # Assert all nodes have been converted.
    if not len(converted_nodes) == len(G) and not len(neurons) == len(G):
        raise Exception("Not all nodes were converted.")

    # Assert the network is connected.
    # if not nx.is_connected(G):
    if not nx.is_strongly_connected(G) and not nx.is_weakly_connected(G):
        raise Exception(
            "Error, the network:{G} is not connected. (There are "
            + " separate/loose nodes)."
        )

    # Append neurons to networkx graph.
    append_neurons_to_networkx_graph(G, neuron_dict)

    # TODO: Verify all lava neurons are appended to networkx graph.
    verify_networkx_snn_spec(G, t, backend="lava")


def append_neurons_to_networkx_graph(G: nx.Graph, neuron_dict: dict) -> None:
    """Appends lava neuron objects as keys to the networkx graph nodes.

    :param G: The original graph on which the MDSA algorithm is ran. nx.Graph:
    :param neuron_dict: dict:
    """
    for node in G.nodes:
        neuron = list(neuron_dict.keys())[
            list(neuron_dict.values()).index(node)
        ]
        G.nodes[node]["lava_LIF"] = neuron

    # TODO: assert all neurons in the graph are unique.
