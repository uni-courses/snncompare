"""Applies brain adaptation to a MDSA SNN graph."""
import copy
from typing import Any

import networkx as nx

from src.snncompare.old_conversion import (
    convert_networkx_graph_to_snn_with_one_neuron,
)
from tests.exp_setts.unsorted.test_create_testobject import add_monitor_to_dict


def implement_adaptation_mechanism(
    adaptation_graph: nx.DiGraph,
    # m,
) -> nx.DiGraph:
    """
    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    """

    # Create a copy of the original list of nodes of the input graph.
    original_nodes = copy.deepcopy(adaptation_graph.nodes)

    for node_name in original_nodes:
        # Get input synapses as dictionaries, one per node, store as node
        # attribute.
        store_input_synapses(adaptation_graph, node_name)

        # Get output synapses as dictionaries, one per node, store as node
        # attribute.
        store_output_synapses(adaptation_graph, node_name)

        # Create redundant neurons.
        create_redundant_node(adaptation_graph, node_name)

    # Start new loop before adding edges, because all reduundant neurons need
    # to exist before creating synapses.
    for node_name in original_nodes:
        # Add input synapses to redundant node.
        add_input_synapses(adaptation_graph, node_name)

        # Add output synapses  to redundant node.
        add_output_synapses(adaptation_graph, node_name)

        # Add inhibitory synapse from node to redundant node.
        # TODO: set edge weight
        add_inhibitory_synapse(adaptation_graph, node_name)

        # TODO: Add recurrent self inhibitory synapse for some redundant nodes.
        add_recurrent_inhibitiory_synapses(adaptation_graph, node_name)


def store_input_synapses(adaptation_graph: nx.DiGraph, node_name: str) -> None:
    """

    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param node_name: Node of the name of a networkx graph.

    """
    input_edges = []
    for edge in adaptation_graph.edges:
        if edge[1] == node_name:
            input_edges.append(edge)
    adaptation_graph.nodes[node_name]["input_edges"] = input_edges


def store_output_synapses(
    adaptation_graph: nx.digraph, node_name: str
) -> None:
    """

    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param node_name: Node of the name of a networkx graph.

    """
    output_edges = []
    for edge in adaptation_graph.edges:
        if edge[0] == node_name:
            output_edges.append(edge)
    adaptation_graph.nodes[node_name]["output_edges"] = output_edges


def create_redundant_node(
    adaptation_graph: nx.digraph, node_name: str
) -> None:
    """Create neuron and set coordinate position.

    :param d: Unit length of the spacing used in the positions of the nodes for
    plotting.
    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param node_name: Node of the name of a networkx graph.
    """
    # TODO: get d from node from algorithm graph.
    vth = compute_vth_for_delay(adaptation_graph, node_name)
    adaptation_graph.add_node(
        f"red_{node_name}",
        du=adaptation_graph.nodes[node_name]["du"],
        dv=adaptation_graph.nodes[node_name]["dv"],
        bias=adaptation_graph.nodes[node_name]["bias"],
        vth=vth,
        pos=(
            float(
                adaptation_graph.nodes[node_name]["pos"][0]
                + 0.25 * adaptation_graph.graph["alg_props"]["d"]
            ),
            float(
                adaptation_graph.nodes[node_name]["pos"][1]
                - 0.25 * adaptation_graph.graph["alg_props"]["d"]
            ),
        ),
        spike={},
        is_redundant=True,
    )


def compute_vth_for_delay(
    adaptation_graph: nx.digraph, node_name: str
) -> float:
    """Increases vth with 1 to realise a delay of t=1 for the redundant
    spike_once neurons, rand neurons and selector neurons.

    Returns dv of default node otherwise.

    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param node_name: Node of the name of a networkx graph.
    """
    if (
        node_name[:11] == "spike_once_"
        or node_name[:5] == "rand_"
        # or node_name[:9] == "selector_"
        or node_name[:16] == "degree_receiver_"
    ):
        vth = adaptation_graph.nodes[node_name]["vth"] + 1
    else:
        vth = adaptation_graph.nodes[node_name]["vth"]
    return vth


def add_input_synapses(adaptation_graph: nx.digraph, node_name: str) -> None:
    """

    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param node_name: Node of the name of a networkx graph.

    """
    for edge in adaptation_graph.nodes[node_name]["input_edges"]:
        # Compute set edge weight
        left_node = edge[0]
        right_node = f"red_{node_name}"
        weight = adaptation_graph[edge[0]][edge[1]]["weight"]

        # Create edge
        adaptation_graph.add_edge(
            left_node, right_node, weight=weight, is_redundant=True
        )


def add_output_synapses(adaptation_graph: nx.digraph, node_name: str) -> None:
    """

    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param node_name: Node of the name of a networkx graph.

    """
    adaptation_graph.add_edges_from(
        adaptation_graph.nodes[node_name]["output_edges"]
    )
    for edge in adaptation_graph.nodes[node_name]["output_edges"]:
        # Compute set edge weight
        left_node = f"red_{node_name}"
        right_node = edge[1]
        weight = adaptation_graph[edge[0]][edge[1]]["weight"]

        # Create edge
        adaptation_graph.add_edge(
            left_node, right_node, weight=weight, is_redundant=True
        )


def add_inhibitory_synapse(
    adaptation_graph: nx.DiGraph, node_name: str
) -> None:
    """

    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param node_name: Node of the name of a networkx graph.

    """
    # TODO: compute what minimum inhibitory weight should be in network to
    # prevent all neurons from spiking.
    adaptation_graph.add_edges_from(
        [(node_name, f"red_{node_name}")], weight=-100
    )
    # TODO: set edge weight


def add_recurrent_inhibitiory_synapses(
    adaptation_graph: nx.DiGraph, nodename: str
) -> None:
    """

    :param adaptation_graph: Graph with the MDSA SNN approximation solution.
    :param nodename: Node of the name of a networkx graph.

    """
    if "recur" in adaptation_graph.nodes[nodename].keys():
        adaptation_graph.add_edges_from(
            [
                (
                    f"red_{nodename}",
                    f"red_{nodename}",
                )
            ],
            weight=adaptation_graph.nodes[nodename]["recur"],
        )


def convert_new_graph_to_snn(test_object: Any, sim_time: int) -> Any:
    """

    :param test_object: Object containing test settings.
    :param sim_time: Nr. of timesteps for which the experiment is ran.

    """
    # Convert the snn networkx graph into a Loihi implementation.
    (
        test_object.converted_nodes,
        test_object.lhs_neuron,
        test_object.neurons,
        test_object.lhs_node,
        test_object.neuron_dict,
    ) = convert_networkx_graph_to_snn_with_one_neuron(
        test_object.adaptation_graph
    )

    # Create monitor dict
    test_object.monitor_dict = {}
    for neuron in test_object.neurons:
        test_object.monitor_dict = add_monitor_to_dict(
            neuron, test_object.monitor_dict, sim_time
        )
    return test_object
