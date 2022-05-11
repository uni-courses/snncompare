# -*- coding: utf-8 -*-
# Converts networkx graph representing lava spiking-neural-network into SNN.

# Instantiate Lava processes to build network
import networkx as nx
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF

from verify_graph_is_snn import assert_all_synapse_properties_are_specified


def initialise_networkx_to_snn_conversion(G):
    """Prepares a networkx graph G to be converted into a Lava-nc neural
    network.

    :param G: Networkx graph that specifies the Lava neural network.
    """

    # 1. Start with first incoming node.
    first_node = list(G.nodes)[0]

    # Create dictionary with Lava LIF neurons as keys, neuron names as values.
    neuron_dict = {}

    (
        converted_nodes,
        lhs_neuron,
        neurons,
        lhs_nodename,
        neuron_dict,
        _,
    ) = convert_networkx_to_lava_snn(G, [], [], first_node, [], neuron_dict)
    return converted_nodes, lhs_neuron, neurons, lhs_nodename, neuron_dict


def convert_networkx_to_lava_snn(
    G, converted_nodes, neurons, lhs_nodename, visited_nodes, neuron_dict
):
    """Recursively converts the networkx graph G into a Lava SNN.

    :param G: Networkx graph that specifies the Lava neural network.
    :param converted_nodes: List of networkx nodenames that already have been
    converted to the Lava SNN.
    :param neurons: List of Lava neuron objects.
    :param lhs_nodename: The left-hand-side nodename that is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param visited_nodes: Nodes that have been the lhs node in this recursive
    conversion function. Neighbours are also converted, so neighbours can be
    converted but not visited.
    :param neuron_dict: Dictionary with Lava neuron objects as keys, and the
    nodename as items. (Default value = {})
    """
    # pylint: disable=too-many-arguments
    visited_nodes.append(lhs_nodename)

    # Incoming node, if it is not yet converted, then convert to neuron.
    if not node_is_converted(converted_nodes, lhs_nodename):
        (
            converted_nodes,
            lhs_neuron,
            neurons,
            lhs_nodename,
        ) = create_neuron_from_node(G, converted_nodes, neurons, lhs_nodename)
    else:
        lhs_neuron = get_neuron_belonging_to_node_from_list(
            neurons, lhs_nodename, converted_nodes
        )

    # For all edges of node, if synapse does not yet exists:
    # Is a set  because bi-directional edges create neighbour duplicates.
    for neighbour in set(nx.all_neighbors(G, lhs_nodename)):
        if neighbour not in visited_nodes:

            # Convert the neigbhour neurons of the lhs_nodename into a Lava
            # neuron.
            if not node_is_converted(converted_nodes, neighbour):
                (
                    converted_nodes,
                    rhs_neuron,
                    neurons,
                    _,
                ) = create_neuron_from_node(
                    G, converted_nodes, neurons, neighbour
                )
            else:
                # Even if the neighbour is already converted, the lhs and rhs
                # neurons are still retrieved to create a synapse between them.
                lhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons, lhs_nodename, converted_nodes
                )
                rhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons, neighbour, converted_nodes
                )

            # Create neuron dictionary, LIF objects as keys, neuron
            # descriptions as values.
            neuron_dict = add_neuron_to_dict(
                neighbour, neuron_dict, rhs_neuron
            )

            # Create synapse between lhs neuron and neighbour/rhs neuron.
            lhs_neuron = add_synapse_between_nodes(
                G, lhs_neuron, lhs_nodename, neighbour, rhs_neuron
            )

        # At the first time this function is called, initialise the dictionary.
        if len(visited_nodes) == 1:
            # print(f'ADD{lhs_nodename}')
            neuron_dict = add_neuron_to_dict(
                lhs_nodename, neuron_dict, lhs_neuron
            )

    # Recursively call that function on the neighbour neurons until no
    # new neurons are discovered.
    for neighbour in nx.all_neighbors(G, lhs_nodename):
        if neighbour not in visited_nodes:
            (
                converted_nodes,
                _,
                neurons,
                _,
                neuron_dict,
                visited_nodes,
            ) = convert_networkx_to_lava_snn(
                G,
                converted_nodes,
                neurons,
                neighbour,
                visited_nodes,
                neuron_dict,
            )
    return (
        converted_nodes,
        lhs_neuron,
        neurons,
        lhs_nodename,
        neuron_dict,
        visited_nodes,
    )


def node_is_converted(converted_nodes, nodename):
    """Verifies that the incoming node is not converted into a neuron yet.

    :param converted_nodes: List of networkx nodenames that already have been
    converted to the Lava SNN.
    :param neurons: List of Lava neuron objects.
    :param nodename: Name of the node of the networkx graph.
    """
    return nodename in converted_nodes


def create_neuron_from_node(G, converted_nodes, neurons, nodename):
    """

    :param G: Networkx graph that specifies the Lava neural network.

    :param converted_nodes: List of networkx nodenames that already have been
    converted to the Lava SNN.
    :param neurons: List of Lava neuron objects.
    :param nodename: Name of the node of the networkx graph.

    """

    bias, du, dv, vth = get_neuron_properties(G, nodename)

    neuron = LIF(bias=bias, du=du, dv=dv, vth=vth)

    # If spike_once_neuron, create recurrent synapse
    if nodename[0:11] == "spike_once_" or nodename[0:5] == "rand_":
        neuron = create_recurrent_synapse(neuron, -2)

    if nodename[0:16] == "degree_receiver_":
        neuron = create_recurrent_synapse(neuron, -20)
    # if nodename[0:6] == "count_":
    #    neuron = create_recurrent_synapse(neuron, -1)
    if nodename[0:6] == "delay_":
        neuron = create_recurrent_synapse(
            neuron, -(len(G) * 2 - 1) - 2
        )  # TODO: or -1?

    neurons.append(neuron)
    converted_nodes.append(nodename)
    return converted_nodes, neuron, neurons, nodename


def get_neuron_properties(G, nodename):
    """

    :param G: Networkx graph that specifies the Lava neural network.
    :param nodename: Name of the node of the networkx graph.

    """
    bias = G.nodes[nodename]["bias"]
    du = G.nodes[nodename]["du"]
    dv = G.nodes[nodename]["dv"]
    vth = G.nodes[nodename]["vth"]
    return bias, du, dv, vth


def create_recurrent_synapse(neuron, weight):
    """

    :param neuron: Lava neuron object.
    :param weight: Synaptic weight.
    """
    dense = create_weighted_synapse(weight)

    # Connect neuron to itself.
    neuron = connect_synapse(neuron, neuron, dense)
    return neuron


def create_weighted_synapse(w):
    """Creates a weighted synapse between neuron a and neuron b.

    :param w: Synaptic weight.
    """
    shape = (1, 1)
    # weights = np.random.randint(100, size=shape)
    weights = [[w]]  # Needs to be this shape for a 1-1 neuron connection.
    weight_exp = 2
    num_weight_bits = 7
    sign_mode = 1

    dense = Dense(
        shape=shape,
        weights=weights,
        weight_exp=weight_exp,
        num_weight_bits=num_weight_bits,
        sign_mode=sign_mode,
    )
    return dense


def connect_synapse(neuron_a, neuron_b, dense):
    """Connects a synapse named dense from neuron a to neuron b.

    :param neuron_a: Lava neuron object for lhs neuron.
    :param neuron_b: Lava neuron object for rhs neuron.
    :param dense: Lava object representing synapse.
    """
    neuron_a.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(neuron_b.in_ports.a_in)
    return neuron_a


def get_neuron_belonging_to_node_from_list(neurons, nodename, nodes):
    """

    :param neurons: List of Lava neuron objects.
    :param nodename: Name of the node of the networkx graph.
    :param nodes: List of nodenames of networkx graph.

    """
    index = nodes.index(nodename)
    return neurons[index]


def add_synapse_between_nodes(
    G, lhs_neuron, lhs_nodename, neighbour, rhs_neuron
):
    """

    :param G: Networkx graph that specifies the Lava neural network.
    :param lhs_neuron:
    :param lhs_nodename: The left-hand-side nodename that is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param neighbour: Name of the rhs node of the networkx graph.
    :param rhs_neuron:
    """
    # TODO: ensure the synapses are created in both directions.
    lhs_neuron = add_synapse_left_to_right(
        G, lhs_neuron, lhs_nodename, neighbour, rhs_neuron
    )
    lhs_neuron = add_synapse_right_to_left(
        G, lhs_neuron, lhs_nodename, neighbour, rhs_neuron
    )
    return lhs_neuron


def add_synapse_left_to_right(
    G, lhs_neuron, lhs_nodename, neighbour, rhs_neuron
):
    """

    :param G: Networkx graph that specifies the Lava neural network.
    :param lhs_neuron:
    :param lhs_nodename: The left-hand-side nodename that is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param neighbour: Name of the rhs node of the networkx graph.
    :param rhs_neuron:
    :param lhs_neuron:


    """
    # 3. Get the edge between lhs and rhs nodes. They are neighbours
    # so they have an edge by definition.However it is a directed graph.
    edge = get_edge_if_exists(G, lhs_nodename, neighbour)

    if edge is not None:
        # 3. Assert the synapses are fully specified.
        assert_all_synapse_properties_are_specified(G, edge)

        # 4. Create synapse between incoming node and neighbour.
        dense = create_weighted_synapse(G.edges[edge]["weight"])

        # 5. Connect neurons using created synapse.
        # TODO: write function that checks if synapse is created or not.
        lhs_neuron = connect_synapse_left_to_right(
            lhs_neuron, rhs_neuron, dense
        )
    return lhs_neuron


def add_synapse_right_to_left(
    G, lhs_neuron, lhs_nodename, neighbour, rhs_neuron
):
    """

    :param G: Networkx graph that specifies the Lava neural network.
    :param lhs_neuron:
    :param lhs_nodename: The left-hand-side nodename that is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param neighbour: Name of the rhs node of the networkx graph.
    :param rhs_neuron:
    :param rhs_node:
    :param lhs_neuron:

    """
    # 3. Get the edge between lhs and rhs nodes. They are neighbours
    # so they have an edge by definition.However it is a directed graph.
    edge = get_edge_if_exists(G, neighbour, lhs_nodename)

    if edge is not None:
        # 3. Assert the synapses are fully specified.
        assert_all_synapse_properties_are_specified(G, edge)

        # 4. Create synapse between incoming node and neighbour.
        dense = create_weighted_synapse(G.edges[edge]["weight"])

        # 5. Connect neurons using created synapse.
        # TODO: write function that checks if synapse is created or not.
        lhs_neuron = connect_synapse_right_to_left(
            lhs_neuron, rhs_neuron, dense
        )
    return lhs_neuron


def get_edge_if_exists(G, lhs_nodename, rhs_node):
    """Returns the edge object if the graph G has an edge between the two
    nodes. Returns None otherwise.

    :param G: Networkx graph that specifies the Lava neural network.
    :param lhs_nodename: The left-hand-side nodename that is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param rhs_node:
    """
    if G.has_edge(lhs_nodename, rhs_node):
        for edge in G.edges:
            if edge == (lhs_nodename, rhs_node):
                # print_edge_properties(G, edge)
                return edge
        # Verify at least an edge the other way round exists.
        if not G.has_edge(rhs_node, lhs_nodename):
            raise Exception(
                "Would expect an edge between a node and"
                + " its neighbour in the other direction."
            )
    # Verify at least an edge the other way round exists.
    if not G.has_edge(rhs_node, lhs_nodename):
        raise Exception(
            "Would expect an edge between a node and"
            + " its neighbour in the other direction."
        )
    return None


def connect_synapse_left_to_right(lhs_neuron, rhs_neuron, dense):
    """Connects a synapse named dense from lhs_neuron to rhs_neuron.

    :param lhs_neuron:
    :param rhs_neuron:
    :param dense:
    :param rhs_neuron:
    """
    lhs_neuron.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(rhs_neuron.in_ports.a_in)
    return lhs_neuron


def connect_synapse_right_to_left(lhs_neuron, rhs_neuron, dense):
    """Connects a synapse named dense from lhs_neuron to rhs_neuron.

    :param lhs_neuron:
    :param rhs_neuron:
    :param dense:
    :param rhs_neuron:
    """
    rhs_neuron.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(lhs_neuron.in_ports.a_in)
    return lhs_neuron


def add_neuron_to_dict(neighbour, neuron_dict, rhs_neuron):
    """

    :param neighbour: Name of the rhs node of the networkx graph.
    :param neuron_dict: Dictionary with Lava neuron objects as keys, and the
    nodename as items.
    :param rhs_neuron:

    """
    neuron_dict[rhs_neuron] = neighbour
    return neuron_dict
