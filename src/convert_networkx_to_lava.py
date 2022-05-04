# Converts networkx graph representing lava spiking-neural-network into SNN.

# Instantiate Lava processes to build network
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
import networkx as nx

from verify_graph_is_snn import assert_all_synapse_properties_are_specified
from src.helper import add_neuron_to_dict


def initialise_networkx_to_snn_conversion(G):

    # 1. Start with first incoming node.
    first_node = list(G.nodes)[0]

    # Create dictionary with Lava LIF neurons as keys, neuron names as values.
    neuron_dict = {}

    (
        converted_nodes,
        lhs_neuron,
        neurons,
        lhs_node,
        neuron_dict,
        visited_nodes,
    ) = convert_networkx_to_lava_snn(G, [], [], first_node, [], neuron_dict)
    return converted_nodes, lhs_neuron, neurons, lhs_node, neuron_dict


def convert_networkx_to_lava_snn(
    G, converted_nodes, neurons, lhs_node, visited_nodes, neuron_dict={}
):

    visited_nodes.append(lhs_node)

    # Incoming node, if it is not yet converted, then convert to neuron.
    if not node_is_converted(G, converted_nodes, neurons, lhs_node):
        (
            converted_nodes,
            lhs_neuron,
            neurons,
            lhs_node,
        ) = create_neuron_from_node(G, converted_nodes, neurons, lhs_node)
    else:
        lhs_neuron = get_neuron_belonging_to_node_from_list(
            neurons, lhs_node, converted_nodes
        )

    # For all edges of node, if synapse does not yet  exists:
    # Is a set  because bi-directional edges create neighbour duplicates.
    for neighbour in set(nx.all_neighbors(G, lhs_node)):
        if neighbour not in visited_nodes:

            # Ensure target neuron is created.
            if not node_is_converted(G, converted_nodes, neurons, neighbour):
                (
                    converted_nodes,
                    rhs_neuron,
                    neurons,
                    rhs_node,
                ) = create_neuron_from_node(
                    G, converted_nodes, neurons, neighbour
                )
            else:
                lhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons, lhs_node, converted_nodes
                )
                rhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons, neighbour, converted_nodes
                )

            # Create neuron dictionary, LIF objects as keys, neuron
            # descriptions as values.
            neuron_dict = add_neuron_to_dict(
                neighbour, neuron_dict, rhs_neuron
            )

            # 5. Add synapse
            lhs_neuron = add_synapse_between_nodes(
                G, lhs_neuron, lhs_node, neighbour, rhs_neuron, neighbour
            )
        if len(visited_nodes) == 1:
            # print(f'ADD{lhs_node}')
            neuron_dict = add_neuron_to_dict(lhs_node, neuron_dict, lhs_neuron)

    # 6. recursively call that function on the neighbour neurons until no
    # new neurons are discovered.
    for neighbour in nx.all_neighbors(G, lhs_node):
        if neighbour not in visited_nodes:
            if neighbour not in visited_nodes:
                (
                    converted_nodes,
                    discarded_neuron,
                    neurons,
                    discarded_node,
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
        lhs_node,
        neuron_dict,
        visited_nodes,
    )


def node_is_converted(G, converted_nodes, neurons, node):
    """Verifies that the incoming node is not converted into
    a neuron yet."""
    return node in converted_nodes


def create_neuron_from_node(G, converted_nodes, neurons, node):

    bias, du, dv, vth = get_neuron_properties(G, node)

    neuron = LIF(bias=bias, du=du, dv=dv, vth=vth)

    # If spike_once_neuron, create recurrent synapse
    if node[0:11] == "spike_once_" or node[0:5] == "rand_":
        neuron = create_recurrent_synapse(neuron, -2)

    if node[0:16] == "degree_receiver_":
        neuron = create_recurrent_synapse(neuron, -20)
    # if node[0:6] == "count_":
    #    neuron = create_recurrent_synapse(neuron, -1)
    if node[0:6] == "delay_":
        neuron = create_recurrent_synapse(
            neuron, -(len(G) * 2 - 1) - 2
        )  # TODO: or -1?

    neurons.append(neuron)
    converted_nodes.append(node)
    return converted_nodes, neuron, neurons, node


def get_neuron_properties(G, node):
    bias = G.nodes[node]["bias"]
    du = G.nodes[node]["du"]
    dv = G.nodes[node]["dv"]
    vth = G.nodes[node]["vth"]
    return bias, du, dv, vth


def create_recurrent_synapse(neuron, weight):
    dense = create_weighted_synapse(weight)

    # Connect neuron to itself.
    neuron = connect_synapse(neuron, neuron, dense)
    return neuron


def create_weighted_synapse(w):
    """
    Creates a weighted synapse between neuron a and neuron b.
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
    """Connects a synapse named dense from neuron a to neuron b."""
    neuron_a.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(neuron_b.in_ports.a_in)
    return neuron_a


def get_neuron_belonging_to_node_from_list(neurons, node, nodes):
    index = nodes.index(node)
    return neurons[index]


def add_synapse_between_nodes(
    G, lhs_neuron, lhs_node, neighbour, rhs_neuron, rhs_node
):
    # TODO: ensure the synapses are created in both directions.
    lhs_neuron = add_synapse_left_to_right(
        G, lhs_neuron, lhs_node, neighbour, rhs_neuron, rhs_node
    )
    lhs_neuron = add_synapse_right_to_left(
        G, lhs_neuron, lhs_node, neighbour, rhs_neuron, rhs_node
    )
    return lhs_neuron


def add_synapse_left_to_right(
    G, lhs_neuron, lhs_node, neighbour, rhs_neuron, rhs_node
):
    # 3. Get the edge between lhs and rhs nodes. They are neighbours
    # so they have an edge by definition.However it is a directed graph.
    edge = get_edge_if_exists(G, lhs_node, neighbour)

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
    G, lhs_neuron, lhs_node, neighbour, rhs_neuron, rhs_node
):
    # 3. Get the edge between lhs and rhs nodes. They are neighbours
    # so they have an edge by definition.However it is a directed graph.
    edge = get_edge_if_exists(G, neighbour, lhs_node)

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


def get_edge_if_exists(G, lhs_node, rhs_node):
    """Returns the edge object if the graph G has an edge between the two
    nodes. Returns None otherwise."""
    if G.has_edge(lhs_node, rhs_node):
        for edge in G.edges:
            if edge == (lhs_node, rhs_node):
                # print_edge_properties(G, edge)
                return edge
        # Verify at least an edge the other way round exists.
        if not G.has_edge(rhs_node, lhs_node):
            raise Exception(
                "Would expect an edge between a node and"
                + " its neighbour in the other direction."
            )
    # Verify at least an edge the other way round exists.
    if not G.has_edge(rhs_node, lhs_node):
        raise Exception(
            "Would expect an edge between a node and"
            + " its neighbour in the other direction."
        )


def connect_synapse_left_to_right(lhs_neuron, rhs_neuron, dense):
    """Connects a synapse named dense from lhs_neuron to rhs_neuron."""
    lhs_neuron.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(rhs_neuron.in_ports.a_in)
    return lhs_neuron


def connect_synapse_right_to_left(lhs_neuron, rhs_neuron, dense):
    """Connects a synapse named dense from lhs_neuron to rhs_neuron."""
    rhs_neuron.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(lhs_neuron.in_ports.a_in)
    return lhs_neuron
