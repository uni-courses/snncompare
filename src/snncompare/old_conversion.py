"""Old code used to convert the networkx graph to an MDSA SNN algorithm."""
from typing import Any, Dict, List, Tuple, Union

import networkx as nx

from src.snncompare.graph_generation.convert_networkx_to_lava import (
    add_neuron_to_dict,
    add_synapse_between_nodes,
    create_neuron_from_node,
    node_is_converted,
)


def convert_networkx_graph_to_snn_with_one_neuron(
    G: nx.DiGraph,
) -> Tuple[List, List, List, Any, dict]:
    """

    :param G: The original graph on which the MDSA algorithm is ran.

    """
    # TODO: rewrite function to:
    # 0. Verify the graph is connected (no lose nodes exist).
    # 1. Start with first incoming node.
    first_node = list(G.nodes)[0]

    # Append dictionary as property to G.
    neuron_dict: dict = {}

    (
        converted_nodes,
        lhs_neuron,
        neurons,
        lhs_node,
        neuron_dict,
        _,
    ) = retry_build_snn(G, [], [], first_node, [], neuron_dict)
    return converted_nodes, lhs_neuron, neurons, lhs_node, neuron_dict


def retry_build_snn(
    G: nx.DiGraph,
    converted_nodes: List,
    neurons: List,
    lhs_node: Any,
    visited_nodes: List,
    neuron_dict: dict = None,
) -> Tuple[
    List, Any, List, Any, Union[Dict[Any, int], Dict[Any, Any]], List[Any]
]:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param converted_nodes:
    :param neurons:
    :param lhs_node:
    :param visited_nodes: param neuron_dict:  (Default value = {})
    :param converted_nodes:
    :param lhs_node:
    :param neuron_dict:  (Default value = {})

    """
    # pylint: disable=R0913
    # TODO: reduce input arguments from 6/5 to at most 5/5.
    if neuron_dict is None:
        neuron_dict = {}
    # Verify prerequisites
    # print_node_properties(G, lhs_node)
    assert_all_neuron_properties_are_specified(G, lhs_node)
    # TODO: assert graph G is connected.

    visited_nodes.append(lhs_node)

    # Incoming node, if it is not yet converted, then convert to neuron.
    if not node_is_converted(converted_nodes, lhs_node):
        (
            converted_nodes,
            lhs_neuron,
            neurons,
            lhs_node,
        ) = create_neuron_from_node(
            G, converted_nodes, neurons, lhs_node, old_code=True
        )
    else:
        lhs_neuron = get_neuron_belonging_to_node_from_list(
            neurons, lhs_node, converted_nodes
        )

    # For all edges of node, if synapse does not yet  exists:
    # Needs to be a set  because bi-directional edges create neighbour
    # duplicates.
    for neighbour in set(nx.all_neighbors(G, lhs_node)):
        if neighbour not in visited_nodes:

            # Ensure target neuron is created.
            if not node_is_converted(converted_nodes, neighbour):
                (
                    converted_nodes,
                    rhs_neuron,
                    neurons,
                    _,
                ) = create_neuron_from_node(
                    G, converted_nodes, neurons, neighbour, old_code=True
                )
            else:
                lhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons, lhs_node, converted_nodes
                )
                rhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons, neighbour, converted_nodes
                )

            # Create a neuron dictionary which returns the node name if you
            #  input a neuron.
            neuron_dict = add_neuron_to_dict(
                neighbour, neuron_dict, rhs_neuron
            )

            # 5. Add synapse
            lhs_neuron = add_synapse_between_nodes(
                # G, lhs_neuron, lhs_node, neighbour, rhs_neuron, neighbour
                G,
                lhs_neuron,
                lhs_node,
                neighbour,
                rhs_neuron,
            )
        if len(visited_nodes) == 1:
            neuron_dict = add_neuron_to_dict(lhs_node, neuron_dict, lhs_neuron)

    # 6. recursively call that function on the neighbour neurons until no
    # new neurons are discovered.
    for neighbour in nx.all_neighbors(G, lhs_node):
        if neighbour not in visited_nodes:
            if neighbour not in visited_nodes:
                (
                    converted_nodes,
                    _,
                    neurons,
                    _,
                    neuron_dict,
                    visited_nodes,
                ) = retry_build_snn(
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


def get_neuron_belonging_to_node_from_list(
    neurons: List[Any], node: int, nodes: List[int]
) -> List[Any]:
    """

    :param neurons: param node:
    :param nodes:
    :param node:

    """
    index = nodes.index(node)
    return neurons[index]


def get_node_belonging_to_neuron_from_list(
    neuron: int, neurons: List[int], nodes: List[int]
) -> int:
    """

    :param neuron: Lava neuron object. param neurons:
    :param nodes:
    :param neurons:

    """
    index = neurons.index(neuron)
    return nodes[index]


def get_edge_if_exists(
    G: nx.DiGraph, lhs_node: int, rhs_node: int
) -> Union[None, Tuple[int, int]]:
    """Returns the edge object if the graph G has an edge between the two
    nodes.

    Returns None otherwise.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param rhs_node:
    :param lhs_node:
    """
    if G.has_edge(lhs_node, rhs_node):
        for edge in G.edges:
            if edge == (lhs_node, rhs_node):
                # print_edge_properties(G, edge)
                return edge
        # Verify at least an edge the other way round exists.
        if not G.has_edge(rhs_node, lhs_node):
            raise Exception(
                "Would expect an edge between a node and its neighbour in the"
                + " other direction."
            )
    # Verify at least an edge the other way round exists.
    if not G.has_edge(rhs_node, lhs_node):
        raise Exception(
            "Would expect an edge between a node and its neighbour in the"
            + " other direction."
        )
    return None


def assert_all_neuron_properties_are_specified(
    G: nx.DiGraph, node: str
) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param node:

    """
    if not all_neuron_properties_are_specified(G, node):
        raise Exception(
            f"Not all neuron prpoerties of node: {node} are specified. It only"
            + f" contains attributes:{get_neuron_property_names(G,node)}"
        )


def all_neuron_properties_are_specified(G: nx.DiGraph, node: str) -> bool:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param node:

    """
    neuron_property_names = get_neuron_property_names(G, node)
    # if ['bias', 'du', 'dv','vth'] in neuron_properties:
    if "bias" in neuron_property_names:
        if "du" in neuron_property_names:
            if "dv" in neuron_property_names:
                if "vth" in neuron_property_names:
                    return True
    return False


def get_neuron_property_names(G: nx.DiGraph, node: str) -> List:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param node:

    """
    return G.nodes[node].keys()
