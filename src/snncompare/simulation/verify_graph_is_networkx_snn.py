"""Verifies the graph represents a connected and valid SNN, with all required
neuron and synapse properties specified."""

from typing import Tuple

# Import the networkx module.
import networkx as nx
from _collections_abc import dict_keys
from networkx.classes.digraph import DiGraph


def verify_nx_neuron_properties_are_specified(
    node: nx.DiGraph.nodes, t: int
) -> None:
    """

    :param node: nx.DiGraph.nodes:
    :param node: nx.DiGraph.nodes:

    """
    if not isinstance(node["nx_LIF"][t].bias.get(), float):
        raise Exception("Bias is not a float.")
    if not isinstance(node["nx_LIF"][t].du.get(), float):
        raise Exception("du is not a float.")
    if not isinstance(node["nx_LIF"][t].dv.get(), float):
        raise Exception("dv is not a float.")
    if not isinstance(node["nx_LIF"][t].vth.get(), float):
        raise Exception("vth is not a float.")


def assert_synaptic_edgeweight_type_is_correct(
    G: nx.DiGraph, edge: nx.DiGraph.edges
) -> None:
    """

    :param edge: nx.DiGraph.edges:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param edge: nx.DiGraph.edges:

    """
    if nx.get_edge_attributes(G, "weight") != {}:

        # TODO: determine why a float is expected when the edge weights are
        # specified as ints.
        # TODO: make sure just 1 datatype, float is supported and used.
        # if not isinstance(G.edges[edge]["weight"], float):
        if not isinstance(G.edges[edge]["weight"], float) and not isinstance(
            G.edges[edge]["weight"], int
        ):
            raise Exception(
                f"Weight of edge {edge} is not a"
                + " float. It is"
                + f': {G.edges[edge]["weight"]} of type:'
                f'{type(G.edges[edge]["weight"])}'
            )
    else:
        raise Exception(
            f"Weight of edge {edge} is an attribute (in the"
            + ' form of: "weight").'
        )


def assert_synapse_properties_are_specified(
    G: DiGraph, edge: Tuple[int, int]
) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param edge:

    """
    if not check_if_synapse_properties_are_specified(G, edge):
        raise Exception(
            f"Not all synapse properties of edge: {edge} are"
            + " specified. It only contains attributes:"
            + f"{get_synapse_property_names(G,edge)}"
        )


def check_if_synapse_properties_are_specified(
    G: DiGraph, edge: Tuple[int, int]
) -> bool:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param edge:

    """
    synapse_property_names = get_synapse_property_names(G, edge)
    if "weight" in synapse_property_names:
        # if 'delay' in synapse_property_names:
        # TODO: implement delay using a chain of neurons in series since this
        # is not yet supported by lava-nc.

        return True
    return False


def get_synapse_property_names(G: DiGraph, edge: Tuple[int, int]) -> dict_keys:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param edge:

    """
    return G.edges[edge].keys()


def assert_no_duplicate_edges_exist(G: DiGraph) -> None:
    """Asserts no duplicate edges exist, throws error otherwise.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    visited_edges = []
    for edge in G.edges:
        if edge not in visited_edges:
            visited_edges.append(edge)
        else:
            raise Exception(
                f"Error, edge:{edge} is a duplicate edge as it"
                + f" already is in:{visited_edges}"
            )
