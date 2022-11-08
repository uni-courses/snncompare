"""TODO: change filename."""
from typing import List, Tuple  # nosec

import networkx as nx
from typeguard import typechecked


@typechecked
def get_desired_properties_for_graph_printing() -> List[str]:
    """Returns the properties that are to be printed to CLI."""
    desired_properties = [
        "bias",
        # "du",
        # "dv",
        "u",
        "v",
        "vth",
        "a_in_next",
    ]
    return desired_properties


@typechecked
def get_neurons(
    G: nx.DiGraph, sim_type: str, neuron_types: str
) -> Tuple[List, dict]:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param sim_type:
    :param neuron_types:

    """
    neurons_dict_per_type: dict = {}
    if sim_type not in ["nx_LIF", "lava_LIF"]:
        raise Exception(f"Unexpected simulation type demanded:{sim_type}")
    for neuron_type in neuron_types:
        if neuron_type not in [
            "counter",
            "spike_once",
            "degree_receiver",
            "rand",
            "selector",
        ]:
            raise Exception(f"Unexpected neuron_type demanded:{neuron_type}")
        neurons_dict_per_type[neuron_type] = []

    neurons = list(map(lambda x: G.nodes[x][sim_type], G.nodes))

    for nodename in G.nodes:
        for neuron_type in neuron_types:
            if nodename[: len(neuron_type)] == neuron_type:
                neurons_dict_per_type[neuron_type].append(
                    G.nodes[nodename][sim_type]
                )

    return neurons, neurons_dict_per_type
