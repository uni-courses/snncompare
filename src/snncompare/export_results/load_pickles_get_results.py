"""Loads pickle test result files ."""
import pickle  # nosec
from typing import Any, List, Tuple  # nosec

import networkx as nx


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


def load_pickle(pickle_filename: str) -> Any:
    """TODO: change to load hierarchic objects instead of parameter list.
    TODO: delete"""
    # pylint: disable=R0914
    # TODO: reduce the amount of local variables from 27/15 to at most 15/15.

    # Load graphs with encoded SNNs from pickle file.
    with open(
        pickle_filename,
        "rb",
    ) as pickle_off:
        # pylint: disable=R0801
        [
            has_adaptation,
            G,
            has_radiation,
            iteration,
            m,
            neuron_death_probability,
            rand_props,
            seed,
            sim_time,
            mdsa_graph,
            brain_adaptation_graph,
            rad_damaged_graph,
            dead_neuron_names,
            unique_hash,
        ] = pickle.load(  # nosec - User is trusted not to load malicious
            # pickle files.
            pickle_off
        )
    # pylint: disable=R0801
    return (
        has_adaptation,
        G,
        has_radiation,
        iteration,
        m,
        neuron_death_probability,
        rand_props,
        seed,
        sim_time,
        mdsa_graph,
        brain_adaptation_graph,
        rad_damaged_graph,
        dead_neuron_names,
        unique_hash,
    )
