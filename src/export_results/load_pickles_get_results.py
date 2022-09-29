"""Loads pickle test result files ."""
import pickle  # nosec - User is trusted not to load malicious pickle files.

from src.graph_generation.helper_network_structure import (
    plot_coordinated_graph,
)
from src.helper import old_graph_to_new_graph_properties

# from src.process_results.process_results import get_run_results
from src.simulation.run_on_networkx import run_snn_on_networkx


def get_desired_properties_for_graph_printing():
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


def generate_output_graphs(
    graphs, identifiers, desired_properties, output_name
):
    """Generates the output graphs of the SNNs over time.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param graphs:
    :param identifiers:
    :param desired_properties:
    :param output_name:
    """
    # pylint: disable=R0913
    # TODO: reduce the amount of arguments from 6/5 to at most 5/5.
    for i in enumerate(graphs):
        plot_graph_behaviour(
            graphs[i],
            desired_properties,
            f"{identifiers[i]}_{output_name}",
        )


def get_graph_behaviour(G, sim_time):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param sim_time: Nr. of timesteps for which the experiment is ran.

    """
    old_graph_to_new_graph_properties(G)
    G_behaviour = simulate_graph(G, sim_time)
    return G_behaviour


def plot_graph_behaviour(G_behaviour, desired_properties, output_name):
    """

    :param G_behaviour:
    :param desired_properties:
    :param output_name:

    """
    for t in enumerate(G_behaviour):
        plot_coordinated_graph(
            G_behaviour[t],
            desired_properties,
            t,
            False,
            filename=f"{output_name}_t={t}",
        )


def simulate_graph(G, sim_time):
    """

    :param counter_neurons: Neuron objects at the counter position.
    Type unknown.
    :param G: The original graph on which the MDSA algorithm is ran.
    :param sim_time: Nr. of timesteps for which the experiment is ran.

    """

    G_behaviour = []
    for _ in range(sim_time + 2):
        G_behaviour.extend(run_snn_on_networkx(G, 1))
    return G_behaviour


def get_neurons(G, sim_type, neuron_types):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param sim_type:
    :param neuron_types:

    """
    neurons_dict_per_type = {}
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


def load_pickle(pickle_filename):
    """TODO: change to load hierarchic objects instead of parameter list."""
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
