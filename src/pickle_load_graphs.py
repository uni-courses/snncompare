import pickle
import random
from src.LIF_neuron import LIF_neuron, print_neuron_properties

from src.helper import file_exists
from src.helper_network_structure import (
    plot_coordinated_graph,
    plot_neuron_behaviour_over_time,
)
from src.plot_graphs import plot_uncoordinated_graph
from src.run_on_networkx import run_snn_on_networkx
from src.verify_graph_is_snn import verify_networkx_snn_spec


def get_desired_properties_for_graph_printing():
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


def load_pickle_graphs():
    """Loads graphs from pickle files if they exist."""

    seed = 42
    random.seed(seed)
    unique_run_id = random.randrange(1, 10**6)
    print(f"unique_run_id={unique_run_id}")
    desired_properties = get_desired_properties_for_graph_printing()

    for m in range(0, 1):
        for iteration in range(0, 2, 1):
            for size in range(3, 4, 1):
                # for neuron_death_probability in [0.1, 0.25, 0.50]:
                for neuron_death_probability in [
                    0.01,
                    0.05,
                    0.1,
                    0.2,
                    0.25,
                ]:
                    for adaptation in [True, False]:
                        configuration = (
                            f"id{unique_run_id}_probabilit"
                            + f"y_{neuron_death_probability}_adapt_{adaptation}_"
                            + f"{seed}_size{size}_m{m}_iter{iteration}"
                        )
                        pickle_filename = f"pickles/{configuration}.pkl"
                        print(f"pickle_filename={pickle_filename}")
                        if file_exists(pickle_filename):

                            pickle_off = open(
                                pickle_filename,
                                "rb",
                            )

                            # [G, get_degree, iteration, m, run_result, seed, size] = pickle.load(
                            [
                                G,
                                get_degree,
                                iteration,
                                m,
                                run_result,
                                seed,
                                sim_time,
                                size,
                                mdsa_graph,
                                brain_adaptation_graph,
                                second_rad_damage_graph,
                                second_dead_neuron_names,
                            ] = pickle.load(pickle_off)

                            ###properties_original_graph(
                            ###    configuration, G, iteration, size
                            ###)
                            ###print("")
                            ###properties_mdsa_graph(
                            ###    configuration,
                            ###    desired_properties,
                            ###    mdsa_graph,
                            ###    iteration,
                            ###    sim_time,
                            ###    size,
                            ###)

                            print("")
                            if adaptation:
                                ###properties_brain_adaptation_graph(
                                ###    configuration,
                                ###    desired_properties,
                                ###    brain_adaptation_graph,
                                ###    iteration,
                                ###    sim_time,
                                ###    size,
                                ###)
                                ###print("")
                                if neuron_death_probability > 0.05:
                                    # Assume if no adaptation is implemented, that also no radiation
                                    # is implemented.
                                    properties_second_rad_damage_graph(
                                        configuration,
                                        desired_properties,
                                        second_rad_damage_graph,
                                        second_dead_neuron_names,
                                        iteration,
                                        sim_time,
                                        size,
                                    )
                        else:
                            print(f"Did not find:{pickle_filename}")


def plot_graph_behaviour(
    adaptation,
    get_degree,
    iteration,
    m,
    neuron_death_probability,
    seed,
    sim_time,
    size,
):
    for t in range(sim_time - 1):
        print(f"in helper, t={t},sim_time={sim_time}")
        plot_neuron_behaviour_over_time(
            adaptation,
            f"pickle_probability_{neuron_death_probability}_adapt_{adaptation}_{seed}_size{size}_m{m}_iter{iteration}_t{t}",
            get_degree,
            iteration,
            seed,
            size,
            m,
            t + 1,
            show=False,
            current=True,
        )


def properties_original_graph(configuration, G, iteration, size):
    """Shows the properties of the original graph."""
    # plot_uncoordinated_graph(G, export=False, show=True)
    pass


def properties_mdsa_graph(
    configuration, desired_properties, mdsa_graph, iteration, sim_time, size
):
    """Shows the properties of the MDSA graph."""
    counter_neurons = get_counter_neurons(mdsa_graph)
    old_graph_to_new_graph_properties(mdsa_graph)
    G_behaviour = simulate_graph(counter_neurons, mdsa_graph, sim_time)

    # plot_uncoordinated_graph(mdsa_graph,export=False,show=True)
    for t in range(len(G_behaviour)):
        plot_coordinated_graph(
            G_behaviour[t],
            iteration,
            size,
            desired_properties=desired_properties,
            show=False,
            filename=f"mdsa_{configuration}_t={t}",
        )


def properties_brain_adaptation_graph(
    configuration,
    desired_properties,
    brain_adaptation_graph,
    iteration,
    sim_time,
    size,
):
    """Shows the properties of the MDSA graph with brain adaptation."""
    print(f"brain_adaptation_graph={brain_adaptation_graph}")
    counter_neurons = get_counter_neurons(brain_adaptation_graph)

    old_graph_to_new_graph_properties(brain_adaptation_graph)
    G_behaviour = simulate_graph(
        counter_neurons, brain_adaptation_graph, sim_time
    )
    for t in range(len(G_behaviour)):
        plot_coordinated_graph(
            G_behaviour[t],
            iteration,
            size,
            desired_properties=desired_properties,
            show=False,
            filename=f"brain_adaptation_{configuration}_t={t}",
        )


def properties_second_rad_damage_graph(
    configuration,
    desired_properties,
    second_rad_damage_graph,
    second_dead_neuron_names,
    iteration,
    sim_time,
    size,
    show=True,
):
    """Shows the properties of the MDSA graph with brain adaptation and the
    second radiation changes."""
    print(f"second_rad_damage_graph={second_rad_damage_graph}")
    counter_neurons = get_counter_neurons(second_rad_damage_graph)
    old_graph_to_new_graph_properties(second_rad_damage_graph)

    G_behaviour = simulate_graph(
        counter_neurons, second_rad_damage_graph, sim_time
    )
    print(f"len(G_behaviour)={len(G_behaviour)}")
    for t in range(len(G_behaviour)):
        plot_coordinated_graph(
            G_behaviour[t],
            iteration,
            size,
            desired_properties=desired_properties,
            show=False,
            filename=f"second_rad_{configuration}_t={t}",
        )


def old_graph_to_new_graph_properties(G):
    for nodename in G.nodes:
        G.nodes[nodename]["nx_LIF"] = LIF_neuron(
            name=nodename,
            bias=float(G.nodes[nodename]["bias"]),
            du=float(G.nodes[nodename]["du"]),
            dv=float(G.nodes[nodename]["dv"]),
            vth=float(G.nodes[nodename]["vth"]),
        )
    verify_networkx_snn_spec(G)


def print_graph_properties(G):

    # Print graph properties.
    for nodename in G.nodes:
        print(nodename)
        print(f'bias={G.nodes[nodename]["bias"]}')
        print(f'du={G.nodes[nodename]["du"]}')
        print(f'dv={G.nodes[nodename]["dv"]}')
        print(f'vth={G.nodes[nodename]["vth"]}')

    for edge in G.edges:
        print(f'edge={edge},weight={G.edges[edge]["weight"]}')


def get_counter_neurons(G):
    counter_neurons = []
    for nodename in G.nodes:
        if nodename[:7] == "counter":
            counter_neurons.append(nodename)
    return counter_neurons


def simulate_graph(counter_neurons, G, sim_time):

    G_behaviour = []
    for t in range(sim_time + 2):
        G_behaviour.extend(run_snn_on_networkx(G, 1))
    return G_behaviour


def get_neurons(G, sim_type, neuron_types):
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
        else:
            neurons_dict_per_type[neuron_type] = []

    neurons = list(map(lambda x: G.nodes[x][sim_type], G.nodes))

    for nodename in G.nodes:
        for neuron_type in neuron_types:
            if nodename[: len(neuron_type)] == neuron_type:
                neurons_dict_per_type[neuron_type].append(
                    G.nodes[nodename][sim_type]
                )

    return neurons, neurons_dict_per_type
