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


def load_pickle_graphs():
    """Loads graphs from pickle files if they exist."""

    seed = 42
    random.seed(seed)
    unique_run_id = random.randrange(1, 10**6)

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
                        pickle_filename = (
                            f"pickles/id{unique_run_id}_probabilit"
                            + f"y_{neuron_death_probability}_adapt_{adaptation}_"
                            + f"{seed}_size{size}_m{m}_iter{iteration}.pkl"
                        )
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
                                first_rad_damage_graph,
                                second_rad_damage_graph,
                                first_dead_neuron_names,
                                second_dead_neuron_names,
                            ] = pickle.load(pickle_off)

                            properties_original_graph(G, iteration, size)
                            print("")
                            properties_mdsa_graph(
                                mdsa_graph, iteration, sim_time, size
                            )
                            print("")
                            properties_brain_adaptation_graph(
                                brain_adaptation_graph, iteration, size
                            )
                            print("")
                            properties_first_rad_damage_graph(
                                first_rad_damage_graph,
                                first_dead_neuron_names,
                                iteration,
                                size,
                            )
                            print("")
                            properties_second_rad_damage_graph(
                                second_rad_damage_graph,
                                second_dead_neuron_names,
                                iteration,
                                size,
                            )
                            exit()

                            print(f"m={m}")
                            print(f"adaptation={adaptation}")
                            print(f"seed={seed}")
                            print(f"size={size}")
                            print(f"m={m}")
                            print(f"iteration={iteration}")
                            print(
                                f"neuron_death_probability={neuron_death_probability}"
                            )

                            print(
                                f"dead_neuron_names={run_result.dead_neuron_names}"
                            )
                            print(f"has_passed={run_result.has_passed}")
                            print(
                                f"amount_of_neurons={run_result.amount_of_neurons}"
                            )
                            print(
                                f"amount_synapses={run_result.amount_synapses}"
                            )
                            print(
                                f"has_adaptation={run_result.has_adaptation}"
                            )

                            # plot_graph_behaviour(adaptation,get_degree,iteration,m,neuron_death_probability,seed,sim_time,size)

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


def properties_original_graph(G, iteration, size):
    """Shows the properties of the original graph."""
    # plot_uncoordinated_graph(G, export=False, show=True)
    pass


def properties_mdsa_graph(mdsa_graph, iteration, sim_time, size):
    """Shows the properties of the MDSA graph."""
    counter_neurons = print_graph_properties(mdsa_graph)
    old_graph_to_new_graph_properties(mdsa_graph)
    G_behaviour = simulate_graph(counter_neurons, mdsa_graph, sim_time)

    # plot_uncoordinated_graph(mdsa_graph,export=False,show=True)
    desired_properties = [
        "bias",
        "du",
        "dv",
        "u",
        "v",
        "vth",
        "a_in_next",
    ]
    for t in range(len(G_behaviour)):
        plot_coordinated_graph(
            G_behaviour[t],
            iteration,
            size,
            desired_properties=desired_properties,
            show=False,
            t=t,
        )
    exit()


def properties_brain_adaptation_graph(brain_adaptation_graph, iteration, size):
    """Shows the properties of the MDSA graph with brain adaptation."""
    counter_neurons = print_graph_properties(brain_adaptation_graph)
    old_graph_to_new_graph_properties(brain_adaptation_graph)
    plot_coordinated_graph(brain_adaptation_graph, iteration, size, show=True)


def properties_first_rad_damage_graph(
    first_rad_damage_graph, first_dead_neuron_names, iteration, size, show=True
):
    """Shows the properties of the MDSA graph with brain adaptation and the
    first radiation changes."""
    plot_coordinated_graph(first_rad_damage_graph, iteration, size, show=True)


def properties_second_rad_damage_graph(
    second_rad_damage_graph,
    second_dead_neuron_names,
    iteration,
    size,
    show=True,
):
    """Shows the properties of the MDSA graph with brain adaptation and the
    second radiation changes."""
    counter_neurons = print_graph_properties(second_rad_damage_graph)
    old_graph_to_new_graph_properties(second_rad_damage_graph)
    plot_coordinated_graph(second_rad_damage_graph, iteration, size, show=True)


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
    counter_neurons = []
    # Print graph properties.
    for nodename in G.nodes:
        print(nodename)
        print(f'bias={G.nodes[nodename]["bias"]}')
        print(f'du={G.nodes[nodename]["du"]}')
        print(f'dv={G.nodes[nodename]["dv"]}')
        print(f'vth={G.nodes[nodename]["vth"]}')
        if nodename[:7] == "counter":
            counter_neurons.append(nodename)

    for edge in G.edges:
        print(f'edge={edge},weight={G.edges[edge]["weight"]}')
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
