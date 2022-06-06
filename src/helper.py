import os
import pickle
import random
import shutil
import traceback
from datetime import datetime

import networkx as nx
import pylab as plt
from lava.proc.monitor.process import Monitor

from src import Plot_to_tex
from src.plot_graphs import create_root_dir_if_not_exists


def fill_dictionary(
    neuron_dict,
    neurons,
    previous_us,
    previous_vs,
    previous_selector=None,
    previous_has_spiked=None,
):
    sorted_neurons = sort_neurons(neurons, neuron_dict)
    for neuron in sorted_neurons:
        neuron_name = neuron_dict[neuron]
        previous_us[neuron_name] = 0
        previous_vs[neuron_name] = 0
        if previous_selector is not None:
            previous_selector[neuron_name] = 0
        if previous_has_spiked is not None:
            previous_has_spiked[neuron_name] = False

    if previous_selector is not None:
        if previous_has_spiked is not None:
            return (
                previous_us,
                previous_vs,
                previous_selector,
                previous_has_spiked,
            )
        else:
            return previous_us, previous_vs, previous_selector
    else:
        if previous_has_spiked is not None:
            return previous_us, previous_vs, previous_has_spiked
        else:
            return previous_us, previous_vs
    raise Exception("Expected to have returned the correct set.")


def sort_neurons(neurons, neuron_dict):
    sorted_neurons = []
    # Sort by value.
    sorted_dict = dict(sorted(neuron_dict.items(), key=lambda item: item[1]))
    for neuron, neuron_name in sorted_dict.items():
        if neuron in neurons:
            sorted_neurons.append(neuron)
    return sorted_neurons


def generate_list_of_n_random_nrs(G, max=None, seed=None):
    """Generates list of numbers in range of 1 to (and including) len(G), or:

    Generates list of numbers in range of 1 to (and including) max, or:
    TODO: Verify list does not contain duplicates, throw error if it does.
    """
    if max is None:
        return list(range(1, len(G) + 1))
    elif max == len(G):
        return list(range(1, len(G) + 1))
    elif max > len(G):
        large_list = list(range(1, max + 1))
        if seed is not None:
            random.seed(seed)
        return random.sample(large_list, len(G))


def get_y_position(G, node, neighbour, d):
    """Ensures the degree receiver nodes per node are aligned with continuous
    interval.

    for example for node 1, the positions 0,2,3 are mapped to positions:
    0,1,2 by subtracting 1.
    """
    if neighbour > node:
        return float((node + (neighbour - 1) / len(G)) * 4 * d)
    else:
        return float((node + neighbour / len(G)) * 4 * d)


def delete_files_in_folder(folder):
    os.makedirs(folder, exist_ok=True)
    try:
        shutil.rmtree(folder)
    except Exception:
        print(traceback.format_exc())
    os.makedirs(folder, exist_ok=False)


def export_get_degree_graph(
    adaptation,
    G,
    get_degree,
    iteration,
    m,
    neuron_death_probability,
    run_result,
    seed,
    size,
    test_object,
    unique_run_id,
):
    remove_monitors_from_get_degree(get_degree)
    create_root_dir_if_not_exists("pickles")
    with open(
        f"pickles/id{unique_run_id}_probability_{neuron_death_probability}"
        + f"_adapt_{adaptation}_{seed}_size{size}_m{m}_iter{iteration}.pkl",
        "wb",
    ) as fh:
        pickle.dump(
            [
                G,
                get_degree,
                iteration,
                m,
                run_result,
                seed,
                size,
                test_object.mdsa_graph,
                test_object.brain_adaptation_graph,
                test_object.first_rad_damage_graph,
                test_object.second_rad_damage_graph,
                test_object.first_dead_neuron_names,
                test_object.second_dead_neuron_names,
            ],
            fh,
        )


def remove_monitors_from_get_degree(get_degree):
    for node_name in get_degree.nodes:
        get_degree.nodes[node_name]["neuron"] = None
        get_degree.nodes[node_name]["spike_monitor"] = None
        get_degree.nodes[node_name]["spike_monitor_id"] = None


def get_counter_neurons_from_dict(expected_nr_of_neurons, neuron_dict, m):
    counter_neurons = []
    neurons = list(neuron_dict.keys())
    neuron_names = list(neuron_dict.values())
    # Get sorted counter neurons.
    for node_index in range(expected_nr_of_neurons):
        for neuron_name in neuron_names:
            if neuron_name == f"counter_{node_index}_{m}":
                counter_neurons.append(
                    get_neuron_from_dict(neuron_dict, neurons, neuron_name)
                )

    # pprint(f'neuron_names={neuron_names}')

    if expected_nr_of_neurons != len(counter_neurons):
        raise Exception(
            f"Error, expected {expected_nr_of_neurons} neurons, yet found"
            + f" {len(counter_neurons)} neurons"
        )
    return counter_neurons


def get_neuron_from_dict(neuron_dict, neurons, neuron_name):
    for neuron in neurons:
        if neuron_dict[neuron] == neuron_name:
            return neuron
    raise Exception("Did not find neuron:{neuron_name} in dict:{neuron_dict}")


def load_pickle_and_plot(
    adaptation,
    iteration,
    m,
    neuron_death_probability,
    seed,
    sim_time,
    size,
    unique_run_id,
):
    from src.helper_network_structure import plot_neuron_behaviour_over_time

    pickle_off = open(
        f"pickles/id{unique_run_id}_probability_{neuron_death_probability}"
        + f"_adapt_{adaptation}_{seed}_size{size}_m{m}_iter{iteration}.pkl",
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
        size,
        mdsa_graph,
        brain_adaptation_graph,
        first_rad_damage_graph,
        second_rad_damage_graph,
        first_dead_neuron_names,
        second_dead_neuron_names,
    ] = pickle.load(pickle_off)

    print(f"m={m}")
    print(f"adaptation={adaptation}")
    print(f"seed={seed}")
    print(f"size={size}")
    print(f"m={m}")
    print(f"iteration={iteration}")
    print(f"neuron_death_probability={neuron_death_probability}")

    print(f"dead_neuron_names={run_result.dead_neuron_names}")
    print(f"has_passed={run_result.has_passed}")
    print(f"amount_of_neurons={run_result.amount_of_neurons}")
    print(f"amount_synapses={run_result.amount_synapses}")
    print(f"has_adaptation={run_result.has_adaptation}")

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


def print_time(status, previous_time, previous_millis):
    now = datetime.now()
    # durationTime = (now - previous_time).total_seconds()
    now - previous_time
    import time

    now_millis = int(round(time.time() * 1000))

    duration_millis = now_millis - previous_millis
    print(
        f"{str(now.time())[:8]}, Duration:{duration_millis} [ms], status:{status}"
    )
    return now, now_millis


def write_results_to_file(
    has_passed, m, G, iteration, G_alipour, counter_neurons
):
    # Append-adds at last
    file1 = open("results.txt", "a")  # append mode
    now = datetime.now()
    file1.write(now.strftime("%Y-%m-%d %H:%M:%S\n"))
    file1.write(f"m={m}\n")
    file1.write(f"len(G)={len(G)}\n")
    file1.write(f"has_passed={has_passed,}\n")
    file1.write("edges\n")
    for edge in G.edges:
        file1.write(f"{str(edge)}\n")
    file1.write(f"iteration={iteration}\n")
    file1.write("G_alipour countermarks-SNN counter current\n")
    for node in G.nodes:
        file1.write(
            f'{G_alipour.nodes[node]["countermarks"]}-{counter_neurons[node].u.get()}\n'
        )
    file1.write("\n\n")
    file1.close()


def create_neuron_monitors(test_object, sim_time):
    get_degree = test_object.get_degree
    for node_name in get_degree.nodes:
        # The connecting node does not interact with the snn, it serves merely
        # to connect the snn for simulation purposes.
        if node_name != "connecting_node":
            neuron = get_neuron_from_dict(
                test_object.neuron_dict, test_object.neurons, node_name
            )

            if neuron is None:
                raise Exception(
                    "Error, was not able to find the neuron for node:{node_name}"
                )

            # Create monitor
            monitor = Monitor()

            # Specify what the monitor monitors, and for how long.
            monitor.probe(neuron.s_out, sim_time)

            # Get monitor process id
            monitor_process_id = list(monitor.get_data())[0]

            # Read out the boolean spike (at time t, with 1=1 being after 1 second of running.) or no spike with:
            # s_out=monitor.get_data()[monitor_process_id]["s_out"][t]
            # You need to call this again after every timestep.

            # Store monitor in node attribute.
            get_degree.nodes[node_name]["neuron"] = neuron
            get_degree.nodes[node_name]["spike_monitor"] = monitor
            get_degree.nodes[node_name][
                "spike_monitor_id"
            ] = monitor_process_id


def store_spike_values_in_neurons(get_degree, t):
    for node_name in get_degree.nodes:
        if node_name != "connecting_node":
            # Add neuron as attribute of node.

            monitor = get_degree.nodes[node_name]["spike_monitor"]
            monitor_process_id = get_degree.nodes[node_name][
                "spike_monitor_id"
            ]

            # TODO: doubt, or t=1?
            # simulation starts at time t=1, then 1 timestep is simulated, after
            # that time step, this monitoring function is called, which has the
            # first spike value stored in list index 0, hence t-1. Spike value
            # for t=2 is then stored in list index 1. (The first 0 is because
            #  it is a list of lists.)
            s_out = monitor.get_data()[monitor_process_id]["s_out"][t - 1][0]
            if s_out == 1:
                get_degree.nodes[node_name]["spike"][t] = True
            elif s_out == 0:
                get_degree.nodes[node_name]["spike"][t] = False
            else:
                raise Exception(
                    f"Was not able to compute spike or not for node:{node_name}"
                )


def full_alipour(
    delta,
    inhibition,
    iteration,
    G,
    rand_ceil,
    rand_nrs,
    m,
    seed,
    size,
    show=False,
    export=False,
):
    # Reverse engineer actual rand nrs:
    uninhibited_rand_nrs = [(x + inhibition) for x in rand_nrs]
    print(f"uninhibited_rand_nrs={uninhibited_rand_nrs}")
    print(f"m={m}")

    for node in G.nodes:
        # Initialise values.
        # G.nodes[node]["marks"] = 0
        G.nodes[node]["marks"] = G.degree(node) * (rand_ceil + 1) * delta
        G.nodes[node]["countermarks"] = 0
        G.nodes[node]["random_number"] = 1 * uninhibited_rand_nrs[node]
        G.nodes[node]["weight"] = (
            G.degree(node) * (rand_ceil + 1) * delta
            + G.nodes[node]["random_number"]
        )
        G.nodes[node]["inhibited_weight"] = (
            G.nodes[node]["weight"] - inhibition
        )

    if show or export:
        plot_alipour("0rand_mark", iteration, seed, size, 0, G, show=show)
        plot_alipour("1weight", iteration, seed, size, 0, G, show=show)
        plot_alipour("2inhib_weight", iteration, seed, size, 0, G, show=show)

    # Compute the mark based on degree+randomness=weight
    for node in G.nodes:
        max_weight = max(
            G.nodes[n]["weight"] for n in nx.all_neighbors(G, node)
        )

        nr_of_max_weights = 0
        for n in nx.all_neighbors(G, node):
            if (
                G.nodes[n]["weight"] == max_weight
            ):  # should all max weight neurons be marked or only one of them?

                # Always raise mark always by (rand_ceil + 1) * delta (not by 1).
                # Read of the score from countermarks, not marks.
                G.nodes[n]["marks"] += (rand_ceil + 1) * delta
                G.nodes[n]["countermarks"] += 1
                nr_of_max_weights = nr_of_max_weights + 1

                # Verify there is only one max weight neuron.
                if nr_of_max_weights > 1:
                    raise Exception("Two numbers with identical max weight.")

    # Don't compute for m=0
    for loop in range(1, m + 1):
        for node in G.nodes:
            G.nodes[node]["weight"] = (
                G.nodes[node]["marks"] + G.nodes[node]["random_number"]
            )
            G.nodes[node]["inhibited_weight"] = (
                G.nodes[node]["weight"] - inhibition
            )
            # Reset marks.
            G.nodes[node]["marks"] = 0
            G.nodes[node]["countermarks"] = 0

        for node in G.nodes:
            max_weight = max(
                G.nodes[n]["weight"] for n in nx.all_neighbors(G, node)
            )
            for n in nx.all_neighbors(G, node):
                if G.nodes[n]["weight"] == max_weight:

                    # Always raise mark always by (rand_ceil + 1) * delta (not by 1).
                    G.nodes[n]["marks"] += (rand_ceil + 1) * delta
                    G.nodes[n]["countermarks"] += 1

        if show or export:
            plot_alipour(
                "0rand_mark", iteration, seed, size, loop, G, show=show
            )
            plot_alipour("1weight", iteration, seed, size, loop, G, show=show)
            plot_alipour(
                "2inhib_weight", iteration, seed, size, loop, G, show=show
            )
        for node in G.nodes:
            print(f'node:{node}, ali-mark:{G.nodes[node]["countermarks"]}')
    return G


def plot_alipour(
    configuration, iteration, seed, size, m, G, export=True, show=False
):
    the_labels = get_alipour_labels(G, configuration=configuration)
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, labels=the_labels, with_labels=True)
    if show:
        plt.show()
    if export:
        plot_export = Plot_to_tex()
        plot_export.export_plot(
            plt,
            f"alipour_{seed}_size{size}_m{m}_iter{iteration}_combined_"
            + f"{configuration}",
        )

    plt.clf()
    plt.close()


def get_alipour_labels(G, configuration):
    labels = {}
    for node_name in G.nodes:
        if configuration == "0rand_mark":
            labels[
                node_name
            ] = f'{node_name},R:{G.nodes[node_name]["random_number"]}, M:'
            +f'{G.nodes[node_name]["marks"]}'
        elif configuration == "1weight":
            labels[
                node_name
            ] = f'{node_name}, W:{G.nodes[node_name]["weight"]}'
        elif configuration == "2inhib_weight":
            labels[
                node_name
            ] = f'{node_name}, W:{G.nodes[node_name]["inhibited_weight"]}'

    return labels
