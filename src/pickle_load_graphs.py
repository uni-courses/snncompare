import pickle
import random

from src.helper import file_exists
from src.helper_network_structure import plot_neuron_behaviour_over_time


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
