import glob
import pickle

from src.export_json_results import export_end_results
from src.helper import delete_files_in_folder, get_counter_neurons
from src.helper_network_structure import plot_coordinated_graph
from src.LIF_neuron import LIF_neuron
from src.process_results import get_run_results
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
    delete_files_in_folder("latex/Images/graphs")
    delete_files_in_folder("results")

    desired_properties = get_desired_properties_for_graph_printing()

    # Loop through the pickles that contain the graphs that will be simulated.
    for pickle_filename in glob.iglob("pickles/*.pkl"):
        last_G_brain_adaptation = None
        last_G_rad_dam = None

        pickle_off = open(
            pickle_filename,
            "rb",
        )

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
        ] = pickle.load(pickle_off)

        # TODO: verify unique_hash equals output of: get_unique_hash().
        output_name = f"_death_prob{neuron_death_probability}_adapt_{has_adaptation}_raddam{has_radiation}__seed{seed}_size{len(G)}_m{m}_iter{iteration}_hash{unique_hash}"
        if (
            has_adaptation
            and has_radiation
            and neuron_death_probability > 0.05
        ):
            G_behaviour_mdsa = get_graph_behaviour(mdsa_graph, sim_time)
            plot_graph_behaviour(
                G_behaviour_mdsa,
                iteration,
                len(G),
                desired_properties,
                f"mdsa_{output_name}",
            )
            if has_adaptation:

                G_behaviour_brain_adaptation = get_graph_behaviour(
                    brain_adaptation_graph, sim_time
                )
                plot_graph_behaviour(
                    G_behaviour_brain_adaptation,
                    iteration,
                    len(G),
                    desired_properties,
                    f"brain_{output_name}",
                )
                last_G_brain_adaptation = G_behaviour_brain_adaptation[-1]

                if neuron_death_probability > 0.05:
                    # Assume if no adaptation is implemented, that also no radiation
                    # is implemented.
                    G_behaviour_rad_damage = get_graph_behaviour(
                        rad_damaged_graph, sim_time
                    )
                    plot_graph_behaviour(
                        G_behaviour_rad_damage,
                        iteration,
                        len(G),
                        desired_properties,
                        f"rad_dam_{output_name}",
                    )
                    last_G_rad_dam = G_behaviour_rad_damage[-1]

            print(f"pickle_filename={pickle_filename}")

            export_end_results(
                dead_neuron_names,
                G,
                G_behaviour_mdsa,
                G_behaviour_brain_adaptation,
                G_behaviour_rad_damage,
                brain_adaptation_graph,
                has_adaptation,
                has_radiation,
                iteration,
                m,
                mdsa_graph,
                neuron_death_probability,
                rad_damaged_graph,
                rand_props,
                seed,
                sim_time,
                unique_hash,
            )

            get_run_results(
                G,
                G_behaviour_mdsa[-1],
                last_G_brain_adaptation,
                last_G_rad_dam,
                m,
                rand_props,
            )


def get_graph_behaviour(G, sim_time):
    counter_neurons = get_counter_neurons(G)
    old_graph_to_new_graph_properties(G)
    G_behaviour = simulate_graph(counter_neurons, G, sim_time)
    return G_behaviour


def plot_graph_behaviour(
    G_behaviour, iteration, size, desired_properties, output_name
):
    if False:
        for t in range(len(G_behaviour)):
            plot_coordinated_graph(
                G_behaviour[t],
                iteration,
                size,
                desired_properties=desired_properties,
                show=False,
                filename=f"{output_name}_t={t}",
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
