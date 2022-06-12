import json
import pickle

import networkx as nx
from networkx.readwrite import json_graph

from src.plot_graphs import create_root_dir_if_not_exists


def export_end_results(
    dead_neuron_names,
    G: nx.DiGraph,
    G_behaviour_mdsa,
    G_behaviour_brain_adaptation,
    G_behaviour_rad_damage,
    brain_adaptation_graph: nx.DiGraph,
    has_adaptation: bool,
    has_radiation,
    iteration,
    m,
    mdsa_graph: nx.DiGraph,
    neuron_death_probability,
    rad_damaged_graph: nx.DiGraph,
    rand_props,
    scores,
    seed,
    selected_nodes,
    sim_time,
    unique_hash,
):

    create_root_dir_if_not_exists("results")

    output_name = f"_death_prob{neuron_death_probability}_adapt_{has_adaptation}_raddam{has_radiation}__seed{seed}_size{len(G)}_m{m}_iter{iteration}_hash{unique_hash}"

    export_results_as_json(
        G,
        dead_neuron_names,
        has_adaptation,
        has_radiation,
        iteration,
        m,
        neuron_death_probability,
        output_name,
        scores,
        seed,
        selected_nodes,
        sim_time,
    )

    export_graphs_as_pickle(
        brain_adaptation_graph,
        G,
        G_behaviour_mdsa,
        G_behaviour_brain_adaptation,
        G_behaviour_rad_damage,
        mdsa_graph,
        output_name,
        rad_damaged_graph,
        rand_props,
    )


def export_results_as_json(
    G: nx.DiGraph,
    dead_neuron_names,
    has_adaptation: bool,
    has_radiation,
    iteration,
    m,
    neuron_death_probability,
    output_name,
    scores,
    seed,
    selected_nodes,
    sim_time,
):

    dict = {
        "has_adaptation": has_adaptation,
        "has_radiation": has_radiation,
        "iteration": iteration,
        "m": m,
        "neuron_death_probability": neuron_death_probability,
        "seed": seed,
        "sim_time": sim_time,
        "G": digraph_to_json(G),
        "dead_neuron_names": dead_neuron_names,
        "scores": scores,
        "selected_nodes": selected_nodes,
    }

    with open(f"results/{output_name}.json", "w") as fp:
        json.dump(dict, fp)
    print(f"CALLED.")


def export_graphs_as_pickle(
    brain_adaptation_graph: nx.DiGraph,
    G: nx.DiGraph,
    G_behaviour_mdsa,
    G_behaviour_brain_adaptation,
    G_behaviour_rad_damage,
    mdsa_graph: nx.DiGraph,
    output_name,
    rad_damaged_graph: nx.DiGraph,
    rand_props,
):
    with open(
        f"results/{output_name}.pkl",
        "wb",
    ) as fh:
        pickle.dump(
            [
                G,
                G_behaviour_mdsa,
                G_behaviour_brain_adaptation,
                G_behaviour_rad_damage,
                mdsa_graph,
                brain_adaptation_graph,
                rad_damaged_graph,
                rand_props,
            ],
            fh,
        )


def digraph_to_json(G):
    if G is not None:
        return json_graph.node_link_data(G)
        # return json_graph.dumps(G)
    else:
        return None


def get_unique_hash(
    dead_neuron_names,
    has_adaptation,
    has_radiation,
    iteration,
    m,
    neuron_death_probability,
    seed,
    sim_time,
):
    if dead_neuron_names is None:
        dead_neuron_names = []

    # Needs to be frozen, because otherwise the set is mutable, and mutable
    # Python objects are not hashable.
    hash_set = frozenset(
        [
            frozenset(dead_neuron_names),
            has_adaptation,
            has_radiation,
            iteration,
            m,
            neuron_death_probability,
            seed,
            sim_time,
        ]
    )
    # set(dead_neuron_names),
    # hashable_set = [frozenset(i) for i in hash_set]

    # return hash(hashable_set)
    return hash(hash_set)


def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def sort_and_deduplicate(l):
    return list(uniq(sorted(l, reverse=True)))
