"""Exports the test results to a json file."""
import json
import pickle  # nosec - User is trusted not to load malicious pickle files.
from pathlib import Path
from typing import Any, Generator, List

import networkx as nx

from src.snncompare.export_results.export_nx_graph_to_json import (
    digraph_to_json,
)
from src.snncompare.export_results.plot_graphs import (
    create_root_dir_if_not_exists,
)


def write_dict_to_json(output_filepath: str, some_dict: dict) -> None:
    """Writes a dict file to a .json file."""
    with open(output_filepath, "w", encoding="utf-8") as fp:
        json.dump(some_dict, fp, indent=4, sort_keys=True)

    # Verify the file exists.
    if not Path(output_filepath).is_file():
        raise Exception(f"Error, filepath:{output_filepath} was not created.")
    # TODO: verify the file content is valid.


def export_end_results(
    dead_neuron_names: List[str],
    G: nx.DiGraph,
    G_behaviour_mdsa: nx.DiGraph,
    G_behaviour_brain_adaptation: nx.DiGraph,
    G_behaviour_rad_damage: nx.DiGraph,
    brain_adaptation_graph: nx.DiGraph,
    has_adaptation: bool,
    has_radiation: bool,
    iteration: int,
    m: int,
    mdsa_graph: nx.DiGraph,
    neuron_death_probability: float,
    rad_damaged_graph: nx.DiGraph,
    rand_props: Any,
    scores: Any,
    seed: int,
    selected_nodes: List[Any],
    sim_time: int,
    unique_hash: int,
) -> None:
    """
    TODO: Delete as end results are stored in json.
    Exports the results of the run to a json file and to a pickle file.

    :param dead_neuron_names:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param G_behaviour_mdsa:
    :param G_behaviour_brain_adaptation:
    :param G_behaviour_rad_damage:
    :param brain_adaptation_graph: nx.DiGraph:
    :param has_adaptation: bool:
    :param has_radiation: Indicates whether the experiment simulates radiation
    or not.
    param iteration: The initialisation iteration that is used.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    :param mdsa_graph: nx.DiGraph:
    :param neuron_death_probability:
    :param rad_damaged_graph: nx.DiGraph:
    :param rand_props:
    :param scores:
    :param seed: The value of the random seed used for this test.
    :param selected_nodes:
    :param sim_time: Nr. of timesteps for which the experiment is ran.
    :param unique_hash:
    """
    # pylint: disable=R0913
    # pylint: disable=R0914
    # One could perform work to cluster the properties into different objects.
    # Then one could read out these separate objects and write them to file.

    create_root_dir_if_not_exists("results")

    # Specify output filename for both json and pickle files.
    output_name = (
        f"_death_prob{neuron_death_probability}_adapt_{has_adaptation}_raddam"
        + f"{has_radiation}__seed{seed}_size{len(G)}_m{m}"
        + f"_iter{iteration}_hash{unique_hash}"
    )

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
    dead_neuron_names: List[str],
    has_adaptation: bool,
    has_radiation: bool,
    iteration: int,
    m: int,
    neuron_death_probability: float,
    output_name: str,
    scores: Any,
    seed: int,
    selected_nodes: List,
    sim_time: int,
) -> None:
    """Ensures the graphs can be exported to json format.
    TODO: delete because it is stored in json of graph.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param dead_neuron_names:
    :param has_adaptation: bool:
    :param has_radiation: Indicates whether the experiment simulates radiation
     or not.
    param iteration: The initialisation iteration that is used.
    :param m: The amount of approximation iterations used in the MDSA
     approximation.
    :param neuron_death_probability:
    :param output_name:
    :param scores:
    :param seed: The value of the random seed used for this test.
    :param selected_nodes:
    :param sim_time: Nr. of timesteps for which the experiment is ran.
    """
    # pylint: disable=R0913
    # One could perform work to cluster the properties into different objects.
    # Then one could read out these separate objects and write them to file.

    test_results_dict = {
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

    with open(f"results/{output_name}.json", "w", encoding="utf-8") as fp:
        json.dump(test_results_dict, fp)

    # TODO: verify the file exists.
    # TODO: verify the file content is valid.


def export_graphs_as_pickle(
    brain_adaptation_graph: nx.DiGraph,
    G: nx.DiGraph,
    G_behaviour_mdsa: nx.DiGraph,
    G_behaviour_brain_adaptation: nx.DiGraph,
    G_behaviour_rad_damage: nx.DiGraph,
    mdsa_graph: nx.DiGraph,
    output_name: str,
    rad_damaged_graph: nx.DiGraph,
    rand_props: Any,
) -> None:
    """
    TODO: remove

    :param brain_adaptation_graph: nx.DiGraph:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param G_behaviour_mdsa:
    :param G_behaviour_brain_adaptation:
    :param G_behaviour_rad_damage:
    :param mdsa_graph: nx.DiGraph:
    :param output_name:
    :param rad_damaged_graph: nx.DiGraph:
    :param rand_props:

    """
    # pylint: disable=R0913
    # One could perform work to cluster the properties into different objects.
    # Then one could read out these separate objects and write them to file.

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


def get_unique_hash(
    dead_neuron_names: List[str],
    has_adaptation: bool,
    has_radiation: bool,
    iteration: int,
    m: int,
    neuron_death_probability: float,
    seed: int,
    sim_time: int,
) -> int:
    """

    :param dead_neuron_names:
    :param has_adaptation:
    :param has_radiation: Indicates whether the experiment simulates radiation
    or not.
    param iteration: The initialisation iteration that is used.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    :param neuron_death_probability:
    :param seed: The value of the random seed used for this test.
    :param sim_time: Nr. of timesteps for which the experiment is ran.

    """
    # pylint: disable=R0913
    # One could perform work to cluster the properties into different objects.
    # Then one could read out these separate objects and write them to file.
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


def uniq(lst: List) -> Generator:
    """

    :param lst:

    """
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item


def sort_and_deduplicate(some_list: List[Any]) -> List[Any]:
    """

    :param list:

    """
    return list(uniq(sorted(some_list, reverse=True)))
