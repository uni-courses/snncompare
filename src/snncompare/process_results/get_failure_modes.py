"""Computes what the failure modes were, and then stores this data in the
graphs."""
from typing import Dict, List, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_rand_nrs_and_hash,
)
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.import_results.load_stage_1_and_2 import load_snn_graph_stage_2
from snncompare.run_config import Run_config


# pylint: disable=R0912
# pylint: disable=R0914
@typechecked
def add_failure_modes_to_graph(
    *,
    snn_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
) -> None:
    # ) -> Dict[str,Dict[int,List[str]]]:
    """Loads the SNN behaviour of an adapted radiated, and adapted unradiated
    SNN from file. If there is a difference between the unadapted and adapted.

    SNN, it exports:
     - the neuron names and timesteps at which the behaviour differs.
    """

    incorrectly_spikes: Dict[int, List[str]] = {}
    incorrectly_silent: Dict[int, List[str]] = {}

    # Get adapted unradiated SNN.
    adapted_unradiated_snn: Simulator = snn_graphs["adapted_snn_graph"]
    if "spikes" in adapted_unradiated_snn.raster.__dict__.keys():
        unradiated_spikes: List = adapted_unradiated_snn.raster.spikes.tolist()
    else:
        _, rand_nrs_hash = get_rand_nrs_and_hash(
            input_graph=snn_graphs["input_graph"]
        )
        simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
            output_category="snns",
            input_graph=snn_graphs["input_graph"],
            run_config=run_config,
            with_adaptation=True,
            stage_index=2,
            rand_nrs_hash=rand_nrs_hash,
            rad_affected_neurons_hash=None,
        )

        if simsnn_exists:
            load_snn_graph_stage_2(
                output_filepath=simsnn_filepath,
                stage_1_simsnn_simulator=adapted_unradiated_snn,
            )
        else:
            raise FileNotFoundError(
                "Error, was not able to find the SNN propagation results"
                + f" at:{simsnn_filepath}."
            )
        unradiated_spikes = adapted_unradiated_snn.raster.spikes.tolist()

    # Get adapted radiated SNN.
    adapted_radiated_snn: Simulator = snn_graphs["rad_adapted_snn_graph"]
    radiated_spikes: List = adapted_radiated_snn.raster.spikes.tolist()

    # Loop over timesteps
    for t, unradiated_spikes_at_t in enumerate(unradiated_spikes):
        # Loop over neurons
        for neuron_index, neuron_name in enumerate(
            list(
                map(
                    lambda neuron: neuron.name,
                    adapted_unradiated_snn.network.nodes,
                )
            )
        ):
            if t < len(radiated_spikes):
                if (
                    unradiated_spikes_at_t[neuron_index]
                    != radiated_spikes[t][neuron_index]
                ):
                    # pylint: disable=R1736
                    if unradiated_spikes[t][neuron_index]:
                        store_incorrect_spike(
                            failures=incorrectly_silent,
                            neuron_name=neuron_name,
                            t=t,
                        )
                    else:
                        store_incorrect_spike(
                            failures=incorrectly_spikes,
                            neuron_name=neuron_name,
                            t=t,
                        )

    for graph_name in snn_graphs.keys():
        if graph_name != "input_graph":
            if graph_name == "rad_adapted_snn_graph":
                snn_graphs[graph_name].network.graph.graph["failure_modes"] = {
                    "incorrectly_spikes": incorrectly_spikes,
                    "incorrectly_silent": incorrectly_silent,
                }
            else:
                snn_graphs[graph_name].network.graph.graph[
                    "failure_modes"
                ] = {}


@typechecked
def store_incorrect_spike(
    *,
    failures: Dict[int, List[str]],
    neuron_name: str,
    t: int,
) -> None:
    """Stores the time and adds the neuron name to the list."""
    if t not in failures.keys():
        failures[t] = []
    failures[t].append(neuron_name)
