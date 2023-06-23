"""Computes what the failure modes were, and then stores this data in the
graphs."""
from typing import Dict, List, Tuple, Union

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

    # Get adapted unradiated SNN.
    adapted_unradiated_snn: Simulator = snn_graphs["adapted_snn_graph"]

    unradiated_spikes, unradiated_I, _ = get_unradiated_spike_list(
        adapted_unradiated_snn=adapted_unradiated_snn,
        run_config=run_config,
        snn_graphs=snn_graphs,
    )

    (
        incorrectly_spikes,
        incorrectly_silent,
        excitatory_delta_u,
        inhibitory_delta_u,
    ) = get_incorrect_spikes(
        adapted_unradiated_snn=adapted_unradiated_snn,
        snn_graphs=snn_graphs,
        unradiated_I=unradiated_I,
        unradiated_spikes=unradiated_spikes,
    )

    for graph_name in snn_graphs.keys():
        if graph_name != "input_graph":
            if graph_name == "rad_adapted_snn_graph":
                snn_graphs[graph_name].network.graph.graph["failure_modes"] = {
                    "incorrectly_spikes": incorrectly_spikes,
                    "incorrectly_silent": incorrectly_silent,
                    "excitatory_delta_u": excitatory_delta_u,
                    "inhibitory_delta_u": inhibitory_delta_u,
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
    """Stores the time and adds the neuron name to the list.

    TODO: rename for its dual use case.
    """
    if t not in failures.keys():
        failures[t] = []
    failures[t].append(neuron_name)


@typechecked
def get_unradiated_spike_list(
    *,
    adapted_unradiated_snn: Simulator,
    run_config: Run_config,
    snn_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> Tuple[List[List[bool]], List[List[float]], List[List[float]]]:
    """Get the boolean list of spikes for the unradiated snn.

    This function may be called directly after simulating the SNNs, or
    after their behaviour has been stored to a file. That is why it
    first checks if the spikes are still in the incoming snn. Otherwise,
    it loads the spike behaviour from file.
    """
    if "spikes" in adapted_unradiated_snn.raster.__dict__.keys():
        # If spikes are (still) stored in incoming object, load them directly.
        unradiated_spikes: List[
            List[bool]
        ] = adapted_unradiated_snn.raster.spikes.tolist()
        unradiated_I: List[
            List[float]
        ] = adapted_unradiated_snn.multimeter.I.tolist()
        unradiated_V: List[
            List[float]
        ] = adapted_unradiated_snn.multimeter.V.tolist()
    else:  # Load the data from the snn behaviour file.
        # Get boilerplate data to receive the snn behaviour.
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
        # Store the boolean spike list for the unradiated snn.
        unradiated_spikes = adapted_unradiated_snn.raster.spikes.tolist()
        unradiated_I = adapted_unradiated_snn.multimeter.I.tolist()
        unradiated_V = adapted_unradiated_snn.multimeter.V.tolist()
    return unradiated_spikes, unradiated_I, unradiated_V


@typechecked
def get_incorrect_spikes(
    *,
    adapted_unradiated_snn: Simulator,
    snn_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    unradiated_I: List[List[float]],
    unradiated_spikes: List[List[bool]],
) -> Tuple[
    Dict[int, List[str]],
    Dict[int, List[str]],
    Dict[int, List[str]],
    Dict[int, List[str]],
]:
    """Creates dictionaries with the times at which neuron(s) of the radiated
    adapted SNN shows a different spike behaviour than the unradiated adapted
    SNN."""

    # Create the dictionaries with timestep and neuron names for the neurons
    # in the radiated SNN that behave different from those in the unadapted
    # snn.
    incorrectly_spikes: Dict[int, List[str]] = {}
    incorrectly_silent: Dict[int, List[str]] = {}

    excitatory_delta_u: Dict[int, List[str]] = {}
    inhibitory_delta_u: Dict[int, List[str]] = {}

    # Get adapted radiated SNN.
    adapted_radiated_snn: Simulator = snn_graphs["rad_adapted_snn_graph"]
    radiated_spikes: List[
        List[bool]
    ] = adapted_radiated_snn.raster.spikes.tolist()
    radiated_I: List[List[float]] = adapted_radiated_snn.multimeter.I.tolist()

    # Loop over timesteps
    # Loop over neurons
    for neuron_index, neuron_name in enumerate(
        list(
            map(
                lambda neuron: neuron.name,
                adapted_unradiated_snn.network.nodes,
            )
        )
    ):
        for t, unradiated_spikes_at_t in enumerate(unradiated_spikes):
            add_neurons_with_spike_difference(
                incorrectly_spikes=incorrectly_spikes,
                incorrectly_silent=incorrectly_silent,
                neuron_index=neuron_index,
                neuron_name=neuron_name,
                radiated_spikes=radiated_spikes,
                t=t,
                unradiated_spikes=unradiated_spikes,
                unradiated_spikes_at_t=unradiated_spikes_at_t,
            )

        for t, unradiated_I_at_t in enumerate(unradiated_I):
            add_neurons_with_u_difference(
                excitatory_delta_u=excitatory_delta_u,
                inhibitory_delta_u=inhibitory_delta_u,
                neuron_index=neuron_index,
                neuron_name=neuron_name,
                radiated_I=radiated_I,
                t=t,
                unradiated_I_at_t=unradiated_I_at_t,
            )
    return (
        incorrectly_spikes,
        incorrectly_silent,
        excitatory_delta_u,
        inhibitory_delta_u,
    )


@typechecked
def add_neurons_with_spike_difference(
    *,
    incorrectly_spikes: Dict[int, List[str]],
    incorrectly_silent: Dict[int, List[str]],
    neuron_index: int,
    neuron_name: str,
    radiated_spikes: List[List[bool]],
    t: int,
    unradiated_spikes: List[List[bool]],
    unradiated_spikes_at_t: List[bool],
) -> None:
    """Stores the neurons that show alternative spike behaviour."""
    # Check if a spike boolean is stored for each timestep.
    if t < len(radiated_spikes):
        # Check if the unradiated neuron behaves different than the
        # radiated neuron.
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
    sort_lists_in_dict_values(some_dict=incorrectly_silent)
    sort_lists_in_dict_values(some_dict=incorrectly_spikes)


@typechecked
def add_neurons_with_u_difference(
    *,
    excitatory_delta_u: Dict[int, List[str]],
    inhibitory_delta_u: Dict[int, List[str]],
    neuron_index: int,
    neuron_name: str,
    radiated_I: List[List[float]],
    t: int,
    unradiated_I_at_t: List[float],
) -> None:
    """Stores the neurons that show alternative spike behaviour."""
    # Check if a spike boolean is stored for each timestep.
    if t < len(radiated_I):
        # Check if the unradiated neuron behaves different than the
        # radiated neuron.
        if unradiated_I_at_t[neuron_index] < radiated_I[t][neuron_index]:
            store_incorrect_spike(
                failures=excitatory_delta_u,
                neuron_name=neuron_name,
                t=t,
            )
        elif unradiated_I_at_t[neuron_index] > radiated_I[t][neuron_index]:
            store_incorrect_spike(
                failures=inhibitory_delta_u,
                neuron_name=neuron_name,
                t=t,
            )
    sort_lists_in_dict_values(some_dict=excitatory_delta_u)
    sort_lists_in_dict_values(some_dict=inhibitory_delta_u)


@typechecked
def sort_lists_in_dict_values(
    *,
    some_dict: Dict[int, List[str]],
) -> None:
    """Sorts each list in this dictionary."""
    for key, some_list in some_dict.items():
        some_dict[key] = sorted(some_list)
