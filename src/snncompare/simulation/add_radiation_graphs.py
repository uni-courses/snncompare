"""Checks whether the radiation graphs already exist in graph_dict, and adds
them if they don't."""
import copy
from typing import Dict

from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.graph_generation.stage_1_create_graphs import (
    get_new_radiation_graph,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def ensure_empty_rad_snns_exist(
    *,
    run_config: "Run_config",
    stage_1_graphs: Dict,
) -> None:
    """Copies the un-radiated snn graph into the radiated snn graph, for
    simulation."""

    if "rad_snn_algo_graph" not in stage_1_graphs.keys():
        stage_1_graphs["rad_snn_algo_graph"] = copy.deepcopy(
            stage_1_graphs["snn_algo_graph"]
        )
    if "rad_adapted_snn_graph" not in stage_1_graphs.keys():
        stage_1_graphs["rad_adapted_snn_graph"] = copy.deepcopy(
            stage_1_graphs["adapted_snn_graph"]
        )

    apply_radiation_to_empty_simsnn_graphs(
        run_config=run_config,
        stage_1_graphs=stage_1_graphs,
    )


@typechecked
def apply_radiation_to_empty_simsnn_graphs(
    *,
    run_config: "Run_config",
    stage_1_graphs: Dict,
) -> None:
    """Copies the un-radiated snn graph into the radiated snn graph, for
    simulation."""

    # Get the type of radiation used in this run_config.

    if run_config.radiation:
        for with_adaptation in [False, True]:
            if with_adaptation:
                stage_1_graphs[
                    "rad_adapted_snn_graph"
                ] = get_new_radiation_graph(
                    snn_graph=stage_1_graphs["adapted_snn_graph"],
                    run_config=run_config,
                )

            else:
                stage_1_graphs["rad_snn_algo_graph"] = get_new_radiation_graph(
                    snn_graph=stage_1_graphs["snn_algo_graph"],
                    run_config=run_config,
                )


@typechecked
def apply_radiation_death_to_empty_simsnn_neuron(
    *,
    neuron_name: str,
    snn_graph: Simulator,
) -> None:
    """Applies the radiation type to the simsnn neuron."""
    for simsnn_lif in snn_graph.network.nodes:
        if simsnn_lif.name == neuron_name:
            simsnn_lif.thr = 999
