"""Checks whether the radiation graphs already exist in graph_dict, and adds
them if they don't."""
import copy
from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    Radiation_data,
    get_rad_name_filepath_and_exists,
)
from snncompare.export_results.output_stage2_snns import get_desired_snn_graph
from snncompare.helper import get_snn_graph_from_graphs_dict
from snncompare.import_results.helper import get_radiation_description
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
    radiation_name, _ = get_radiation_description(run_config=run_config)
    if radiation_name == "neuron_death":
        for with_adaptation in [False, True]:
            snn_graph: Union[
                nx.DiGraph, Simulator
            ] = get_snn_graph_from_graphs_dict(
                with_adaptation=with_adaptation,
                with_radiation=False,  # No radiation graph is needed to
                # compute which neurons are affected by radiation.
                graphs_dict=stage_1_graphs,
            )
            radiation_data: Radiation_data = get_rad_name_filepath_and_exists(
                input_graph=stage_1_graphs["input_graph"],
                snn_graph=snn_graph,
                run_config=run_config,
                stage_index=2,
                with_adaptation=with_adaptation,
            )
            snn_graph = get_desired_snn_graph(
                graphs_dict=stage_1_graphs,
                with_adaptation=with_adaptation,
                with_radiation=True,
            )
            print(radiation_data)
            raise NotImplementedError("TODO: apply radiation.")
    else:
        raise NotImplementedError(
            f"Error:{radiation_name} is not yet implemented."
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
