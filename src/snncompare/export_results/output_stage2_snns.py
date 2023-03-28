"""Exports the following structure to an output file for simsnn:

/stage_2/
    snn_algo_graph: spikes, du, dv.
    adapted_snn_algo_graph: spikes, du, dv.
    rad_snn_algo_graph: spikes, du, dv.
    rad_adapted_snn_algo_graph: spikes, du, dv.
"""
import json
from typing import Dict, List, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    Radiation_data,
    get_radiation_names_filepath_and_exists,
    get_rand_nrs_and_hash,
)
from snncompare.helper import get_snn_graph_from_graphs_dict
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.run_config.Run_config import Run_config


@typechecked
def output_stage_2_snns(
    *,
    run_config: Run_config,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> None:
    """Exports results dict to a json file."""
    stage_index: int = 2

    for with_adaptation in [False, True]:
        for with_radiation in [False, True]:
            # pylint:disable=R0801
            if with_radiation:
                snn_graph: Union[
                    nx.DiGraph, Simulator
                ] = get_snn_graph_from_graphs_dict(
                    with_adaptation=with_adaptation,
                    with_radiation=False,  # No radiation graph is needed to
                    # compute which neurons are affected by radiation.
                    graphs_dict=graphs_dict,
                )

                radiation_data: Radiation_data = (
                    get_radiation_names_filepath_and_exists(
                        input_graph=graphs_dict["input_graph"],
                        snn_graph=snn_graph,
                        run_config=run_config,
                        stage_index=stage_index,
                        with_adaptation=with_adaptation,
                    )
                )
                rad_affected_neurons_hash: Union[
                    None, str
                ] = radiation_data.rad_affected_neurons_hash
            else:
                rad_affected_neurons_hash = None

            _, rand_nrs_hash = get_rand_nrs_and_hash(
                input_graph=graphs_dict["input_graph"]
            )

            # pylint:disable=R0801
            simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
                output_category="snns",
                input_graph=graphs_dict["input_graph"],
                run_config=run_config,
                with_adaptation=with_adaptation,
                stage_index=stage_index,
                rad_affected_neurons_hash=rad_affected_neurons_hash,
                rand_nrs_hash=rand_nrs_hash,
            )

            print(f"with_adaptation={with_adaptation}")
            print(f"with_radiation={with_radiation}")
            print(f"{simsnn_exists} at:{simsnn_filepath}")
            if not simsnn_exists:
                output_snn_graph_stage_2(
                    output_filepath=simsnn_filepath,
                    snn_graph=get_desired_snn_graph(
                        graphs_dict=graphs_dict,
                        with_adaptation=with_adaptation,
                        with_radiation=with_radiation,
                    ),
                )


@typechecked
def get_desired_snn_graph(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    with_adaptation: bool,
    with_radiation: bool,
) -> Union[nx.DiGraph, Simulator]:
    """Outputs the simsnn neuron behaviour over time."""
    if with_adaptation:
        if with_radiation:
            snn_graph = graphs_dict["rad_adapted_snn_graph"]
        else:
            snn_graph = graphs_dict["adapted_snn_graph"]
    else:
        if with_radiation:
            snn_graph = graphs_dict["rad_snn_algo_graph"]
        else:
            snn_graph = graphs_dict["snn_algo_graph"]

    return snn_graph


@typechecked
def output_snn_graph_stage_2(
    *,
    output_filepath: str,
    snn_graph: Union[nx.DiGraph, Simulator],
) -> None:
    """Outputs the simsnn neuron behaviour over time."""
    # TODO: change this into an object with: name and a list of parameters
    # instead.

    if isinstance(snn_graph, Simulator):
        v: List[float] = snn_graph.multimeter.V.tolist()
        i: List[float] = snn_graph.multimeter.I.tolist()
        spikes: List[bool] = snn_graph.raster.spikes.tolist()
        neuron_dict: Dict = {"V": v, "I": i, "spikes": spikes}
        with open(output_filepath, "w", encoding="utf-8") as fp:
            json.dump(
                neuron_dict,
                fp,
                indent=4,
                sort_keys=True,
            )
            fp.close()
