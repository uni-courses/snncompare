"""Exports the following structure to an output file for simsnn:

/stage_2/
    snn_algo_graph: spikes, du, dv.
    adapted_snn_algo_graph: spikes, du, dv.
    rad_snn_algo_graph: spikes, du, dv.
    rad_adapted_snn_algo_graph: spikes, du, dv.
"""
import json
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_rand_nrs_and_hash,
)
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config
from snncompare.simulation.stage2_sim import (
    get_output_category_and_rad_affected_neuron_hash,
    simulate_load_or_skip,
)


@typechecked
def output_stage_2_snns(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    output_config: Output_config,
    run_config: Run_config,
) -> None:
    """Exports results dict to a json file."""
    stage_index: int = 2

    if not output_config.extra_storing_config.skip_stage_2_output:
        for with_adaptation in [False, True]:
            for with_radiation in [False, True]:
                next_action: str = simulate_load_or_skip(
                    output_config=output_config,
                    run_config=run_config,
                    stage_1_graphs=graphs_dict,
                    with_adaptation=with_adaptation,
                    with_radiation=with_radiation,
                )

                if next_action == "Simulate":
                    (
                        output_category,
                        rad_affected_neurons_hash,
                    ) = get_output_category_and_rad_affected_neuron_hash(
                        graphs_dict=graphs_dict,
                        run_config=run_config,
                        with_adaptation=with_adaptation,
                        with_radiation=with_radiation,
                        stage_index=stage_index,
                    )

                    _, rand_nrs_hash = get_rand_nrs_and_hash(
                        input_graph=graphs_dict["input_graph"]
                    )

                    # pylint:disable=R0801
                    (
                        simsnn_exists,
                        simsnn_filepath,
                    ) = simsnn_files_exists_and_get_path(
                        output_category=output_category,
                        input_graph=graphs_dict["input_graph"],
                        run_config=run_config,
                        with_adaptation=with_adaptation,
                        stage_index=stage_index,
                        rad_affected_neurons_hash=rad_affected_neurons_hash,
                        rand_nrs_hash=rand_nrs_hash,
                    )
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
        v: List = snn_graph.multimeter.V.tolist()
        i: List = snn_graph.multimeter.I.tolist()
        spikes: List = snn_graph.raster.spikes.tolist()
        neuron_dict: Dict = {"V": v, "I": i, "spikes": spikes}
        with open(output_filepath, "w", encoding="utf-8") as fp:
            json.dump(
                neuron_dict,
                fp,
                indent=4,
                sort_keys=True,
            )
            fp.close()

        # Verify the file exists.
        if not Path(output_filepath).is_file():
            raise FileExistsError(
                f"Error, filepath:{output_filepath} was not created."
            )
    else:
        raise NotImplementedError(f"Error, {type(snn_graph)} not supported.")
