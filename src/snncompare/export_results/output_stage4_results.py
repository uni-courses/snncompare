"""Exports the following structure to an output file for simsnn:

/stage_2/
    snn_algo_graph: spikes, du, dv.
    adapted_snn_algo_graph: spikes, du, dv.
    rad_snn_algo_graph: spikes, du, dv.
    rad_adapted_snn_algo_graph: spikes, du, dv.
"""
import json
from pathlib import Path
from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    Radiation_data,
    get_rad_name_filepath_and_exists,
    get_rand_nrs_and_hash,
)
from snncompare.export_results.output_stage2_snns import get_desired_snn_graph
from snncompare.helper import get_snn_graph_from_graphs_dict
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.run_config.Run_config import Run_config


@typechecked
def output_stage_4_results(
    *,
    run_config: Run_config,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> None:
    """Exports results dict to a json file.

    TODO: also output Neumann results.
    """
    for with_adaptation in [False, True]:
        _, rand_nrs_hash = get_rand_nrs_and_hash(
            input_graph=graphs_dict["input_graph"]
        )
        for with_radiation in [False, True]:
            # pylint:disable=R0801
            if with_radiation:
                snn_graph: Union[
                    nx.DiGraph, Simulator
                ] = get_snn_graph_from_graphs_dict(
                    with_adaptation=with_adaptation,
                    with_radiation=with_radiation,
                    graphs_dict=graphs_dict,
                )

                radiation_data: Radiation_data = (
                    get_rad_name_filepath_and_exists(
                        input_graph=graphs_dict["input_graph"],
                        snn_graph=snn_graph,
                        run_config=run_config,
                        stage_index=4,
                        with_adaptation=with_adaptation,
                    )
                )
                rad_affected_neurons_hash: Union[
                    None, str
                ] = radiation_data.rad_affected_neurons_hash
                output_category: str = (
                    f"{radiation_data.radiation_name}"
                    + f"_{radiation_data.radiation_parameter}"
                )
            else:
                rad_affected_neurons_hash = None
                output_category = "snns"

            # pylint:disable=R0801
            simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
                output_category=output_category,
                input_graph=graphs_dict["input_graph"],
                run_config=run_config,
                with_adaptation=with_adaptation,
                stage_index=4,
                rad_affected_neurons_hash=rad_affected_neurons_hash,
                rand_nrs_hash=rand_nrs_hash,
            )
            if not simsnn_exists:
                output_results(
                    output_filepath=simsnn_filepath,
                    snn_graph=get_desired_snn_graph(
                        graphs_dict=graphs_dict,
                        with_adaptation=with_adaptation,
                        with_radiation=with_radiation,
                    ),
                    simulator=run_config.simulator,
                )


@typechecked
def output_results(
    *,
    simulator: str,
    output_filepath: str,
    snn_graph: Union[nx.DiGraph, Simulator],
) -> None:
    """Outputs the stage 4 snn results to json."""

    if simulator == "simsnn":
        results = snn_graph.network.graph.graph["results"]
    elif simulator == "nx":
        results = snn_graph.graph["results"]
    else:
        raise NotImplementedError(
            f"Error, simulator:{simulator} not implemented."
        )

    with open(output_filepath, "w", encoding="utf-8") as fp:
        json.dump(
            results,
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

    # loaded_results: Dict = load_json_file_into_dict(
    #     json_filepath=output_filepath
    # )
