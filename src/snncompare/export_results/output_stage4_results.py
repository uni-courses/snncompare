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
def output_snn_results(
    *,
    output_data_type: str,
    run_config: Run_config,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    stage_index: int,
) -> None:
    """Exports results dict to a json file. TODO: also output Neumann results.

    Args:
    :output_data_type: (str), Specifies the type of output data.
    :run_config: (Run_config), Configuration settings for the run.
    :graphs_dict: (Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]]), A
    dictionary containing graphs or simulators.
    :stage_index: (int), Index indicating the stage.
    Returns:
    This function does not return anything.
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
                output_category: str = f"{radiation_data.radiation_name}"
            else:
                rad_affected_neurons_hash = None
                output_category = "snns"

            # pylint:disable=R0801
            simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
                output_category=output_category,
                input_graph=graphs_dict["input_graph"],
                run_config=run_config,
                with_adaptation=with_adaptation,
                stage_index=stage_index,
                rad_affected_neurons_hash=rad_affected_neurons_hash,
                rand_nrs_hash=rand_nrs_hash,
            )
            if not simsnn_exists:
                output_some_graph_property_dict(
                    dict_name=output_data_type,
                    output_filepath=simsnn_filepath,
                    snn_graph=get_desired_snn_graph(
                        graphs_dict=graphs_dict,
                        with_adaptation=with_adaptation,
                        with_radiation=with_radiation,
                    ),
                    simulator=run_config.simulator,
                )


@typechecked
def output_some_graph_property_dict(
    *,
    dict_name: str,
    simulator: str,
    output_filepath: str,
    snn_graph: Union[nx.DiGraph, Simulator],
) -> None:
    """Outputs the stage 4 SNN results to JSON.

    Args:
    :dict_name: (str), The name of the dictionary property within the graph.
    :simulator: (str), The type of simulator used, either "simsnn" or "nx".
    :output_filepath: (str), The file path where the JSON output will be
    saved.
    :snn_graph: (Union[nx.DiGraph, Simulator]), The graph object containing
    SNN results, either a NetworkX DiGraph or a Simulator object.
    Returns:
    No return value; writes SNN results to a JSON file.
    """
    if dict_name not in [
        "results",
        "failure_modes",
    ]:
        raise ValueError(
            f"Error, {dict_name} is not a supported graph property."
        )

    if simulator == "simsnn":
        dict_content = snn_graph.network.graph.graph[dict_name]
    elif simulator == "nx":
        dict_content = snn_graph.graph[dict_name]
    else:
        raise NotImplementedError(
            f"Error, simulator:{simulator} not implemented."
        )

    with open(output_filepath, "w", encoding="utf-8") as fp:
        json.dump(
            dict_content,
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
