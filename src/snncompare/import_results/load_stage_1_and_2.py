"""Imports the following structure to an output file for simsnn:

/stage_1/run_config_name.json with content of stage1 algo dict.
    input_graph: nodes and edges
    snn_algo_graph: nodes, lif values and edges.
    adapted_snn_algo_graph: nodes, lif values and edges.
    radiation type, died neurons list without adaptation.
    radiation type, Died neurons list with adaptation.
"""
import json
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
import numpy as np
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    Radiation_data,
    get_radiation_names_filepath_and_exists,
    get_rand_nrs_and_hash,
)
from snncompare.helper import add_stage_completion_to_graph
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.progress_report.has_completed_stage1 import (
    has_outputted_stage_1,
)
from snncompare.run_config.Run_config import Run_config

from .read_json import load_json_file_into_dict


@typechecked
def load_input_graph(
    *,
    run_config: Run_config,
) -> nx.Graph:
    """Loads the input_graph."""
    # Get the json filename.
    output_dir: str = f"results/input_graphs/{run_config.graph_size}/"
    # TODO: add dict with isomorphic hashes to run_config, instead of graph
    # nrs.
    output_filepath: str = f"{output_dir}{run_config.isomorphic_hash}.json"
    input_graph_dict = load_json_file_into_dict(json_filepath=output_filepath)
    input_graph: nx.Graph = nx.Graph(**input_graph_dict)
    return input_graph


@typechecked
def input_graph_exists(
    *,
    run_config: Run_config,
) -> bool:
    """Returns True if the input graph file exists, False otherwise."""
    output_dir: str = f"results/input_graphs/{run_config.graph_size}/"
    # TODO: add dict with isomorphic hashes to run_config, instead of graph
    # nrs.
    output_filepath: str = f"{output_dir}{run_config.isomorphic_hash}.json"
    return Path(output_filepath).is_file()


@typechecked
def load_simsnn_graphs(
    *,
    run_config: Run_config,
    input_graph: nx.Graph,
    with_adaptation: bool,
    with_radiation: bool,
    stage_index: int,
) -> Simulator:
    """Loads the input_graph."""
    if not has_outputted_stage_1(
        input_graph=input_graph,
        run_config=run_config,
    ):
        raise SystemError("Can not load graph from file if it doesn't exist.")

    _, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)
    _, stage_1_simsnn_filepath = simsnn_files_exists_and_get_path(
        output_category="snns",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=with_adaptation,
        stage_index=1,
        rand_nrs_hash=rand_nrs_hash,
        rad_affected_neurons_hash=None,
    )

    return load_simsnn_graph_from_file(
        run_config=run_config,
        input_graph=input_graph,
        rand_nrs_hash=rand_nrs_hash,
        stage_1_simsnn_filepath=stage_1_simsnn_filepath,
        with_adaptation=with_adaptation,
        with_radiation=with_radiation,
        stage_index=stage_index,
    )


@typechecked
def load_simsnn_graph_from_file(
    *,
    run_config: Run_config,
    input_graph: nx.Graph,
    stage_1_simsnn_filepath: str,
    with_adaptation: bool,
    with_radiation: bool,
    rand_nrs_hash: str,
    stage_index: int,
) -> Simulator:
    """Loads the simsnn filepath and converts it into a simsnn graph file."""
    # Read output JSON file into dict.
    with open(stage_1_simsnn_filepath, encoding="utf-8") as json_file:
        some_dict: Dict[str, List] = json.load(json_file)
        json_file.close()

    stage_1_simsnn_simulator = stage1_simsnn_graph_from_file_to_simulator(
        add_to_raster=True,
        add_to_multimeter=True,
        simsnn_dict=some_dict,
    )
    add_stage_completion_to_graph(snn=stage_1_simsnn_simulator, stage_index=1)

    if with_radiation:
        radiation_data: Radiation_data = (
            get_radiation_names_filepath_and_exists(
                input_graph=input_graph,
                snn_graph=stage_1_simsnn_simulator,
                run_config=run_config,
                stage_index=stage_index,
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
        output_category = "snns"
        rad_affected_neurons_hash = None

    if stage_index == 2:
        simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
            output_category=output_category,
            input_graph=input_graph,
            run_config=run_config,
            with_adaptation=with_adaptation,
            stage_index=stage_index,
            rand_nrs_hash=rand_nrs_hash,
            rad_affected_neurons_hash=rad_affected_neurons_hash,
        )
        if simsnn_exists:
            load_snn_graph_stage_2(
                output_filepath=simsnn_filepath,
                stage_1_simsnn_simulator=stage_1_simsnn_simulator,
            )

            add_stage_completion_to_graph(
                snn=stage_1_simsnn_simulator, stage_index=2
            )
        else:
            raise FileNotFoundError(
                f"Error, simsnn not found at:{simsnn_filepath}"
            )
    return stage_1_simsnn_simulator


@typechecked
def stage1_simsnn_graph_from_file_to_simulator(
    *,
    add_to_raster: bool,
    add_to_multimeter: bool,
    simsnn_dict: Dict,
) -> Simulator:
    """Loads the simsnn filepath and converts it into a simsnn graph file."""
    net = Network()
    sim = Simulator(net, monitor_I=True)

    simsnn: Dict[str, LIF] = {}
    for neuron_dict in simsnn_dict["neurons"]:
        simsnn[neuron_dict["name"]] = net.createLIF(**neuron_dict)

    for synapse in simsnn_dict["synapses"]:
        edge_names = synapse["ID"]
        net.createSynapse(
            pre=simsnn[edge_names[0]],
            post=simsnn[edge_names[1]],
            ID=edge_names,
            w=synapse["w"],
            d=1,  # TODO: make explicit/load from file.
        )
    if add_to_raster:
        # Add all neurons to the raster.
        sim.raster.addTarget(net.nodes)
    if add_to_multimeter:
        # Add all neurons to the multimeter.
        sim.multimeter.addTarget(net.nodes)

    # TODO: Add (redundant) graph properties.
    return sim


@typechecked
def load_snn_graph_stage_2(
    *,
    output_filepath: str,
    stage_1_simsnn_simulator: Simulator,
) -> None:
    """Adds the spikes, I and V of an snn into a simsnn Simulator object."""
    # Verify the file exists.
    if not Path(output_filepath).is_file():
        raise FileExistsError(
            f"Error, filepath:{output_filepath} was not created."
        )

    loaded_snn: Dict = load_json_file_into_dict(json_filepath=output_filepath)

    for key, value in loaded_snn.items():
        loaded_snn[key] = np.array(value)
        if key == "spikes":
            stage_1_simsnn_simulator.raster.spikes = loaded_snn[key]
        elif key == "V":
            stage_1_simsnn_simulator.multimeter.V = loaded_snn[key]
        elif key == "I":
            stage_1_simsnn_simulator.multimeter.I = loaded_snn[key]
        else:
            raise KeyError(f"Error:{key} not supported in stage 2 snn dict.")
