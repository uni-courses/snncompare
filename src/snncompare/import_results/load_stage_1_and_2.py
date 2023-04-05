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
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    Radiation_data,
    Rand_nrs_data,
    get_input_graph_output_filepath,
    get_rad_name_filepath_and_exists,
    get_rand_nrs_and_hash,
    get_rand_nrs_data,
)
from snncompare.graph_generation.stage_1_create_graphs import (
    get_input_graph_of_run_config,
)
from snncompare.helper import add_stage_completion_to_graph
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.import_results.load_stage1_results import (
    get_run_config_filepath,
)
from snncompare.run_config.Run_config import Run_config

from .read_json import load_json_file_into_dict


@typechecked
def has_outputted_stage_1(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
) -> bool:
    """Returns True if the:

    - radiation names
    - random numbers
    - snn graph
    - (optional) adapted snn graphs
    have been outputted for the isomorphic hash belonging to this run_config.
    """
    for with_adaptation in [False, True]:
        if not has_outputted_snn_graph(
            input_graph=input_graph,
            run_config=run_config,
            with_adaptation=with_adaptation,
            stage_index=1,
        ):
            return False
        if not has_outputted_input_graph(
            input_graph=input_graph,
        ):
            return False
        json_filepath: str = get_run_config_filepath(run_config=run_config)
        if not Path(json_filepath).is_file():
            return False
    return True


def has_outputted_input_graph(
    *,
    input_graph: nx.Graph,
) -> bool:
    """Returns True if the rand_nrs for this run config has been outputted."""
    output_filepath: str = get_input_graph_output_filepath(
        input_graph=input_graph
    )
    return Path(output_filepath).is_file()


def has_outputted_snn_graph(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
    with_adaptation: bool,
    stage_index: int,
    rad_affected_neurons_hash: Optional[str] = None,
) -> bool:
    """Returns True if the rand_nrs for this run config has been outputted."""
    _, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)
    simsnn_exists, _ = simsnn_files_exists_and_get_path(
        output_category="snns",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=with_adaptation,
        stage_index=stage_index,
        rand_nrs_hash=rand_nrs_hash,
        rad_affected_neurons_hash=rad_affected_neurons_hash,
    )
    if simsnn_exists and stage_index == 1:
        # Check if rand_nrs are outputted.
        rand_nrs_data: Rand_nrs_data = get_rand_nrs_data(
            input_graph=input_graph,
            run_config=run_config,
            stage_index=stage_index,
        )
        if (
            rand_nrs_data.rand_nrs_file_exists
            and rand_nrs_data.seed_in_seed_hash_file
        ):
            # Check if radiation is outputted.
            radiation_data: Radiation_data = get_rad_name_filepath_and_exists(
                input_graph=input_graph,
                snn_graph=load_simsnn_graphs(
                    run_config=run_config,
                    input_graph=input_graph,
                    with_adaptation=with_adaptation,
                    with_radiation=False,
                    stage_index=1,
                ),
                run_config=run_config,
                stage_index=stage_index,
                with_adaptation=with_adaptation,
                rand_nrs_hash=None,
            )
            if (
                radiation_data.radiation_file_exists
                and radiation_data.seed_in_seed_hash_file
            ):
                return True
            return False
        return False
    return False


def has_outputted_rand_nrs(
    *, input_graph: nx.Graph, run_config: Run_config
) -> bool:
    """Returns True if the rand_nrs for this run config has been outputted."""
    rand_nrs_exists, rand_nrs_filepath = simsnn_files_exists_and_get_path(
        output_category="rand_nrs",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=False,
        stage_index=1,
    )
    if not rand_nrs_exists:
        return False
    # TODO: verify if file contains radiation neurons for this seed.
    raise FileNotFoundError(f"{rand_nrs_filepath} does not exist.")


@typechecked
def assert_has_outputted_stage_1(run_config: Run_config) -> None:
    """Throws error if stage 1 is not outputted."""
    if not has_outputted_stage_1(
        input_graph=get_input_graph_of_run_config(run_config=run_config),
        run_config=run_config,
    ):
        raise ValueError("Error, stage 1 was not completed.")


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
def load_stage1_simsnn_graphs(
    *,
    run_config: Run_config,
    input_graph: nx.Graph,
    stage_1_graphs_dict: Optional[Dict] = None,
) -> Dict:
    """Loads stage1 simsnn graphs and input graph."""
    if stage_1_graphs_dict is None:
        stage_1_graphs_dict = {}
        stage_1_graphs_dict["input_graph"] = get_input_graph_of_run_config(
            run_config=run_config
        )

    for with_adaptation in [False, True]:
        if with_adaptation:
            graph_name: str = "adapted_snn_graph"
        else:
            graph_name = "snn_algo_graph"
        stage_1_graphs_dict[graph_name] = load_simsnn_graphs(
            run_config=run_config,
            input_graph=input_graph,
            with_adaptation=with_adaptation,
            with_radiation=False,
            stage_index=1,
        )
    return stage_1_graphs_dict


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

    stage1_simsnn: Simulator = stage1_simsnn_graph_from_file_to_simulator(
        add_to_raster=True,
        add_to_multimeter=True,
        simsnn_dict=some_dict,
    )
    add_stage_completion_to_graph(snn=stage1_simsnn, stage_index=1)

    if with_radiation:
        radiation_data: Radiation_data = get_rad_name_filepath_and_exists(
            input_graph=input_graph,
            snn_graph=stage1_simsnn,
            run_config=run_config,
            stage_index=stage_index,
            with_adaptation=with_adaptation,
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

    if stage_index in [2, 4]:
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
            if stage_index == 2:
                load_snn_graph_stage_2(
                    output_filepath=simsnn_filepath,
                    stage_1_simsnn_simulator=stage1_simsnn,
                )
            elif stage_index == 4:
                add_stage4_results_from_file_to_snn(
                    output_filepath=simsnn_filepath,
                    stage_1_simsnn_simulator=stage1_simsnn,
                )
            else:
                raise NotImplementedError(
                    f"Error, loading stage:{stage_index} not supported."
                )

            add_stage_completion_to_graph(
                snn=stage1_simsnn, stage_index=stage_index
            )
        else:
            raise FileNotFoundError(
                f"Error, simsnn not found at:{simsnn_filepath}"
            )
    return stage1_simsnn


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


@typechecked
def add_stage4_results_from_file_to_snn(
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
    stage_1_simsnn_simulator.network.graph.graph["results"] = {}
    for key, value in loaded_snn.items():
        stage_1_simsnn_simulator.network.graph.graph["results"][key] = value
