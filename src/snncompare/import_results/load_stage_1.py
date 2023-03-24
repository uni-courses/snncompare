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
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_snn_graphs import (
    simsnn_files_exists_and_get_path,
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
) -> Union[None, Simulator]:
    """Loads the input_graph."""
    simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=with_adaptation,
    )
    if simsnn_exists:
        return load_simsnn_graph_from_file(
            simsnn_filepath=simsnn_filepath,
        )
    return None


@typechecked
def load_simsnn_graph_from_file(
    *,
    simsnn_filepath: str,
) -> Simulator:
    """Loads the simsnn filepath and converts it into a simsnn graph file."""
    # Read output JSON file into dict.
    with open(simsnn_filepath, encoding="utf-8") as json_file:
        some_dict: Dict[str, List] = json.load(json_file)
        json_file.close()

    return simsnn_graph_from_file_to_simulator(
        add_to_raster=True,
        add_to_multimeter=True,
        simsnn_dict=some_dict,
    )


@typechecked
def simsnn_graph_from_file_to_simulator(
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

    for edge in simsnn_dict["synapses"]:
        # synapse = snn_graph.edges[edge]["synapse"]
        net.createSynapse(
            pre=simsnn[edge["ID"][0]],
            post=simsnn[edge["ID"][1]],
            ID=edge,
            w=edge["w"],
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
