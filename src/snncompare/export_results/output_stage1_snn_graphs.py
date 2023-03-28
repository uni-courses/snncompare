"""Exports the following structure to an output file for simsnn:

/stage_1/run_config_name.json with content of stage1 algo dict.
    snn_algo_graph: nodes, lif values and edges.
    adapted_snn_algo_graph: nodes, lif values and edges.
    radiation type, died neurons list without adaptation.
    radiation type, Died neurons list with adaptation.
"""
import copy
from typing import Dict, List, Union

import networkx as nx
import numpy as np
from simsnn.core.connections import Synapse
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.export_json_results import write_to_json
from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_rand_nrs_and_hash,
)
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.run_config.Run_config import Run_config


@typechecked
def output_stage_1_snns(
    *,
    run_config: Run_config,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    with_adaptation: bool,
) -> None:
    """Exports results dict to a json file."""
    _, rand_nrs_hash = get_rand_nrs_and_hash(
        input_graph=graphs_dict["input_graph"]
    )
    simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
        output_category="snns",
        input_graph=graphs_dict["input_graph"],
        run_config=run_config,
        with_adaptation=with_adaptation,
        stage_index=1,
        rand_nrs_hash=rand_nrs_hash,
    )
    if not simsnn_exists:
        if with_adaptation:
            # Export default snn.
            output_snn_graph_stage_1(
                output_filepath=simsnn_filepath,
                snn_graph=graphs_dict["snn_algo_graph"],
            )
        else:
            # Export adapted snn.
            output_snn_graph_stage_1(
                output_filepath=simsnn_filepath,
                snn_graph=graphs_dict["adapted_snn_graph"],
            )


@typechecked
def output_snn_graph_stage_1(
    *,
    output_filepath: str,
    snn_graph: Union[nx.DiGraph, Simulator],
) -> None:
    """Outputs the simsnn neuron properties, synapse properties. Explicit graph
    attributes are not stored, as they should be a function of the run config
    with which they are called, not a part of the snn graph. The graphs are
    stored in the folder:

    <algorithm>/<algorithm config>/<isomorphic graph hash>.json
    """
    # TODO: change this into an object with: name and a list of parameters
    # instead.

    if isinstance(snn_graph, Simulator):
        json_simsns_neurons: Dict = simsnn_nodes_to_json(
            simsnn_neurons=copy.deepcopy(snn_graph.network.nodes)
        )
        json_simsnn_synapses: Dict = simsnn_synapses_to_json(
            simsnn_synapses=copy.deepcopy(snn_graph.network.synapses)
        )

        write_to_json(
            output_filepath=output_filepath,
            some_dict={
                "neurons": json_simsns_neurons,
                "synapses": json_simsnn_synapses,
            },
        )

    else:
        raise NotImplementedError("TODO: convert into simsnn and export.")


@typechecked
def simsnn_nodes_to_json(*, simsnn_neurons: List[LIF]) -> List[Dict]:
    """Converts list of simsnn LIF neurons into dict that can be exported to
    json."""
    json_simsns_neurons: List[Dict] = []
    for neuron in simsnn_neurons:
        output_neuron: Dict = vars(neuron)
        for key in ["I", "V", "out"]:  # Remove non-static neuron properties.
            output_neuron.pop(key)
        json_simsns_neurons.append(vars(neuron))
    return json_simsns_neurons


@typechecked
def simsnn_synapses_to_json(*, simsnn_synapses: List[Synapse]) -> List[Dict]:
    """Converts list of simsnn synapses into dict that can be exported to
    json."""
    json_simsnn_synapses: List[Dict] = []
    for synapse in simsnn_synapses:
        json_synapse: Dict = vars(synapse)
        json_synapse.pop("pre")
        json_synapse.pop("post")
        if json_synapse["out_pre"] != np.array([0.0]):
            raise ValueError(f"out_pre is not zero:{json_synapse}")
        json_synapse.pop("out_pre")
        json_simsnn_synapses.append(json_synapse)
    return json_simsnn_synapses
