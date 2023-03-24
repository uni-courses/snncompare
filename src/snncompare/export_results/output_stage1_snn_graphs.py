"""Exports the following structure to an output file for simsnn:

/stage_1/run_config_name.json with content of stage1 algo dict.
    snn_algo_graph: nodes, lif values and edges.
    adapted_snn_algo_graph: nodes, lif values and edges.
    radiation type, died neurons list without adaptation.
    radiation type, Died neurons list with adaptation.
"""
from pprint import pprint
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
from simsnn.core.connections import Synapse
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.export_json_results import write_to_json
from snncompare.export_results.output_stage1_configs_and_input_graph import (
    prepare_target_file_output,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def output_stage_1_snns(
    *,
    run_config: Run_config,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    with_adaptation: bool,
) -> None:
    """Exports results dict to a json file."""
    simsnn_exists, simsnn_filepath = simsnn_files_exists_and_get_path(
        input_graph=graphs_dict["input_graph"],
        run_config=run_config,
        with_adaptation=with_adaptation,
    )
    if not simsnn_exists:
        if with_adaptation:
            # Export default snn.
            output_snn_graph(
                output_filepath=simsnn_filepath,
                snn_graph=graphs_dict["snn_algo_graph"],
            )
        else:
            # Export adapted snn.
            output_snn_graph(
                output_filepath=simsnn_filepath,
                snn_graph=graphs_dict["adapted_snn_graph"],
            )


@typechecked
def output_snn_graph(
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
            simsnn_neurons=snn_graph.network.nodes
        )
        json_simsnn_synapses: Dict = simsnn_synapses_to_json(
            simsnn_synapses=snn_graph.network.synapses
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


@typechecked
def get_single_element(*, some_list: List) -> Union[str, int]:
    """Asserts a list has only one element and returns that element."""
    assert_has_one_element(some_list=some_list)
    return some_list[0]


@typechecked
def assert_has_one_element(*, some_list: List) -> None:
    """Asserts a list contains only 1 element."""
    if len(some_list) != 1:
        raise ValueError(
            "Error the number of algorithms in a single run_config was not 1:"
            + f"{some_list}"
        )


@typechecked
def get_algorithm_description(*, run_config: Run_config) -> Tuple[str, int]:
    """Returns the algorithm name and value as a single string."""
    algorithm_name: str = get_single_element(
        some_list=list(run_config.algorithm.keys())
    )

    algorithm_parameter: int = get_single_element(
        some_list=list(run_config.algorithm[algorithm_name].values())
    )
    return algorithm_name, algorithm_parameter


@typechecked
def get_adaptation_description(*, run_config: Run_config) -> Tuple[str, int]:
    """Returns the adaptation name and value as a single string."""
    adaptation_name: str = get_single_element(
        some_list=list(run_config.adaptation.keys())
    )

    adaptation_parameter: int = run_config.adaptation[adaptation_name]

    if adaptation_parameter == 0:
        pprint(run_config.__dict__)
        raise ValueError(
            "Error, redundancy=0 is a duplicate of original graph."
        )
    return adaptation_name, adaptation_parameter


@typechecked
def simsnn_files_exists_and_get_path(
    *,
    run_config: Run_config,
    input_graph: nx.Graph,
    with_adaptation: bool,
) -> Tuple[bool, str]:
    """Returns two tuples which contain: graph file exists, and the graph
    filepath.

    First tuple for the unadapted snn, the second tuple for the adapted
    tuple.
    """
    algorithm_name, algorithm_parameter = get_algorithm_description(
        run_config=run_config
    )

    adaptation_name, adaptation_parameter = get_adaptation_description(
        run_config=run_config
    )
    if algorithm_name == "MDSA":
        if with_adaptation:
            # Import adapted snn.
            output_dir: str = (
                f"results/{algorithm_name}_{algorithm_parameter}"
                + f"/{adaptation_name}_{adaptation_parameter}/"
            )
            (
                snn_algo_graph_exists,
                snn_algo_graph_filepath,
            ) = prepare_target_file_output(
                output_dir=output_dir, some_graph=input_graph
            )
        else:
            # Import default snn.
            output_dir = (
                f"results/{algorithm_name}_{algorithm_parameter}"
                + "/no_adaptation/"
            )
            (
                snn_algo_graph_exists,
                snn_algo_graph_filepath,
            ) = prepare_target_file_output(
                output_dir=output_dir, some_graph=input_graph
            )
        return (snn_algo_graph_exists, snn_algo_graph_filepath)

    raise NotImplementedError(f"Error:{algorithm_name} is not yet supported.")
