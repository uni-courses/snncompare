"""Simulates the SNN graphs and returns a deep copy of the graph per
timestep."""
from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from snnbackends.networkx.run_on_networkx import run_snn_on_networkx
from snnbackends.simsnn.run_on_simsnn import run_snn_on_simsnn
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_rand_nrs_and_hash,
)
from snncompare.export_results.output_stage2_snns import (
    get_output_category_and_rad_affected_neuron_hash,
)
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.import_results.load_stage_1_and_2 import load_simsnn_graphs
from snncompare.run_config.Run_config import Run_config

from ..helper import (
    add_stage_completion_to_graph,
    get_max_sim_duration,
    get_rand_synapse_weights,
    get_with_adaptation_bool,
    get_with_radiation_bool,
)


@typechecked
def sim_graphs(
    *,
    run_config: "Run_config",
    stage_1_graphs: Dict,
) -> None:
    """Simulates the snn graphs and makes a deep copy for each timestep.

    :param stage_1_graphs: Dict:
    """

    for graph_name, snn in stage_1_graphs.items():
        # Derive the adaptation setting for this graph.

        if graph_name != "input_graph":
            with_adaptation: bool = get_with_adaptation_bool(
                graph_name=graph_name
            )
            with_radiation: bool = get_with_radiation_bool(
                graph_name=graph_name
            )

            if not graph_exists_already(
                input_graph=stage_1_graphs["input_graph"],
                stage_1_graphs=stage_1_graphs,
                run_config=run_config,
                with_adaptation=with_adaptation,
                with_radiation=with_radiation,
            ):
                print(f"graph_name={graph_name} - simulating.")
                sim_snn(
                    input_graph=stage_1_graphs["input_graph"],
                    snn=snn,
                    run_config=run_config,
                )
                add_stage_completion_to_graph(
                    snn=stage_1_graphs[graph_name], stage_index=2
                )
            else:
                print(f"graph_name={graph_name} - loading.")
                stage_1_graphs[graph_name] = load_simsnn_graphs(
                    run_config=run_config,
                    input_graph=stage_1_graphs["input_graph"],
                    with_adaptation=with_adaptation,
                    with_radiation=with_radiation,
                    stage_index=2,
                )
                get_rand_synapse_weights(
                    input_graph=stage_1_graphs["input_graph"],
                    simsnn_synapses=stage_1_graphs[
                        graph_name
                    ].network.synapses,
                )
        else:
            add_stage_completion_to_graph(
                snn=stage_1_graphs[graph_name], stage_index=2
            )


@typechecked
def graph_exists_already(
    *,
    input_graph: nx.Graph,
    stage_1_graphs: Dict,
    run_config: "Run_config",
    with_adaptation: bool,
    with_radiation: bool,
) -> bool:
    """Returns True if a graph already exists.

    False otherwise.
    """

    _, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)
    (
        output_category,
        rad_affected_neurons_hash,
    ) = get_output_category_and_rad_affected_neuron_hash(
        graphs_dict=stage_1_graphs,
        run_config=run_config,
        with_adaptation=with_adaptation,
        with_radiation=with_radiation,
        stage_index=1,
    )

    simsnn_exists, _ = simsnn_files_exists_and_get_path(
        output_category=output_category,
        input_graph=stage_1_graphs["input_graph"],
        run_config=run_config,
        with_adaptation=with_adaptation,
        stage_index=2,
        rad_affected_neurons_hash=rad_affected_neurons_hash,
        rand_nrs_hash=rand_nrs_hash,
    )
    return simsnn_exists


@typechecked
def sim_snn(
    *,
    input_graph: nx.Graph,
    snn: Union[nx.DiGraph, Simulator],
    run_config: "Run_config",
) -> None:
    """Simulates the snn graphs and makes a deep copy for each timestep.

    :param stage_1_graphs: Dict:
    """
    sim_duration: int
    if run_config.simulator == "nx":
        sim_duration = get_max_sim_duration(
            input_graph=input_graph,
            run_config=run_config,
        )
        if not isinstance(snn, nx.DiGraph):
            raise TypeError(
                "Error, snn_graph:{graph_name} was not of the"
                "expected type after conversion, it was of type:"
                f"{type(snn)}"
            )

        run_snn_on_networkx(
            run_config=run_config,
            snn_graph=snn,
            sim_duration=sim_duration,
        )
    elif run_config.simulator == "simsnn":
        sim_duration = get_max_sim_duration(
            input_graph=input_graph,
            run_config=run_config,
        )
        if not isinstance(snn, Simulator):
            raise TypeError(
                "Error, snn should be of type Simulator, it was:"
                + f"{type(snn)}"
            )
        run_snn_on_simsnn(
            run_config=run_config,
            snn=snn,
            sim_duration=sim_duration,
        )

    else:
        # TODO: add lava neurons if run config demands lava.
        raise NotImplementedError(
            "Error, did not yet implement simsnn to nx_lif converter."
        )
