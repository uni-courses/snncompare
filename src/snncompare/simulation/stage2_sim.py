"""Simulates the SNN graphs and returns a deep copy of the graph per
timestep."""
from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from snnbackends.networkx.run_on_networkx import run_snn_on_networkx
from snnbackends.simsnn.run_on_simsnn import run_snn_on_simsnn
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    Radiation_data,
    get_radiation_names_filepath_and_exists,
    get_rand_nrs_and_hash,
)
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.run_config.Run_config import Run_config

from ..helper import add_stage_completion_to_graph, get_max_sim_duration


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
        if graph_name in ["snn_algo_graph", "adapted_snn_graph"]:
            with_adaptation: bool = True
        else:
            with_adaptation = False

        if graph_name != "input_graph":
            if not graph_exists_already(
                input_graph=stage_1_graphs["input_graph"],
                stage_1_graphs=stage_1_graphs,
                run_config=run_config,
                with_adaptation=with_adaptation,
            ):
                print(f"graph_name={graph_name}")
                sim_snn(
                    input_graph=stage_1_graphs["input_graph"],
                    snn=snn,
                    run_config=run_config,
                )
            else:
                raise NotImplementedError(
                    f"Error, need to load graph from file!:{graph_name}"
                )
        add_stage_completion_to_graph(snn=snn, stage_index=2)


@typechecked
def graph_exists_already(
    *,
    input_graph: nx.Graph,
    stage_1_graphs: Dict,
    run_config: "Run_config",
    with_adaptation: bool,
) -> bool:
    """Returns True if a graph already exists.

    False otherwise.
    """
    _, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)
    radiation_data: Radiation_data = get_radiation_names_filepath_and_exists(
        graphs_dict=stage_1_graphs,
        run_config=run_config,
        stage_index=1,
        with_adaptation=with_adaptation,
    )

    simsnn_exists, _ = simsnn_files_exists_and_get_path(
        output_category="snns",
        input_graph=stage_1_graphs["input_graph"],
        run_config=run_config,
        with_adaptation=with_adaptation,
        stage_index=2,
        rad_affected_neurons_hash=radiation_data.rad_affected_neurons_hash,
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
