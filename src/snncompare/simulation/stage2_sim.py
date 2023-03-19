"""Simulates the SNN graphs and returns a deep copy of the graph per
timestep."""
from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from snnbackends.networkx.run_on_networkx import run_snn_on_networkx
from snnbackends.simsnn.run_on_simsnn import run_snn_on_simsnn
from typeguard import typechecked

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
        if graph_name != "input_graph":
            sim_snn(
                input_graph=stage_1_graphs["input_graph"],
                snn=snn,
                run_config=run_config,
            )
        add_stage_completion_to_graph(snn=snn, stage_index=2)


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
