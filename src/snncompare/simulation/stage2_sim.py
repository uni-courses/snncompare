"""Simulates the SNN graphs and returns a deep copy of the graph per
timestep."""


import networkx as nx
from snnbackends.networkx.run_on_networkx import run_snn_on_networkx
from typeguard import typechecked

from ..helper import add_stage_completion_to_graph, get_max_sim_duration


@typechecked
def sim_graphs(
    run_config: dict,
    stage_1_graphs: dict,
) -> None:
    """Simulates the snn graphs and makes a deep copy for each timestep.

    :param stage_1_graphs: dict:
    :param run_config: dict:
    """
    for graph_name, snn_graph in stage_1_graphs.items():
        if graph_name != "input_graph":

            if not isinstance(snn_graph, nx.DiGraph):
                raise Exception(
                    "Error, snn_graph:{graph_name} was not of the"
                    f"expected type, it was of:{type(snn_graph)}"
                )
            # TODO: add lava neurons if run config demands lava.

            if not isinstance(snn_graph, nx.DiGraph):
                raise Exception(
                    "Error, snn_graph:{graph_name} was not of the"
                    "expected type after conversion, it was of type:"
                    f"{type(snn_graph)}"
                )

            stage_1_graphs[graph_name].graph[
                "sim_duration"
            ] = get_max_sim_duration(stage_1_graphs["input_graph"], run_config)
            run_snn_on_networkx(
                run_config,
                snn_graph,
                stage_1_graphs[graph_name].graph["sim_duration"],
            )
        add_stage_completion_to_graph(stage_1_graphs[graph_name], 2)
