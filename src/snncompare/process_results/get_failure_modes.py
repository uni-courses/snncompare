"""Computes what the failure modes were, and then stores this data in the
graphs."""
from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked


@typechecked
def add_failure_modes_to_graph(
    *,
    snn_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> None:
    """Outputs the stage 4 snn results to json."""

    for graph_name, graph in snn_graphs.items():
        if graph_name != "input_graph":
            failure_modes = {
                "excitation_failures": {
                    0: ["first_name", "second_name"],
                    1: ["alt_name", "second_alt_name"],
                },
                "inhibitory_failures": {
                    0: ["in_first_name", "in_second_name"],
                    1: ["in_alt_name", "in_second_alt_name"],
                },
            }
            graph.network.graph.graph["failure_modes"] = failure_modes
