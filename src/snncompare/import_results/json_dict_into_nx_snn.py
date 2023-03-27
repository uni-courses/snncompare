"""Converts json dict back into nx snns for the nx simulator."""
from typing import Dict, Union

import networkx as nx
from networkx.readwrite import json_graph
from snnbackends.networkx.LIF_neuron import Synapse, manually_create_lif_neuron
from snnbackends.simsnn.export import json_to_simsnn
from snnbackends.verify_nx_graphs import verify_results_nx_graphs
from typeguard import typechecked

from snncompare.run_config.Run_config import Run_config


@typechecked
def load_json_graph_to_snn(
    *, json_graphs: Dict, run_config: Run_config
) -> None:
    """Converts the json snn into the desired snn."""
    # Get run config object.

    if run_config.simulator == "nx":
        json_graph_to_nx_snn(json_graphs=json_graphs, run_config=run_config)
    elif run_config.simulator == "simsnn":
        json_graph_to_simsnn_snn(json_graphs=json_graphs)
    else:
        raise NotImplementedError(
            "Error, did not yet implement simsnn to nx_lif converter."
        )


@typechecked
def json_graph_to_nx_snn(*, json_graphs: Dict, run_config: Run_config) -> None:
    """Converts the json snn into the nx snn."""
    for graph_name in json_graphs.keys():
        json_graphs[graph_name] = json_graph.node_link_graph(
            json_graphs[graph_name]
        )
        restore_nx_lif_neuron_and_synapse_objects(
            graph=json_graphs[graph_name]
        )

    # TODO: Verify node and edge attributes are of valid object type.
    verify_results_nx_graphs(
        results_nx_graphs={
            "exp_config": None,
            "run_config": run_config,
            "graphs_dict": json_graphs,
        },
        run_config=run_config,
    )


@typechecked
def json_graph_to_simsnn_snn(*, json_graphs: Dict) -> None:
    """Converts the json snn into the simsnn snn.

    TODO: rename this file and docstring.
    """
    for graph_name in json_graphs.keys():
        if graph_name != "input_graph":
            json_graphs[graph_name] = json_to_simsnn(
                json_simsnn=json_graphs[graph_name]
            )
            # TODO: set completed_stages.
        else:
            json_graphs[graph_name] = json_graph.node_link_graph(
                json_graphs[graph_name]
            )
    # TODO: verify loaded graphs.


@typechecked
def restore_nx_lif_neuron_and_synapse_objects(
    *, graph: Union[nx.Graph, nx.DiGraph]
) -> None:
    """Converts the edge and node attributes Synapse and nx_Lif back into their
    respective objects."""
    for edge in graph.edges:
        if "synapse" in graph.edges[edge].keys():
            graph.edges[edge]["synapse"] = Synapse(
                **graph.edges[edge]["synapse"]
            )
    for node in graph.nodes:
        if "nx_lif" in graph.nodes[node].keys():
            # print(graph.nodes[node]["nx_lif"])
            graph.nodes[node]["nx_lif"] = list(
                map(
                    manually_create_lif_neuron,
                    graph.nodes[node]["nx_lif"],
                )
            )
