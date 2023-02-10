"""Stores the plot data in graph."""
from typing import Dict, List

import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron
from typeguard import typechecked

from snncompare.export_plots.get_graph_colours import get_nx_node_colours
from snncompare.export_plots.get_xy_ticks import store_xy_ticks


@typechecked
def get_neurons_in_graph(
    snn_graph: nx.DiGraph,
    t: int,
) -> List[LIF_neuron]:
    """Returns a list of neurons of the graph at timestep t.

    TODO: support different neuron types.
    """
    neurons: List[LIF_neuron] = []
    for node in snn_graph.nodes():
        nx_lifs: Dict[str, List[LIF_neuron]] = snn_graph.nodes[node]
        neurons.append(nx_lifs["nx_lif"][t])
    return neurons


# pylint: disable=R0912
@typechecked
def get_graph_plot_parameters(
    plotted_graph: nx.DiGraph, snn_graph: nx.DiGraph, t: int
) -> None:
    """Stores the graph plot parameters such as colours, labels and x/y-ticks
    into the networkx graph."""
    # Get neurons in graph.
    lif_neurons: List[LIF_neuron] = get_neurons_in_graph(
        snn_graph,
        t,
    )
    add_nodes_and_edges(
        lif_neurons=lif_neurons,
        plotted_graph=plotted_graph,
        snn_graph=snn_graph,
    )
    store_xy_ticks(lif_neurons=lif_neurons, plotted_graph=plotted_graph)

    store_node_position(plotted_graph=plotted_graph, snn_graph=snn_graph, t=t)
    store_node_labels(lif_neurons=lif_neurons, plotted_graph=plotted_graph)
    store_node_colours_and_opacity(
        plotted_graph=plotted_graph, snn_graph=snn_graph, t=t
    )
    store_edge_colour_and_opacity(plotted_graph=plotted_graph)

    store_edge_labels(plotted_graph=plotted_graph, snn_graph=snn_graph)


# pylint: disable=R0912
@typechecked
def add_nodes_and_edges(
    lif_neurons: List[LIF_neuron],
    plotted_graph: nx.DiGraph,
    snn_graph: nx.DiGraph,
) -> None:
    """Creates/copies the nodes and edges into the plotted graph."""
    # TODO: remove making a duplicate graph.
    for neuron in lif_neurons:
        if "connector" not in neuron.full_name:
            plotted_graph.add_node(neuron.full_name)
    for edge in snn_graph.edges():
        if "connector" not in edge[0] and "connector" not in edge[1]:
            plotted_graph.add_edge(edge[0], edge[1])


# pylint: disable=R0912
@typechecked
def store_node_colours_and_opacity(
    plotted_graph: nx.DiGraph, snn_graph: nx.DiGraph, t: int
) -> None:
    """Creates/copies the nodes and edges into the plotted graph."""

    colour_dict = get_nx_node_colours(G=snn_graph, t=t)
    for node_name, colour in colour_dict.items():
        if "connector" not in node_name:

            # Store colours over time.
            if "temporal_colour" not in plotted_graph.nodes[node_name].keys():
                plotted_graph.nodes[node_name]["temporal_colour"] = []
            if "temporal_opacity" not in plotted_graph.nodes[node_name].keys():
                plotted_graph.nodes[node_name]["temporal_opacity"] = []

            plotted_graph.nodes[node_name]["colour"] = colour
            plotted_graph.nodes[node_name]["temporal_colour"].append(colour)

            if snn_graph.nodes[node_name]["nx_lif"][t].spikes:
                plotted_graph.nodes[node_name]["opacity"] = 0.8
            else:
                plotted_graph.nodes[node_name]["opacity"] = 0.1
            plotted_graph.nodes[node_name]["temporal_opacity"].append(
                plotted_graph.nodes[node_name]["opacity"]
            )


# pylint: disable=R0912
@typechecked
def store_edge_colour_and_opacity(
    plotted_graph: nx.DiGraph,
) -> None:
    """Copies the node_colours and opacity into the edge_colours and
    opacity."""
    for edge in plotted_graph.edges():
        if "connector" not in edge[0] and "connector" not in edge[1]:
            plotted_graph.edges[edge]["colour"] = plotted_graph.nodes[edge[0]][
                "colour"
            ]
            plotted_graph.edges[edge]["opacity"] = plotted_graph.nodes[
                edge[0]
            ]["opacity"]


# pylint: disable=R0912
@typechecked
def store_node_labels(
    lif_neurons: List[LIF_neuron],
    plotted_graph: nx.DiGraph,
) -> None:
    """stores the node labels into the plotted graph."""
    for neuron in lif_neurons:
        if "connector" not in neuron.full_name:
            plotted_graph.nodes[neuron.full_name][
                "label"
            ] = f"V:{neuron.u.get()}/{neuron.vth.get()}"


# pylint: disable=R0912
@typechecked
def store_edge_labels(
    plotted_graph: nx.DiGraph,
    snn_graph: nx.DiGraph,
) -> None:
    """stores the edge labels into the plotted graph."""
    for edge in plotted_graph.edges():
        if "connector" not in edge[0] and "connector" not in edge[1]:
            plotted_graph.edges[edge][
                "label"
            ] = f"W:{snn_graph.edges[edge]['synapse'].weight}"


# pylint: disable=R0912
@typechecked
def store_node_position(
    plotted_graph: nx.DiGraph, snn_graph: nx.DiGraph, t: int
) -> None:
    """stores the node position the plotted graph."""
    for node_name in plotted_graph.nodes():
        if "connector" not in node_name:
            plotted_graph.nodes[node_name]["pos"] = snn_graph.nodes[node_name][
                "nx_lif"
            ][t].pos
