"""Stores the plot data in graph."""
from typing import Dict, List

import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron
from typeguard import typechecked

from snncompare.export_plots.get_graph_colours import get_nx_node_colours
from snncompare.export_plots.get_xy_ticks import store_xy_ticks
from snncompare.optional_config.Output_config import Hover_info


@typechecked
def get_neurons_in_graph(
    snn_graph: nx.DiGraph,
    t: int,
) -> List[LIF_neuron]:
    """Returns a list of neurons of the graph at timestep t.

    TODO: support different neuron types.
    """
    neurons: List[LIF_neuron] = []
    for node_name in snn_graph.nodes():
        nx_lifs: Dict[str, List[LIF_neuron]] = snn_graph.nodes[node_name]
        neurons.append(nx_lifs["nx_lif"][t])
    return neurons


# pylint: disable=R0912
@typechecked
def store_plot_params_in_graph(
    hover_info: Hover_info,
    plotted_graph: nx.DiGraph,
    snn_graph: nx.DiGraph,
    t: int,
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
    store_node_labels(
        hover_info=hover_info,
        lif_neurons=lif_neurons,
        plotted_graph=plotted_graph,
        snn_graph=snn_graph,
        t=t,
    )
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
def store_node_labels(
    hover_info: Hover_info,
    lif_neurons: List[LIF_neuron],
    plotted_graph: nx.DiGraph,
    snn_graph: nx.DiGraph,
    t: int,
) -> None:
    """stores the node labels into the plotted graph."""

    # TODO: move into separate function.
    used_node_names: List[str] = []
    for neuron in lif_neurons:
        if "connector" not in neuron.full_name:
            # Assert no duplicate node_names exist.
            if neuron.full_name in used_node_names:
                raise ValueError(
                    f"Error, duplicate node_names:{neuron.full_name} not "
                    + " supported."
                )
            used_node_names.append(neuron.full_name)

    hovertext: Dict[str, str] = {
        node_name: "" for node_name in used_node_names
    }
    for node_name in used_node_names:
        # Initialise the list of node hovertexts, per node.
        if (
            "temporal_node_hovertext"
            not in plotted_graph.nodes[node_name].keys()
        ):
            plotted_graph.nodes[node_name]["temporal_node_hovertext"] = []

        # Add node hovertext data per node
        if hover_info.node_names:
            hovertext[node_name] = hovertext[node_name] + node_name
        if hover_info.neuron_properties:
            hovertext[node_name] = hovertext[
                node_name
            ] + get_desired_neuron_properties(
                snn_graph=snn_graph,
                neuron_properties=hover_info.neuron_properties,
                node_name=node_name,
                t=t,
            )
        if hover_info.incoming_synapses:
            hovertext[node_name] = hovertext[node_name] + get_edges_of_node(
                snn_graph=snn_graph, node_name=node_name, outgoing=False
            )
        if hover_info.outgoing_synapses:
            hovertext[node_name] = hovertext[node_name] + get_edges_of_node(
                snn_graph=snn_graph, node_name=node_name, outgoing=True
            )

        # plotted_graph.nodes[node_name]["label"] = hovertext[node_name]
        plotted_graph.nodes[node_name]["temporal_node_hovertext"].append(
            hovertext[node_name]
        )


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


@typechecked
def get_desired_neuron_properties(
    snn_graph: nx.DiGraph,
    neuron_properties: List[str],
    node_name: str,
    t: int,
) -> str:
    """Returns a list with one string per node."""
    properties: List[str] = ["<br />"]
    for neuron_property_name in neuron_properties:
        neuron_property_obj = getattr(
            snn_graph.nodes[node_name]["nx_lif"][t], neuron_property_name
        )
        neuron_property = getattr(neuron_property_obj, neuron_property_name)

        properties.append(f"{neuron_property_name}:{neuron_property}<br />")
    return "".join(properties)


@typechecked
def get_edges_of_node(
    snn_graph: nx.DiGraph,
    node_name: str,
    outgoing: bool,
) -> str:
    """Returns (the other) nodenames of the edges of a node."""
    node_edges: List[str] = ["<br />"]
    if outgoing:
        node_edges.append("outgoing:<br />")
    else:
        node_edges.append("incoming:<br />")

    for edge in snn_graph.edges():
        if edge[0] == node_name and outgoing:
            node_edges.append(f"{edge[1]}<br /> ")
        elif edge[1] == node_name and not outgoing:
            node_edges.append(f"{edge[0]}<br /> ")

    node_edge_str = "".join(node_edges)

    return node_edge_str


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
