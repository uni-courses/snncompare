"""Generates interactive view of graph."""

from typing import Dict, List, Tuple, Union

import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron
from typeguard import typechecked

from snncompare.export_plots.export_with_dash import create_svg_with_dash
from snncompare.export_plots.get_graph_colours import (
    create_edge_colours,
    set_nx_node_colours,
)
from snncompare.export_plots.Plot_config import (
    Plot_config,
    get_default_plot_config,
)
from snncompare.run_config.Run_config import Run_config


# Determine which graph(s) the user would like to see.
# If no specific preference specified, show all 4.
# pylint: disable=R0903
@typechecked
def create_svg_plot(
    filepath: str,
    graphs: Dict[str, Union[nx.Graph, nx.DiGraph]],
    run_config: Run_config,
) -> None:
    """Creates the svg plots."""
    plot_config: Plot_config = get_default_plot_config()

    # pylint: disable=R1702
    for graph_name, snn_graph in graphs.items():
        if graph_name != "input_graph":
            sim_duration = snn_graph.graph["sim_duration"]

            # Get the layout of the graph.
            # Get the x-ticks.
            # Get the y-ticks.
            # Compute image width.
            # Compute image height.

            for alg_name, _ in run_config.algorithm.items():
                if alg_name == "MDSA":
                    for (
                        adaptation_name,
                        _,
                    ) in run_config.adaptation.items():
                        sub_graphs: List[nx.DiGraph] = []
                        for t in range(
                            0,
                            sim_duration,
                        ):
                            filename: str = f"{graph_name}_{filepath}_{t}"
                            if adaptation_name == "redundancy":
                                sub_graphs.append(
                                    get_graph_plot_parameters(
                                        snn_graph=snn_graph, t=t
                                    )
                                )
                            else:
                                raise Exception(
                                    f"Error, adaptation:{adaptation_name} not "
                                    + "yet supported."
                                )
                        create_svg_with_dash(
                            filename=filename,
                            graphs=sub_graphs,
                            plot_config=plot_config,
                        )
                else:
                    raise Exception(
                        f"Error, algorithm:{alg_name} not yet supported."
                    )
                # Load the colour sets for the nodes from the results_dict
                # Load the colour sets for the edges from the results_dict.

                # Load which neurons spike in which time step.
                # Load which outgoing edges spike in that timestep.
                # Create the node colour sets.
                # Create the edge colour sets.


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


@typechecked
def get_xy_tick_labels(
    lif_neurons: List[LIF_neuron],
) -> Tuple[Dict[float, str], Dict[float, str]]:
    """Computes the x-tick position and labels for the plot.

    TODO: filter which neuron types you want on x-axis.
    TODO: filter which neuron types you want on y-axis.
    TODO: make algorithm dependent.
    """
    x_tick_labels: Dict[float, str] = {}
    y_tick_labels: Dict[float, str] = {}
    sorted_neurons: Dict[str, List[LIF_neuron]] = get_sorted_neurons(
        lif_neurons=lif_neurons
    )
    for neuron_type, neurons in sorted_neurons.items():
        x = min(list(map(lambda a: a.pos[0], neurons)))
        x_tick_labels[x] = neuron_type
        y = min(list(map(lambda a: a.pos[1], neurons)))
        y_tick_labels[y] = neuron_type
    return x_tick_labels, y_tick_labels


@typechecked
def get_sorted_neurons(
    lif_neurons: List[LIF_neuron],
) -> Dict[str, List[LIF_neuron]]:
    """Sorts the LIF_neurons on neuron name."""
    sorted_neurons: Dict[str, List[LIF_neuron]] = {}
    for lif_neuron in lif_neurons:

        for neuron_name in [
            "spike_once",
            "rand",
            "degree_receiver",
            "selector",
            "counter",
            "next_round",
            "connecting_node",
            "terminator_node",
        ]:
            put_neuron_into_sorted_dict(
                lif_neuron=lif_neuron,
                neuron_name=neuron_name,
                sorted_neurons=sorted_neurons,
            )
    return sorted_neurons


@typechecked
def put_neuron_into_sorted_dict(
    lif_neuron: LIF_neuron,
    neuron_name: str,
    sorted_neurons: Dict[str, List[LIF_neuron]],
) -> None:
    """Puts a neuron in its category/key/neuron type if it ."""
    if lif_neuron.name == neuron_name:
        if neuron_name in sorted_neurons.keys():
            sorted_neurons[neuron_name].append(lif_neuron)
        else:
            sorted_neurons[neuron_name] = [lif_neuron]


def get_graph_plot_parameters(snn_graph: nx.DiGraph, t: int) -> nx.DiGraph:
    """Stores the graph plot parameters such as colours, labels and x/y-ticks
    into the networkx graph."""
    # Get neurons in graph.
    lif_neurons: List[LIF_neuron] = get_neurons_in_graph(
        snn_graph,
        t,
    )

    # TODO: remove making a duplicate graph.
    G = nx.DiGraph()
    for neuron in lif_neurons:
        G.add_node(neuron.full_name)
        G.nodes[neuron.full_name]["nx_lif"] = neuron
    for edge in snn_graph.edges():
        G.add_edge(edge[0], edge[1])
        G.edges[edge]["synapse"] = snn_graph.edges[edge]["synapse"]

    # TODO: compute x-tick labels.
    # TODO: compute y-tick labels.
    x_ticks, y_ticks = get_xy_tick_labels(lif_neurons)
    G.graph["x_tics"] = x_ticks
    G.graph["y_tics"] = y_ticks

    # TODO: compute node colour.
    (
        colour_dict,
        _,
        spiking_edges,
    ) = set_nx_node_colours(G=snn_graph, t=t)
    for node_name, colour in colour_dict.items():
        G.nodes[node_name]["colour"] = colour
    # print(f"color_map={color_map}")
    # print(f"spiking_edges={spiking_edges}")

    # Compute edge colour.
    edge_color_dict = create_edge_colours(
        G=snn_graph, spiking_edges=spiking_edges
    )
    for edge, colour in edge_color_dict.items():
        G.edges[edge]["colour"] = colour

    # Compute node labels.
    for neuron in lif_neurons:
        G.nodes[neuron.full_name][
            "label"
        ] = f"V:{neuron.u.get()}/{neuron.vth.get()}"

    # TODO: compute node position.
    for node_name in G.nodes():
        G.nodes[node_name]["pos"] = snn_graph.nodes[node_name]["nx_lif"][t].pos

    # Compute edge labels.
    for edge in snn_graph.edges():
        G.edges[edge]["label"] = f"W:{snn_graph.edges[edge]['synapse'].weight}"
    return G
