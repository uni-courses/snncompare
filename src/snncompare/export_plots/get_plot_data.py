"""Returns the updated plot data.

TODO: rename and restructure this function along with:
helper_network_structure.
"""

from pprint import pprint
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
from matplotlib import pyplot as plt
from typeguard import typechecked

from snncompare.export_plots.Plot_to_tex import Plot_to_tex


# pylint: disable=R0913
@typechecked
def plot_coordinated_graph(
    extensions: List[str],
    desired_properties: Union[List, None],
    G: Union[nx.Graph, nx.DiGraph],
    t: int,
    filename: str = "no_filename",
    show: bool = False,
    title: str = None,
) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    param iteration: The initialisation iteration that is used.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param desired_properties:  (Default value = [])
    :param show:  (Default value = False)
    :param filename:  (Default value = "no_filename")

    """
    if desired_properties is None:
        desired_properties = []

    color_map, spiking_edges = set_nx_node_colours(G, t)

    edge_color_map = set_edge_colours(G, spiking_edges)

    set_node_positions(G, t)

    # Width=edge width.
    nx.draw(
        G,
        nx.get_node_attributes(G, "pos"),
        with_labels=True,
        node_size=160,
        font_size=6,
        width=0.2,
        node_color=color_map,
        edge_color=edge_color_map,
        # **options,
    )
    # TODO: change to name?
    node_labels_dict = nx.get_node_attributes(G, "")

    # pylint: disable=W0108
    node_labels_list = list(map(lambda x: str(x), G.nodes))

    pos = {
        node: (x, y)
        for (node, (x, y)) in nx.get_node_attributes(G, "pos").items()
    }
    print("")
    print("")
    pprint(pos)
    print("")
    get_edge_labels(node_labels_dict, G, pos)

    plt.axis("off")
    axis = plt.gca()
    axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    axis.set_ylim([1.2 * y for y in axis.get_ylim()])

    # Set a title in the image.
    if title is not None:
        plt.suptitle(title, fontsize=14)

    add_neuron_properties_to_plot(
        axis, desired_properties, G, node_labels_list, pos, t
    )

    # f = plt.figure()
    # f.set_figwidth(10)
    # f.set_figheight(10)
    # plt.subplots_adjust(left=0.0, right=4.0, bottom=0.0, top=4.0)
    if show:
        plt.show()

    plot_export = Plot_to_tex()
    plot_export.export_plot(plt, filename, extensions=extensions)
    # plt.savefig()
    plt.clf()
    plt.close()


@typechecked
def get_edge_labels(
    node_labels_dict: Dict, snn_graph: nx.DiGraph, pos: Any
) -> None:
    """Sets the edge labels.

    pos is node position.
    """
    edge_labels: Dict = {}

    # Get the synaptic weights per node.
    for edge in snn_graph.edges:
        edgeweight = snn_graph.edges[edge]["synapse"].weight

        # Create dictionary with synaptic weights.
        edge_labels[edge] = edgeweight

    nx.draw_networkx_labels(snn_graph, pos, labels=node_labels_dict)

    # TODO: Set edge weight positions per edge to non-overlapping positions.
    # TODO: Make edge weight "hitbox" transparent.
    nx.draw_networkx_edge_labels(
        snn_graph, pos, edge_labels, font_size=5, label_pos=0.2
    )


@typechecked
def set_node_positions(snn_graph: nx.DiGraph, t: int) -> None:
    """Sets the positions of the nodes of the snn graph."""
    # TODO: include backend check.
    for nodename in snn_graph.nodes:
        snn_graph.nodes[nodename]["pos"] = snn_graph.nodes[nodename]["nx_lif"][
            t
        ].pos


@typechecked
def get_annotation_text(
    desired_properties: List[str], G: nx.Graph, nodename: str, t: int
) -> str:
    """Returns a string with the annotation text.

    :param desired_properties:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param nodename: Node of the name of a networkx graph.
    """
    annotation = ""
    if "bias" in desired_properties:
        annotation = (
            annotation + f'bias={G.nodes[nodename]["nx_lif"][t].bias.get()}\n'
        )
    if "du" in desired_properties:
        annotation = (
            annotation + f'du={G.nodes[nodename]["nx_lif"][t].du.get()}\n'
        )
    if "dv" in desired_properties:
        annotation = (
            annotation + f'dv={G.nodes[nodename]["nx_lif"][t].dv.get()}\n'
        )
    if "u" in desired_properties:
        annotation = (
            annotation + f'u={G.nodes[nodename]["nx_lif"][t].u.get()}\n'
        )
    if "v" in desired_properties:
        annotation = (
            annotation + f'v={G.nodes[nodename]["nx_lif"][t].v.get()}\n'
        )
    if "vth" in desired_properties:
        annotation = (
            annotation + f'vth={G.nodes[nodename]["nx_lif"][t].vth.get()}\n'
        )
    if "a_in_next" in desired_properties:
        annotation = (
            annotation
            + f'a_in_next={G.nodes[nodename]["nx_lif"][t].a_in_next}\n'
        )

    return annotation


# pylint: disable=R0913
@typechecked
def add_neuron_properties_to_plot(
    axis: Any,
    desired_properties: List,
    G: nx.DiGraph,
    nodenames: List[str],
    pos: Any,
    t: int,
) -> None:
    """Adds a text (annotation) to each neuron with the desired neuron
    properties.

    :param axis:
    :param desired_properties:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param nodenames:
    :param pos:
    """
    for nodename in nodenames:

        # Shift the x-coordinates of the redundant neurons to right for
        # readability.
        if nodename[:3] == "red":
            shift_right = 0.15
        else:
            shift_right = 0

        annotation_text = get_annotation_text(
            desired_properties, G, nodename, t
        )
        # Include text in plot.
        axis.text(
            pos[nodename][0] + shift_right,
            pos[nodename][1],
            annotation_text,
            transform=axis.transData,
            fontsize=4,
        )


@typechecked
def set_nx_node_colours(G: nx.DiGraph, t: int) -> Tuple[List, List]:
    """Returns a list of node colours in order of G.nodes.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    color_map = []
    spiking_edges = []

    colour_dict = {}
    for node_name in G.nodes:
        # print(f'G.nodes[node_name]={G.nodes[node_name]}')
        if "nx_lif" in G.nodes[node_name].keys():
            if "rad_death" in G.nodes[node_name].keys():
                if G.nodes[node_name]["rad_death"]:
                    colour_dict[node_name] = "red"
                    if G.nodes[node_name]["nx_lif"][t].spikes:
                        raise Exception("Dead neuron can't spike.")
            if G.nodes[node_name]["nx_lif"][t].spikes:
                colour_dict[node_name] = "green"
                for neighbour in nx.all_neighbors(G, node_name):
                    spiking_edges.append((node_name, neighbour))
            if node_name not in colour_dict:
                colour_dict[node_name] = "yellow"
        else:
            colour_dict[node_name] = "black"
    for node_name in G.nodes:
        color_map.append(colour_dict[node_name])
    return color_map, spiking_edges


@typechecked
def set_edge_colours(G: nx.DiGraph, spiking_edges: List) -> List:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param spiking_edges:

    """
    edge_color_map = []
    for edge in G.edges:

        if edge in spiking_edges:
            edge_color_map.append("green")
        else:
            edge_color_map.append("black")
    return edge_color_map


@typechecked
def get_labels(G: nx.DiGraph, current: bool = True) -> Dict[str, Any]:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param current:  (Default value = True)

    """
    node_labels = {}
    reset_labels = False
    if current:
        for node_name in G.nodes:
            if node_name != "connecting_node":
                # print u.
                if not G.nodes[node_name]["neuron"] is None:
                    node_labels[node_name] = G.nodes[node_name][
                        "neuron"
                    ].u.get()[0]
                else:
                    reset_labels = True
            else:
                node_labels[node_name] = "0"
    else:
        node_labels = nx.get_node_attributes(G, "")

    # If neurons were not stored in run, they are None, then get default
    # labels.
    if reset_labels:
        node_labels = nx.get_node_attributes(G, "")
    return node_labels
