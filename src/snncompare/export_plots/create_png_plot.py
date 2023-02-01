"""Returns the updated plot data.

TODO: rename and restructure this function along with:
helper_network_structure.
"""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from snnalgorithms.zoom_in_images import copy_region_of_img
from typeguard import typechecked

from snncompare.export_plots.plot_graphs import (
    create_target_dir_if_not_exists,
    export_plot,
)


# pylint: disable=R0913
# pylint: disable=R0914
@typechecked
def plot_coordinated_graph(
    *,
    extensions: List[str],
    desired_properties: Union[List, None],
    G: Union[nx.Graph, nx.DiGraph],
    height: float,
    t: int,
    width: float,
    filename: str = "no_filename",
    show: Optional[bool] = False,
    title: Optional[str] = None,
    zoom: Optional[bool] = False,
) -> None:
    """Some documentation.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param iteration: The initialisation iteration that is used.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param desired_properties:  (Default value = [])
    :param show:  (Default value = False)
    :param filename:  (Default value = "no_filename")
    """
    node_size = 2500
    label_fontsize = 4
    node_fontsize = 4
    node_props_fontsize = 3

    if desired_properties is None:
        desired_properties = []

    color_map, spiking_edges = set_nx_node_colours(G=G, t=t)
    edge_color_map = set_edge_colours(G=G, spiking_edges=spiking_edges)
    set_node_positions(snn_graph=G, t=t)

    # Width=edge width.
    width, height = get_width_and_height(snn_graph=G, t=t)
    # TODO: limit to max filesize
    plt.figure(3, figsize=(width / 20, height / 20), dpi=100)
    nx.draw(
        G,
        nx.get_node_attributes(G, "pos"),
        with_labels=True,
        node_size=node_size,
        font_size=node_fontsize,
        width=0.2,
        node_color=color_map,
        edge_color=edge_color_map,
        # **options,
    )
    # TODO: change to name?
    node_labels_dict = nx.get_node_attributes(G, "")

    # pylint: disable=W0108
    node_labels_list = list(map(lambda x: str(x), G.nodes))

    node_pos = {
        node: (x, y)
        for (node, (x, y)) in nx.get_node_attributes(G, "pos").items()
    }

    get_edge_labels(
        label_fontsize=label_fontsize,
        node_labels_dict=node_labels_dict,
        snn_graph=G,
        pos=node_pos,
    )

    # plt.axis("off")
    plt.axis("on")
    axis = plt.gca()

    # Specify some padding size.
    # (larger width and height need less percentual padding).
    # if width==2.5: pad 1.2 for xlim
    # if width>=10: pad 1 for xlim
    pad_width = 1 + max((1 - width / 10) * 0.2, 0)
    axis.set_xlim([pad_width * x for x in axis.get_xlim()])
    axis.set_ylim([1.0 * y for y in axis.get_ylim()])

    # Set a title in the image.
    if title is not None:
        plt.suptitle(title, fontsize=14)

    add_neuron_properties_to_plot(
        axis=axis,
        desired_properties=desired_properties,
        G=G,
        node_props_fontsize=node_props_fontsize,
        node_names=node_labels_list,
        pos=node_pos,
        t=t,
    )

    if show:
        plt.show()

    export_plot(some_plt=plt, filename=filename, extensions=extensions)

    if zoom:
        create_target_dir_if_not_exists(some_path="latex/Images/graphs/zoom")
        if "png" in extensions:
            copy_region_of_img(
                src_path="latex/Images/graphs/" + filename + ".png",
                dst_dir="latex/Images/graphs/zoom",
                x_coords=(0.0, 1.0),
                y_coords=(0.3, 0.6),
            )

    # plt.savefig()
    plt.clf()
    plt.close()


@typechecked
def get_edge_labels(
    *,
    label_fontsize: int,
    node_labels_dict: Dict,
    snn_graph: nx.DiGraph,
    pos: Dict[str, Tuple[float, float]],
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
        snn_graph, pos, edge_labels, font_size=label_fontsize, label_pos=0.2
    )


@typechecked
def set_node_positions(*, snn_graph: nx.DiGraph, t: int) -> None:
    """Sets the positions of the nodes of the snn graph."""
    # TODO: include backend check.
    for node_name in snn_graph.nodes:
        snn_graph.nodes[node_name]["pos"] = snn_graph.nodes[node_name][
            "nx_lif"
        ][t].pos


@typechecked
def get_annotation_text(
    *, desired_properties: List[str], G: nx.Graph, node_name: str, t: int
) -> str:
    """Returns a string with the annotation text.

    :param desired_properties:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param node_name: Node of the name of a networkx graph.
    """
    if node_name[:4] == "r_{red_level}_":
        return ""
    annotation = ""
    if "bias" in desired_properties:
        annotation = (
            annotation + f'bias={G.nodes[node_name]["nx_lif"][t].bias.get()}\n'
        )
    if "du" in desired_properties:
        annotation = (
            annotation + f'du={G.nodes[node_name]["nx_lif"][t].du.get()}\n'
        )
    if "dv" in desired_properties:
        annotation = (
            annotation + f'dv={G.nodes[node_name]["nx_lif"][t].dv.get()}\n'
        )
    if "u" in desired_properties:
        annotation = (
            annotation + f'u={G.nodes[node_name]["nx_lif"][t].u.get()}\n'
        )
    if "v" in desired_properties:
        annotation = (
            annotation + f'v={G.nodes[node_name]["nx_lif"][t].v.get()}\n'
        )
    if "vth" in desired_properties:
        annotation = (
            annotation + f'vth={G.nodes[node_name]["nx_lif"][t].vth.get()}\n'
        )
    if "a_in_next" in desired_properties:
        annotation = (
            annotation
            + f'a_in_next={G.nodes[node_name]["nx_lif"][t].a_in_next}\n'
        )

    return annotation


# pylint: disable=R0913
@typechecked
def add_neuron_properties_to_plot(
    *,
    axis: matplotlib.axes._axes.Axes,
    desired_properties: List,
    G: nx.DiGraph,
    node_props_fontsize: int,
    node_names: List[str],
    pos: Dict[str, Tuple[float, float]],
    t: int,
) -> None:
    """Adds a text (annotation) to each neuron with the desired neuron
    properties.

    :param axis:
    :param desired_properties:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param node_names:
    :param pos:
    """
    for node_name in node_names:

        # Shift the x-coordinates of the redundant neurons to right for
        # readability.
        if node_name[:3] == "red":
            shift_right = 0.15
        else:
            shift_right = 0

        annotation_text = get_annotation_text(
            desired_properties=desired_properties,
            G=G,
            node_name=node_name,
            t=t,
        )

        # Include text in plot.
        axis.text(
            pos[node_name][0] + shift_right,
            pos[node_name][1],
            annotation_text,
            transform=axis.transData,
            fontsize=node_props_fontsize,
        )


@typechecked
def set_nx_node_colours(*, G: nx.DiGraph, t: int) -> Tuple[List, List]:
    """Returns a list of node colours in order of G.nodes."""
    color_map = []
    spiking_edges = []

    colour_dict = {}
    for node_name in G.nodes:
        # print(f'G.nodes[node_name]={G.nodes[node_name]}')
        if "nx_lif" in G.nodes[node_name].keys():
            if "rad_death" in G.nodes[node_name].keys():
                if G.nodes[node_name]["rad_death"]:
                    # colour_dict[node_name] = ["red",0.5]
                    colour_dict[node_name] = (1, 0, 0, 0.3)
                    if G.nodes[node_name]["nx_lif"][t].spikes:
                        raise Exception("Dead neuron can't spike.")
            if G.nodes[node_name]["nx_lif"][t].spikes:
                # colour_dict[node_name] = ["green",0.5]
                colour_dict[node_name] = (0, 1, 0, 0.5)
                for neighbour in nx.all_neighbors(G, node_name):
                    spiking_edges.append((node_name, neighbour))
            if node_name not in colour_dict:
                set_node_colours_with_redundancy(
                    colour_dict=colour_dict, node_name=node_name
                )
        else:
            colour_dict[node_name] = (0, 0, 0, 1)
    for node_name in G.nodes:
        color_map.append(colour_dict[node_name])
    return color_map, spiking_edges


@typechecked
def set_node_colours_with_redundancy(
    *, colour_dict: Dict, node_name: str
) -> None:
    """Sets the colour of the redundant node different than the original
    node."""
    if node_name[:4] == "r_{red_level}_":
        # colour_dict[node_name] = ["olive",0.5]
        colour_dict[node_name] = (1, 0.98, 0, 0.5)
    else:
        # colour_dict[node_name] = ["yellow",1]
        colour_dict[node_name] = (1, 1, 0, 0.5)


@typechecked
def set_edge_colours(*, G: nx.DiGraph, spiking_edges: List) -> List:
    """Some documentation.

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
def get_labels(*, G: nx.DiGraph, current: bool = True) -> Dict:
    """Some documentation.

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


@typechecked
def get_width_and_height(*, snn_graph: nx.DiGraph, t: int) -> List[float]:
    """Finds the most left and most right positions of the nodes and computes
    the width.

    Finds the lowest and highest position of the nodes and computes the
    height. Then returns width, height.
    """
    xys = []
    for node_name in snn_graph.nodes:
        xys.append(snn_graph.nodes[node_name]["nx_lif"][t].pos)

    xmin = min(list(map(lambda xy: xy[0], xys)))
    xmax = max(list(map(lambda xy: xy[0], xys)))
    ymin = min(list(map(lambda xy: xy[1], xys)))
    ymax = max(list(map(lambda xy: xy[1], xys)))
    return [xmax - xmin, ymax - ymin]
