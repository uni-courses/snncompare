"""Generates a graph in dash."""

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go
from typeguard import typechecked

from snncompare.export_plots.Plot_config import Plot_config
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists

# TODO:
# Get Graph
# Get positioning parameters
# Verify x-tick labels are correct.
# Verify y-tick labels are correct.
# Verify node position is complete.
# Verify node colour set is complete.
# Verify node labels are complete.
# Verify edge colour set is complete.
# Verify edge labels are complete.


@typechecked
def xy_max(
    *,
    G: nx.DiGraph,
) -> Tuple[float, float]:
    """Computes the max x- and y-positions found in the nodes."""
    positions: List[Tuple[float, float]] = []
    for node_name in G.nodes():
        positions.append(G.nodes[node_name]["pos"])

    x = max(list(map(lambda a: a[0], positions)))
    y = max(list(map(lambda a: a[1], positions)))
    return x, y


def add_recursive_edges(
    *, G: nx.DiGraph, fig: go.Figure, plot_config: Plot_config, radius: float
) -> None:
    """Adds a circle, representing a recursive edge, above a node.

    The circle line/edge colour is updated along with the node colour.
    """
    for node_name in G.nodes:
        x, y = G.nodes[node_name]["pos"]
        # Add circles
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=x - radius,
            y0=y,
            x1=x + radius,
            y1=y + radius,
            line_color=G.nodes[node_name]["colour"],
            opacity=G.nodes[node_name]["opacity"],
            line=go.layout.shape.Line(width=plot_config.edge_width),
        )


# pylint: disable = W0621
def get_edge_arrows(*, G: nx.DiGraph, plot_config: Plot_config) -> List[Dict]:
    """Returns the annotation dictionaries representing the directed edge
    arrows."""
    annotations: List[Dict] = []
    for edge in G.edges:
        # Get coordinates
        left_node_name = edge[0]
        right_node_name = edge[1]
        left_x = G.nodes[left_node_name]["pos"][0]
        left_y = G.nodes[left_node_name]["pos"][1]
        right_x = G.nodes[right_node_name]["pos"][0]
        right_y = G.nodes[right_node_name]["pos"][1]

        # Add annotation.
        annotations.append(
            dict(
                ax=left_x,
                ay=left_y,
                axref="x",
                ayref="y",
                opacity=G.edges[edge]["opacity"],
                x=right_x,
                y=right_y,
                xref="x",
                yref="y",
                arrowwidth=plot_config.edge_width,  # Width of arrow.
                arrowcolor=G.nodes[edge[0]]["colour"],
                arrowsize=0.8,  # (1 gives head 3 times as wide as arrow line)
                showarrow=True,
                arrowhead=1,  # the arrowshape (index).
                hoverlabel=plotly.graph_objs.layout.annotation.Hoverlabel(
                    bordercolor="red"
                ),
            )
        )
    return annotations


# pylint: disable = W0621
def get_edge_labels(
    *,
    G: nx.DiGraph,
    pixel_height: int,
    pixel_width: int,
    plot_config: Plot_config,
    radius: float,
) -> List[Dict]:
    """Returns the annotation dictionaries representing the labels of the
    directed edge arrows.

    Returns the annotation dictionaries representing the labels of the
    recursive edge circles above the nodes. Note, only place 0.25 radius above
    pos, because recursive edge circles are actually ovals.

    with height: 1 * radius, width:2 * radius, and you want to place the
    recursive edge label in the center of the oval.
    """
    annotations = []
    for edge in G.edges:
        if edge[0] != edge[1]:  # For non recursive edges
            mid_x, mid_y = get_edge_mid_point(G=G, edge=edge)
            annotations.append(
                go.layout.Annotation(
                    x=mid_x,
                    y=mid_y,
                    xref="x",
                    yref="y",
                    text=G.edges[edge]["label"],
                    font={"size": plot_config.neuron_text_size},
                    align="center",
                    showarrow=False,
                    yanchor="bottom",
                    textangle=get_stretched_edge_angle(
                        G=G,
                        edge=edge,
                        pixel_height=pixel_height,
                        pixel_width=pixel_width,
                    ),
                )
            )
        else:  # Recursive edge.

            x, y = G.nodes[edge[0]]["pos"]
            annotations.append(
                go.layout.Annotation(
                    x=x,
                    y=y + 0.25 * radius,
                    xref="x",
                    yref="y",
                    text=G.edges[edge]["label"],
                    font={"size": plot_config.neuron_text_size},
                    align="center",
                    showarrow=False,
                    yanchor="bottom",
                )
            )
    return annotations


def get_edge_mid_point(
    edge: Tuple[Tuple[int, int], Tuple[int, int]],
    G: nx.DiGraph,
) -> Tuple[int, int]:
    """Returns the mid point of an edge."""
    left_node_name = edge[0]
    right_node_name = edge[1]
    left_x = G.nodes[left_node_name]["pos"][0]
    left_y = G.nodes[left_node_name]["pos"][1]
    right_x = G.nodes[right_node_name]["pos"][0]
    right_y = G.nodes[right_node_name]["pos"][1]
    mid_x = (right_x + left_x) / 2
    mid_y = (right_y + left_y) / 2
    return mid_x, mid_y


def get_stretched_edge_angle(
    *,
    edge: Tuple[Tuple[int, int], Tuple[int, int]],
    G: nx.DiGraph,
    pixel_height: int,
    pixel_width: int,
) -> Tuple[int, int]:
    """Returns the ccw+ mid point of an edge and adjusts for stretching of the
    image."""
    left_node_name = edge[0]
    right_node_name = edge[1]
    left_x = G.nodes[left_node_name]["pos"][0]
    left_y = G.nodes[left_node_name]["pos"][1]
    right_x = G.nodes[right_node_name]["pos"][0]
    right_y = G.nodes[right_node_name]["pos"][1]
    dx = (right_x - left_x) * (
        1 - ((pixel_height - pixel_width) / pixel_height)
    )
    # dx =
    dy = right_y - left_y
    angle = np.arctan2(dy, dx)
    # return -np.rad2deg((angle) % (2 * np.pi))
    return -np.rad2deg(angle)


def get_pure_edge_angle(
    G: nx.DiGraph, edge: Tuple[Tuple[int, int], Tuple[int, int]]
) -> Tuple[int, int]:
    """Returns the ccw+ mid point of an edge."""
    left_node_name = edge[0]
    right_node_name = edge[1]
    left_x = G.nodes[left_node_name]["pos"][0]
    left_y = G.nodes[left_node_name]["pos"][1]
    right_x = G.nodes[right_node_name]["pos"][0]
    right_y = G.nodes[right_node_name]["pos"][1]
    dx = right_x - left_x
    dy = right_y - left_y
    angle = np.arctan2(dy, dx)
    # return -np.rad2deg((angle) % (2 * np.pi))
    return -np.rad2deg(angle)


def get_annotations(
    G: nx.DiGraph,
    pixel_height: int,
    pixel_width: int,
    plot_config: Plot_config,
    recursive_edge_radius: float,
) -> List[Dict]:
    """Returns the annotations for this graph."""
    annotations = []
    annotations.extend(get_edge_arrows(G=G, plot_config=plot_config))
    annotations.extend(
        get_edge_labels(
            G=G,
            pixel_height=pixel_height,
            pixel_width=pixel_width,
            plot_config=plot_config,
            radius=recursive_edge_radius,
        )
    )

    return annotations


# Build image from incoming graph and positioning parameters
# pylint: disable=R0914
def create_svg_with_dash(
    filename: str, graphs: List[nx.DiGraph], plot_config: Plot_config, t: int
) -> None:
    """Creates an .svg plot of the incoming networkx graph."""
    # for node_name in graphs[t].nodes():
    # if "spike_once_" == node_name[:11] or "rand_" == node_name[:5]:
    # print(f'{node_name}:{graphs[t].nodes[node_name]["pos"]}')
    pixel_width = plot_config.base_pixel_width * xy_max(G=graphs[t])[0]
    pixel_height = plot_config.base_pixel_height * xy_max(G=graphs[t])[1]
    recursive_edge_radius = plot_config.recursive_edge_radius

    # add color to node points
    colour_set: Dict[str, str] = {}
    colour_list: List[str] = []
    for node_name in graphs[t].nodes():
        colour_set[node_name] = graphs[t].nodes[node_name]["colour"]
        colour_list.append(graphs[t].nodes[node_name]["colour"])

    # Load edge colour list.
    edge_colours: List[Dict[Tuple[str, str], str]] = [{}] * len(graphs)
    for edge in graphs[t].edges():
        edge_colours[t][edge] = graphs[t].edges[edge]["colour"]

    node_labels = []
    for node_name in graphs[t].nodes():
        node_labels.append(graphs[t].nodes[node_name]["label"])

    x_pos: List[float] = []
    y_pos: List[float] = []
    text_array = []
    for node_name in graphs[t].nodes():
        x, y = graphs[t].nodes[node_name]["pos"]
        x_pos.append(x)
        y_pos.append(y)
        text_array.append(graphs[t].nodes[node_name]["label"])

    # Create nodes
    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        text=text_array,
        mode="markers+text",
        hoverinfo="none",
        marker=dict(
            size=plot_config.node_size,
            color=colour_list,
        ),
        textfont={"size": plot_config.neuron_text_size},
        # textposition="bottom right",
    )

    # Create figure
    fig = go.Figure(
        # data=[edge_trace, node_trace],
        data=[node_trace],
        layout=go.Layout(
            height=pixel_height,  # height of image in pixels.
            width=pixel_width,  # Width of image in pixels.
            annotations=get_annotations(
                G=graphs[t],
                pixel_height=pixel_height,
                pixel_width=pixel_width,
                plot_config=plot_config,
                recursive_edge_radius=plot_config.recursive_edge_radius,
            ),
            xaxis=go.layout.XAxis(
                tickmode="array",
                tickvals=list(graphs[t].graph["x_tics"].keys()),
                ticktext=list(graphs[t].graph["x_tics"].values()),
                tickfont={"size": plot_config.x_tick_size},
                tickangle=-45,
            ),
            yaxis=go.layout.YAxis(
                tickmode="array",
                tickvals=list(graphs[t].graph["y_tics"].keys()),
                ticktext=list(graphs[t].graph["y_tics"].values()),
                tickfont={"size": plot_config.y_tick_size},
                tickangle=0,
            ),
        ),
    )
    add_recursive_edges(
        G=graphs[t],
        fig=fig,
        plot_config=plot_config,
        radius=recursive_edge_radius,
    )
    create_root_dir_if_not_exists(root_dir_name="latex/Images/graphs")
    fig.write_image(f"latex/Images/graphs/{filename}.svg")
    # fig.show()
    # os.system("nemo /home/name/git/snn/snncompare/latex/Images/graphs")
    return fig
