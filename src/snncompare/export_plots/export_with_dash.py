"""Generates a graph in dash."""

from typing import Dict, List, Tuple

import dash
import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from typeguard import typechecked

from snncompare.export_plots.Plot_config import Plot_config
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists

# TODO: compute x-tick labels.
# TODO: compute y-tick labels.
# TODO: compute node position.
# TODO: compute node colour.
# TODO: compute node labels.
# TODO: compute edge colour.
# TODO: compute edge labels.

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
    *, G: nx.DiGraph, fig: go.Figure, radius: float
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
        )


# pylint: disable = W0621
def get_edge_arrows(*, G: nx.DiGraph) -> List[Dict]:
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
                x=right_x,
                y=right_y,
                xref="x",
                yref="y",
                arrowwidth=5,  # Width of arrow.
                arrowcolor="red",  # Overwrite in update/using user input.
                arrowsize=0.8,  # (1 gives head 3 times as wide as arrow line)
                showarrow=True,
                arrowhead=1,  # the arrowshape (index).
                hoverlabel=plotly.graph_objs.layout.annotation.Hoverlabel(
                    bordercolor="red"
                ),
                hovertext="sometext",
                text="sometext",
                # textangle=-45,
                # xanchor='center',
                # xanchor='right',
                # swag=120,
            )
        )
    return annotations


# pylint: disable = W0621
def get_edge_labels(
    *, G: nx.DiGraph, pixel_height: int, pixel_width: int
) -> List[Dict]:
    """Returns the annotation dictionaries representing the labels of the
    directed edge arrows."""
    annotations = []
    for edge in G.edges:
        mid_x, mid_y = get_edge_mid_point(G=G, edge=edge)
        annotations.append(
            go.layout.Annotation(
                x=mid_x,
                y=mid_y,
                xref="x",
                yref="y",
                text="dict Text",
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
    return annotations


# pylint: disable = W0621
def get_recursive_edge_labels(G: nx.DiGraph, radius: float) -> List[Dict]:
    """Returns the annotation dictionaries representing the labels of the
    recursive edge circles above the nodes. Note, only place 0.25 radius above
    pos, because recursive edge circles are actually ovals.

    with height: 1 * radius, width:2 * radius, and you want to place the
    recursive edge label in the center of the oval.
    """
    annotations = []
    for node in G.nodes:
        x, y = G.nodes[node]["pos"]
        annotations.append(
            go.layout.Annotation(
                x=x,
                y=y + 0.25 * radius,
                xref="x",
                yref="y",
                text="recur",
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
    recursive_edge_radius: float,
) -> List[Dict]:
    """Returns the annotations for this graph."""
    annotations = []
    annotations.extend(get_edge_arrows(G=G))
    annotations.extend(
        get_edge_labels(
            G=G, pixel_height=pixel_height, pixel_width=pixel_width
        )
    )
    annotations.extend(
        get_recursive_edge_labels(G, radius=recursive_edge_radius)
    )

    return annotations


# Build image from incoming graph and positioning parameters
# pylint: disable=R0914
def create_svg_with_dash(
    filename: str, graphs: List[nx.DiGraph], plot_config: Plot_config
) -> None:
    """Creates an .svg plot of the incoming networkx graph."""

    t = 0
    print(f'node={graphs[0].nodes["spike_once_0"]["nx_lif"]}')
    pixel_width = plot_config.base_pixel_height * xy_max(G=graphs[t])[0]
    pixel_height = plot_config.base_pixel_height * xy_max(G=graphs[t])[1]
    recursive_edge_radius = plot_config.recursive_edge_radius

    # add color to node points
    colour_set: Dict[str, str] = {}
    colour_list: List[str] = []
    for node_name in graphs[t].nodes():
        colour_set[node_name] = graphs[t].nodes[node_name]["colour"]
        colour_list.append(graphs[t].nodes[node_name]["colour"])

    # Load edge colour list.
    # edge_colours: List[Dict[str,str]] = set_edge_colours()
    # TODO: change into list of dicts with one dict per timestep.
    edge_colours: List[Dict[Tuple[str, str], str]] = [{}] * len(graphs)
    for edge in graphs[t].edges():
        edge_colours[t][edge] = graphs[t].edges[edge]["colour"]

    # Create nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            size=30,
            color=colour_list,
        )
        # marker=dict(size=30, color=graphs[t].nodes[:]["colour"]),
    )

    for i, node_name in enumerate(graphs[t].nodes()):
        x, y = graphs[t].nodes[node_name]["pos"]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])

        # node_trace["marker"]["color"][i] = colour_set[node_name]

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
                recursive_edge_radius=plot_config.recursive_edge_radius,
            ),
            xaxis=go.layout.XAxis(
                tickmode="array",
                tickvals=[0, 0.1, 0.5, 1, 2.0],
                ticktext=["origin", "r1", "r2", "MID", "end"],
                tickangle=-45,
            ),
            yaxis=go.layout.YAxis(
                tickmode="array",
                tickvals=[0, 0.1, 0.5, 0.9, 1.0],
                ticktext=["yorigin", "yr1", "yr2", "yMID", "yend"],
                tickangle=0,
            ),
        ),
    )
    add_recursive_edges(G=graphs[t], fig=fig, radius=recursive_edge_radius)

    # Start Dash app.
    app = dash.Dash(__name__)

    @app.callback(
        Output("Graph", "figure"), [Input("color-set-slider", "value")]
    )
    def update_color(color_set_index: int) -> go.Figure:
        """Updates the colour of the nodes and edges based on user input."""
        # Update the annotation colour.
        def annotation_colour(
            some_val: int,
            edge_colours: List[Dict[Tuple[str, str], str]],
            edge: Tuple[str, str],
        ) -> str:
            """Updates the colour of the edges based on user input."""
            return edge_colours[some_val][edge]

        # Overwrite annotation with function instead of value.
        for i, edge in enumerate(graphs[t].edges()):

            some_annotation_colour = annotation_colour(
                color_set_index, edge_colours=edge_colours, edge=edge
            )
            fig.layout.annotations[i].arrowcolor = some_annotation_colour

        # Update the node colour.
        fig.data[0]["marker"]["color"] = color_sets[color_set_index]  # nodes

        # Update the recursive edge node colour.
        for i, _ in enumerate(graphs[t].nodes):
            fig.layout.shapes[i]["line"]["color"] = color_sets[
                color_set_index
            ][i]
        create_root_dir_if_not_exists(root_dir_name="latex/Images/graphs")
        fig.write_image(f"latex/Images/graphs/{filename}_{t}.svg")
        return fig

    # State variable to keep track of current color set
    initial_color_set_index = 0
    color_sets = [colour_list, colour_list]
    fig = update_color(initial_color_set_index)

    app.layout = html.Div(
        [
            dcc.Slider(
                id="color-set-slider",
                min=0,
                max=len(color_sets) - 1,
                value=0,
                marks={i: str(i) for i in range(len(color_sets))},
                step=None,
            ),
            html.Div(dcc.Graph(id="Graph", figure=fig)),
        ]
    )

    # app.run_server(debug=True)
    # if __name__ == "__main__":
    #    app.run_server(debug=True)
