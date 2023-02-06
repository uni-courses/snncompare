"""Generates a graph in dash."""

from typing import Dict, List, Tuple

import dash
import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

pixel_width = 1000
pixel_height = 1000
recursive_edge_radius = 0.1
# Create graph G
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2])
G.add_edges_from(
    [
        (0, 1),
        (0, 2),
    ],
    weight=6,
)

# Create a x,y position for each node
pos = {
    0: [0, 0],
    1: [1, 2],
    2: [2, 0],
}
# Set the position attribute with the created positions.
for node in G.nodes:
    G.nodes[node]["pos"] = list(pos[node])

# add color to node points
colour_set_I = ["rgb(31, 119, 180)", "rgb(255, 127, 14)", "rgb(44, 160, 44)"]
colour_set_II = ["rgb(10, 20, 30)", "rgb(255, 255, 0)", "rgb(0, 255, 255)"]

# Create edge colour lists.
# TODO: make edge colour function of edges


def set_edge_colours() -> List[List[str]]:
    """(Manually) set edge colours into list."""
    hardcoded_edge_colours = [
        [
            "rgb(31, 119, 180)",  # edge(0,1)
            "rgb(255, 127, 14)",  # edge(0,2)
        ],
        [
            "rgb(10, 20, 30)",  # edge(0,1)
            "rgb(255, 255, 0)",  # edge(0,2)
        ],
    ]
    return hardcoded_edge_colours


def get_edge_colour(
    t: int,
    edge: Tuple[int, int],
    edge_colours: List[List[str]],  # pylint: disable = W0621
) -> str:
    """Returns an edge colour based on edge
    TODO: support duplicate edges between nodes."""
    if edge == (0, 1):
        return edge_colours[t][0]
    if edge == (0, 2):
        return edge_colours[t][1]
    raise ValueError(f"Error, edge{edge} not found.")


# pylint: disable = W0621
def get_edge_arrows(G: nx.DiGraph) -> List[Dict]:
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
def get_edge_labels(G: nx.DiGraph) -> List[Dict]:
    """Returns the annotation dictionaries representing the labels of the
    directed edge arrows."""
    annotations = []
    for edge in G.edges:
        mid_x, mid_y = get_edge_mid_point(edge)
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
                    edge, pixel_height=pixel_height, pixel_width=pixel_width
                ),
            )
        )
    return annotations


# pylint: disable = W0621
def get_recursive_edge_labels(G: nx.DiGraph, radius: float) -> List[Dict]:
    """Returns the annotation dictionaries representing the labels of the
    recursive edge circles above the nodes. Note, only place 1 radius above
    pos, because recursive edge circles are.

    actually ovals with height: radius, width:2 * radius.
    """
    annotations = []
    for node in G.nodes:
        x, y = G.nodes[node]["pos"]
        annotations.append(
            go.layout.Annotation(
                x=x,
                y=y + 1 * radius,
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
    edge: Tuple[Tuple[int, int], Tuple[int, int]]
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
    edge: Tuple[Tuple[int, int], Tuple[int, int]],
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
    edge: Tuple[Tuple[int, int], Tuple[int, int]]
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


def get_annotations(G: nx.DiGraph) -> List[Dict]:
    """Returns the annotations for this graph."""
    annotations = []
    annotations.extend(get_edge_arrows(G))
    annotations.extend(get_edge_labels(G))
    annotations.extend(
        get_recursive_edge_labels(G, radius=recursive_edge_radius)
    )

    return annotations


def add_recursive_edges(G: nx.DiGraph, fig: go.Figure, radius: float) -> None:
    """Adds a circle, representing a recursive edge, above a node.

    The circle line/edge colour is updated along with the node colour.
    """
    for node in G.nodes:
        x, y = G.nodes[node]["pos"]
        # Add circles
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=x - radius,
            y0=y,
            x1=x + radius,
            y1=y + radius,
            line_color=colour_set_I[node],
        )


# Load edge colour list.
edge_colours: List[List[str]] = set_edge_colours()


# Create nodes
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode="markers",
    hoverinfo="text",
    marker=dict(size=30, color=colour_set_I),
)

for node in G.nodes():
    x, y = G.nodes[node]["pos"]
    node_trace["x"] += tuple([x])
    node_trace["y"] += tuple([y])

# Create figure
fig = go.Figure(
    # data=[edge_trace, node_trace],
    data=[node_trace],
    layout=go.Layout(
        height=pixel_height,  # height of image in pixels.
        width=pixel_width,  # Width of image in pixels.
        annotations=get_annotations(G),
    ),
)
add_recursive_edges(G=G, fig=fig, radius=recursive_edge_radius)

# Start Dash app.
app = dash.Dash(__name__)


@app.callback(Output("Graph", "figure"), [Input("color-set-slider", "value")])
def update_color(color_set_index: int) -> go.Figure:
    """Updates the colour of the nodes and edges based on user input."""
    # Update the annotation colour.
    def annotation_colour(
        some_val: int,
        edge_colours: List[List[str]],
        edge: Tuple[int, int],
    ) -> str:
        """Updates the colour of the edges based on user input."""
        return get_edge_colour(
            t=some_val, edge=edge, edge_colours=edge_colours
        )

    # Overwrite annotation with function instead of value.
    for i, edge in enumerate(G.edges()):

        some_annotation_colour = annotation_colour(
            color_set_index, edge_colours=edge_colours, edge=edge
        )
        fig.layout.annotations[i].arrowcolor = some_annotation_colour

    # Update the node colour.
    fig.data[0]["marker"]["color"] = color_sets[color_set_index]  # nodes

    # Update the recursive edge node colour.
    for node_name in G.nodes:
        fig.layout.shapes[node_name]["line"]["color"] = color_sets[
            color_set_index
        ][node_name]

    return fig


# State variable to keep track of current color set
initial_color_set_index = 0
color_sets = [colour_set_I, colour_set_II]
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

if __name__ == "__main__":
    app.run_server(debug=True)
