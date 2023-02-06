"""Generates a graph in dash."""

import dash
import networkx as nx
import plotly
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

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
edge_colour_I = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
]
edge_colour_II = [
    "rgb(10, 20, 30)",
    "rgb(255, 255, 0)",
]

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
        height=700,  # height of image in pixels.
        width=1000,  # Width of image in pixels.
        annotations=[
            dict(
                ax=G.nodes[edge[0]]["pos"][0],  # starting x.
                ay=G.nodes[edge[0]]["pos"][1],  # starting y.
                axref="x",
                ayref="y",
                x=G.nodes[edge[1]]["pos"][0],  # ending x.
                y=G.nodes[edge[1]]["pos"][1],  # ending y.
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
            for edge in G.edges()
        ],
    ),
)


# Start Dash app.
app = dash.Dash(__name__)


@app.callback(Output("Graph", "figure"), [Input("color-set-slider", "value")])
def update_color(color_set_index: int) -> go.Figure:
    """Updates the colour of the nodes and edges based on user input."""
    # Update the annotation colour.
    def annotation_colour(some_val: int) -> str:
        """Updates the colour of the edges based on user input."""
        if some_val == 0:
            return "yellow"
        return "red"

    # Overwrite annotation with function instead of value.
    for annotation in fig.layout.annotations:
        annotation.arrowcolor = annotation_colour(color_set_index)

    # update the node colour
    fig.data[0]["marker"]["color"] = color_sets[color_set_index]  # nodes
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
