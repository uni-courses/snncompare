"""Generates a graph in dash."""

import dash
import dash_core_components as dcc
import dash_html_components as html
import networkx as nx
import plotly.graph_objs as go
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


# Create Edges
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines",
)

for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]["pos"]
    x1, y1 = G.nodes[edge[1]]["pos"]
    edge_trace["x"] += tuple([x0, x1, None])
    edge_trace["y"] += tuple([y0, y1, None])

# Start of Dash app.
app = dash.Dash(__name__)

fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ),
)

# State variable to keep track of current color set
color_set_index = 0
color_sets = [colour_set_I, colour_set_II]

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


@app.callback(Output("Graph", "figure"), [Input("color-set-slider", "value")])
def update_color(updated_color_set_index: int) -> go.Figure:
    """Updates the colour in realtime based on the color-set-slider element.

    TODO: verify if still works with name changed from color_set_index to
    updated_color_set_index.
    """
    fig.data[1]["marker"]["color"] = color_sets[updated_color_set_index]
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
