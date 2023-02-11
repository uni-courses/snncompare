"""Creates a gif of an SNN propagation."""

import base64
import os
from typing import List, Tuple

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Response, send_from_directory
from typeguard import typechecked

from snncompare.export_plots.create_dash_fig_obj import create_svg_with_dash
from snncompare.export_plots.Plot_config import Plot_config


@typechecked
def show_svg_image_in_dash_I(*, svg_filepath: str) -> None:
    """Shows a svg file in dash using browser."""
    # Start Dash app.
    app = dash.Dash(__name__)
    app.layout = html.Div([html.Img(src=svg_filepath)])
    app.run_server(debug=True)


@typechecked
def show_svg_image_in_dash_II(*, svg_filepath: str) -> None:
    """Shows a svg file in dash using browser."""
    # Start Dash app.
    app = dash.Dash(__name__)

    # pylint: disable=R1732
    encoded_image = base64.b64encode(open(svg_filepath, "rb").read()).decode()
    app.layout = html.Div(
        [html.Img(src=f"data:image/svg;base64,{encoded_image}")]
    )
    app.run_server(debug=True)


@typechecked
# pylint: disable=W0613
def show_svg_image_in_dash_III(*, app: dash.Dash, path: str) -> None:
    """Shows a svg file in dash using browser."""

    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True

    app.layout = html.Div(
        [
            html.Link(rel="stylesheet", href="/static/stylesheet.css"),
            html.Div("Hello world"),
        ]
    )

    # pylint: disable=W0613
    @app.server.route("/static/<path:path>")
    def static_file(path: str) -> "Response":
        static_folder = os.path.join(os.getcwd(), "static")
        return send_from_directory(static_folder, path)

    app.run_server(debug=True)


@typechecked
def show_svg_image_in_dash_IV(*, path: str) -> None:
    """Shows a svg file in dash using browser."""

    app = dash.Dash()

    # pylint: disable=R1732
    encoded_image = base64.b64encode(open(path, "rb").read())

    app.layout = html.Div(
        [
            html.Img(
                # pylint: disable=R1732
                src=(
                    "data:image/svg;base64,"
                    + f"{encoded_image}"  # type:ignore[str-bytes-safe]
                )
            )
        ]
    )

    app.run_server(debug=True)


@typechecked
# def show_fig_in_dash(*, fig: go._figure.Figure) -> None:
def show_fig_in_dash(*, app: dash.Dash, fig: go.Figure) -> None:
    """Shows a figure in dash using browser."""
    print("Showing Dash figure.")
    # Start Dash app.
    app.layout = html.Div(
        [
            html.Div(dcc.Graph(id="Graph", figure=fig)),
        ]
    )
    # app.run_server(debug=True)
    app.run_server()
    print("done running surver")


@typechecked
def show_dash_figures(
    *,
    app: dash.Dash,
    plot_config: Plot_config,
    plotted_graph: nx.DiGraph,
) -> None:
    """Shows a figure in dash using browser."""
    print("SHOWING DASH APP.")
    # Start Dash app.
    dash_figure: go.Figure = create_svg_with_dash(
        graph=plotted_graph,
        plot_config=plot_config,
    )
    temporal_node_colours = list(
        plotted_graph.nodes[n]["temporal_colour"]
        for n in plotted_graph.nodes()
    )

    @app.callback(
        Output("Graph", "figure"), [Input("color-set-slider", "value")]
    )
    def update_color(t: int) -> go.Figure:
        """Updates the colour of the nodes and edges based on user input."""

        # Update the annotation colour.
        def edge_annotation_colour(
            t: int,
            temporal_node_colours: List,
            edge: Tuple[str, str],
        ) -> str:
            """Updates the colour of the edges based on user input."""
            for i, node_name in enumerate(
                list(
                    some_node_name for some_node_name in plotted_graph.nodes()
                )
            ):
                if "connector" not in node_name:
                    if node_name == edge[0]:
                        return temporal_node_colours[i][t]
            # pylint: disable=W0631
            raise Exception(f"Error, node_name:{node_name} not found.")

        # Overwrite annotation with function instead of value.
        if plot_config.update_edge_colours:
            for i, edge in enumerate(plotted_graph.edges()):
                # TODO: remove this check and require all nodes, edges and
                # annotations to be accepted.
                if i < len(dash_figure.layout.annotations):
                    print(f"i={i}")
                    the_edge_annotation_colour = edge_annotation_colour(
                        t,
                        temporal_node_colours=temporal_node_colours,
                        edge=edge,
                    )
                    dash_figure.layout.annotations[
                        i
                    ].arrowcolor = the_edge_annotation_colour
        return dash_figure

        # Update the node colour.
        # dash_figure.data[0]["marker"]["color"] = list(
        # temporal_node_colours[t].values()
        # )

        # Update the recursive edge node colour.
        # for i, _ in enumerate(graphs[t].nodes):
        #    fig.layout.shapes[i]["line"]["color"] = color_sets[t][i]

    # State variable to keep track of current color set
    initial_t = 0
    # color_sets = [colour_list, colour_list]
    # TODO: ensure the colours are initiated at least once regardless of
    # plot_config.update_..
    dash_figure = update_color(initial_t)

    if len(temporal_node_colours[0]) == 0:
        raise ValueError(
            "Not enough timesteps were found. probably took timestep of "
            + "ignored node."
        )
    app.layout = html.Div(
        [
            dcc.Slider(
                id="color-set-slider",
                min=0,
                max=len(temporal_node_colours[0]) - 1,
                value=0,
                marks={
                    i: str(i) for i in range(len(temporal_node_colours[0]))
                },
                step=None,
            ),
            html.Div(dcc.Graph(id="Graph", figure=dash_figure)),
        ]
    )
    app.run_server()
    print("done running surver")
    # app.run_server(debug=True)
    # if __name__ == "__main__":
    #    app.run_server(debug=True)
