"""Creates a gif of an SNN propagation."""

import base64
import os
from typing import Dict, List

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from flask import Response, send_from_directory
from typeguard import typechecked

from snncompare.export_plots.create_dash_fig_obj import create_svg_with_dash
from snncompare.export_plots.dash_plot_updaters import (
    create_app_layout,
    support_updates,
)
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
    print("done running server")


@typechecked
def show_dash_figures(
    *,
    app: dash.Dash,
    plot_config: Plot_config,
    plotted_graphs: Dict[str, nx.DiGraph],
) -> None:
    """Shows a figure in dash using browser."""
    print("SHOWING DASH APP.")
    dash_figures: Dict[str, go.Figure] = {}
    identified_annotations_dict: Dict[str, List] = {}
    temporal_node_colours_dict: Dict[str, List] = {}
    temporal_node_opacity_dict: Dict[str, List] = {}
    for graph_name, plotted_graph in plotted_graphs.items():
        print(f"graph_name={graph_name}")
        # Start Dash app.
        (
            dash_figures[graph_name],
            identified_annotations_dict[graph_name],
        ) = create_svg_with_dash(
            graph=plotted_graph,
            plot_config=plot_config,
        )

        temporal_node_colours_dict[graph_name] = list(
            plotted_graph.nodes[n]["temporal_colour"]
            for n in plotted_graph.nodes()
        )
        temporal_node_opacity_dict[graph_name] = list(
            plotted_graph.nodes[n]["temporal_opacity"]
            for n in plotted_graph.nodes()
        )

    app = create_app_layout(
        app=app,
        dash_figures=dash_figures,
        plotted_graphs=plotted_graphs,
        temporal_node_colours_dict=temporal_node_colours_dict,
    )

    support_updates(
        app=app,
        dash_figures=dash_figures,
        identified_annotations_dict=identified_annotations_dict,
        plot_config=plot_config,
        plotted_graphs=plotted_graphs,
        temporal_node_colours_dict=temporal_node_colours_dict,
        temporal_node_opacity_dict=temporal_node_opacity_dict,
    )

    app.run_server()
    print("done running surver")
    # app.run_server(debug=True)
    # if __name__ == "__main__":
    #    app.run_server(debug=True)
