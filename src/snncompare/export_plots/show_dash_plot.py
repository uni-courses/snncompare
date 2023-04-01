"""Creates a gif of an SNN propagation."""

import logging
from typing import Dict, List

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from typeguard import typechecked

from snncompare.export_plots.create_dash_fig_obj import create_svg_with_dash
from snncompare.export_plots.dash_plot_updaters import (
    create_app_layout,
    support_updates,
)
from snncompare.export_plots.Plot_config import Plot_config


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
    port: int,
) -> None:
    """Shows a figure in dash using browser."""
    dash_figures: Dict[str, go.Figure] = {}
    identified_annotations_dict: Dict[str, List] = {}
    temporal_node_colours_dict: Dict[str, List] = {}
    temporal_node_opacity_dict: Dict[str, List] = {}
    for graph_name, plotted_graph in plotted_graphs.items():
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

    # Silence the dash app verbosity to console.
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    # Launch the dash app in browser.
    app.run_server(port=port, threaded=True)
    print("done running surver")
