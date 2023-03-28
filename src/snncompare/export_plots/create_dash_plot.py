"""Generates interactive view of graph."""
from pathlib import Path
from typing import Dict, List, Optional, Union

import dash
import networkx as nx
import plotly.graph_objs as go
from simsnn.core.simulators import Simulator
from snnbackends.simsnn.simsnn_to_nx_lif import (
    add_simsnn_simulation_data_to_reconstructed_nx_lif,
    simsnn_graph_to_nx_lif_graph,
)
from typeguard import typechecked

from snncompare.export_plots.create_dash_fig_obj import create_svg_with_dash
from snncompare.export_plots.Plot_config import (
    Plot_config,
    get_default_plot_config,
)
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists
from snncompare.export_plots.show_dash_plot import (
    show_dash_figures,
    show_fig_in_dash,
)
from snncompare.export_plots.store_plot_data_in_graph import (
    store_plot_params_in_graph,
)
from snncompare.helper import get_some_duration
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config


# Determine which graph(s) the user would like to see.
# If no specific preference specified, show all 4.
# pylint: disable=R0903
# pylint: disable=R0913
# pylint: disable=R0914
@typechecked
def create_svg_plot(
    run_config_filename: str,
    graph_names: List[str],
    graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    output_config: Output_config,
    run_config: Run_config,
    # single_timestep: Optional[int] = 5,
    single_timestep: Optional[int] = None,
) -> None:
    """Creates the svg plots."""
    plot_config: Plot_config = get_default_plot_config()

    app = dash.Dash(__name__)

    # pylint: disable=R1702
    dash_figures: Dict[str, List[go.Figure]] = {}
    plotted_graphs: Dict[str, nx.DiGraph] = {}

    for _, (graph_name, snn_graph) in enumerate(graphs.items()):
        if graph_name in graph_names:
            print("")
            print("")

            sim_duration = get_some_duration(
                simulator=run_config.simulator,
                snn_graph=snn_graph,
                duration_name="actual_duration",
            )

            print(f"Creating:graph_name={graph_name}")

            # Convert simsnn to nx_LIF
            if (
                run_config.simulator == "simsnn"
                and graph_name != "input_graph"
            ):
                nx_snn: nx.DiGraph = simsnn_graph_to_nx_lif_graph(
                    simsnn=snn_graph
                )

                # Add time dimension to nx_snn that was created from simsnn.
                add_simsnn_simulation_data_to_reconstructed_nx_lif(
                    nx_snn=nx_snn,
                    simsnn=snn_graph,
                )

            else:
                nx_snn = snn_graph
            create_figures(
                graph_name=graph_name,
                run_config_filename=run_config_filename,
                output_config=output_config,
                plot_config=plot_config,
                sim_duration=sim_duration,
                snn_graph=nx_snn,
                single_timestep=single_timestep,
                dash_figures=dash_figures,
                plotted_graphs=plotted_graphs,
            )

    show_figures(
        app=app,
        dash_figures=dash_figures,
        output_config=output_config,
        plot_config=plot_config,
        plotted_graphs=plotted_graphs,
        single_timestep=single_timestep,
    )


# pylint: disable=R0913
@typechecked
def create_figures(
    graph_name: str,
    run_config_filename: str,
    output_config: Output_config,
    plot_config: Plot_config,
    sim_duration: int,
    snn_graph: nx.DiGraph,
    single_timestep: Optional[bool],
    dash_figures: Dict[str, List[go.Figure]],
    plotted_graphs: Dict[str, nx.DiGraph],
) -> None:
    """Creates the dash figures."""
    dash_screens: List[go.Figure] = []
    plotted_graph: nx.DiGraph = nx.DiGraph()
    for t in range(
        0,
        sim_duration,
        # 1,
    ):
        print(f"t={t}/{sim_duration}")

        # Create and store the svg images per timestep.
        filename: str = f"{graph_name}_{run_config_filename}_{t}"
        svg_filepath: str = f"latex/Images/graphs/{filename}.svg"
        if not Path(svg_filepath).is_file() or (
            output_config.extra_storing_config.show_images
            and single_timestep is None
        ):
            dash_figure: go.Figure = create_dash_figure(
                output_config=output_config,
                plot_config=plot_config,
                plotted_graph=plotted_graph,
                snn_graph=snn_graph,
                t=t,
            )
            dash_screens.append(dash_figure)
        if not Path(svg_filepath).is_file():
            # TODO move storing into separate function.
            create_root_dir_if_not_exists(root_dir_name="latex/Images/graphs")
            dash_figure.write_image(svg_filepath)
    dash_figures[graph_name] = dash_screens
    plotted_graphs[graph_name] = plotted_graph


# pylint: disable=R0913
@typechecked
def show_figures(
    app: dash.dash.Dash,
    dash_figures: Dict[str, List[go.Figure]],
    output_config: Output_config,
    plot_config: Plot_config,
    plotted_graphs: Dict[str, nx.DiGraph],
    single_timestep: Optional[bool],
) -> None:
    """Shows the dash figures."""
    # Show the images
    if output_config.extra_storing_config.show_images:
        for dash_screens in dash_figures.values():
            if single_timestep is not None:
                # Show only a single timestep from dash object or svg file.
                if len(dash_screens) >= single_timestep:
                    # TODO: This can be done faster, not complete .svg arr.
                    # needs to be created.
                    show_fig_in_dash(
                        app=app, fig=dash_screens[single_timestep]
                    )
            else:
                # Show a whole timeseries of dash figures.
                # TODO: allow showing multiple graphs.
                show_dash_figures(
                    app=app,
                    plot_config=plot_config,
                    plotted_graphs=plotted_graphs,
                )


def create_dash_figure(
    output_config: Output_config,
    plot_config: Plot_config,
    plotted_graph: nx.DiGraph,
    snn_graph: nx.DiGraph,
    t: int,
) -> go.Figure:
    """Creates and stores a dash figure object as .svg."""
    store_plot_params_in_graph(
        hover_info=output_config.hover_info,
        plotted_graph=plotted_graph,
        snn_graph=snn_graph,
        t=t,
    )

    # TODO: determine whether identified_annotations are needed lateron.
    dash_figure, _ = create_svg_with_dash(
        graph=plotted_graph,
        plot_config=plot_config,
    )
    return dash_figure
