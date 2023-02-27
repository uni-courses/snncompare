"""Generates interactive view of graph."""
from pathlib import Path
from typing import Dict, List, Optional, Union

import dash
import networkx as nx
import plotly.graph_objs as go
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
    show_svg_image_in_dash_III,
)
from snncompare.export_plots.store_plot_data_in_graph import (
    store_plot_params_in_graph,
)
from snncompare.optional_config.Output_config import Output_config


# Determine which graph(s) the user would like to see.
# If no specific preference specified, show all 4.
# pylint: disable=R0903
# pylint: disable=R0914
@typechecked
def create_svg_plot(
    run_config_filename: str,
    graph_names: List[str],
    graphs: Dict[str, Union[nx.Graph, nx.DiGraph]],
    output_config: Output_config,
    # single_timestep: Optional[int] = 5,
    single_timestep: Optional[int] = None,
) -> None:
    """Creates the svg plots."""
    plot_config: Plot_config = get_default_plot_config()

    app = dash.Dash(__name__)
    # pylint: disable=R1702
    for graph_name, snn_graph in graphs.items():
        if graph_name in graph_names:
            plotted_graph: nx.DiGraph = nx.DiGraph()

            dash_figures: List[go.Figure] = []
            print("")
            print("")

            sim_duration = snn_graph.graph["sim_duration"]
            print(f"Creating:graph_name={graph_name}")
            for t in range(
                0,
                sim_duration,
                # 1,
            ):
                print(f"t={t}/{sim_duration}")
                # if t == 5:
                # for edge in snn_graph.edges():
                # if "selector" in edge[0] or "selector" in edge[1]:
                # print(edge)

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
                    dash_figures.append(dash_figure)
                if not Path(svg_filepath).is_file():
                    # TODO move storing into separate function.
                    create_root_dir_if_not_exists(
                        root_dir_name="latex/Images/graphs"
                    )
                    dash_figure.write_image(svg_filepath)
            # Show the images
            if output_config.extra_storing_config.show_images:
                if single_timestep is not None:
                    # Show only a single timestep from dash object or svg file.
                    if len(dash_figures) >= single_timestep:
                        # TODO: This can be done faster, not complete .svg arr.
                        # needs to be created.
                        show_fig_in_dash(
                            app=app, fig=dash_figures[single_timestep]
                        )
                    else:
                        raise NotImplementedError(
                            "Showing svgs not yet supported."
                        )
                        # pylint: disable=W0101
                        show_svg_image_in_dash_III(app=app, path=svg_filepath)
                else:
                    # Show a whole timeseries of bash figures.
                    # TODO: allow showing multiple graphs.
                    show_dash_figures(
                        app=app,
                        plot_config=plot_config,
                        plotted_graph=plotted_graph,
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
