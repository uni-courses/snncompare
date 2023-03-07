"""Updates dash plots."""
import os
from typing import Dict, List, Tuple

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from typeguard import typechecked

from snncompare.export_plots.create_dash_fig_obj import NamedAnnotation
from snncompare.export_plots.Plot_config import Plot_config


@typechecked
def update_node_colour_and_opacity(
    *,
    dash_figure: go.Figure,
    identified_annotations: List[NamedAnnotation],
    plot_config: Plot_config,
    plotted_graph: nx.DiGraph,
    t: int,
    temporal_node_colours: List,
    temporal_node_opacity: List,
) -> None:
    """Updates the node colour and opacities."""
    # Overwrite annotation with function instead of value.
    if plot_config.update_edge_colours:
        for edge in plotted_graph.edges():
            if edge[0] != edge[1]:
                # pylint: disable=R0801
                update_non_recursive_edge_colour(
                    dash_figure=dash_figure,
                    edge=edge,
                    identified_annotations=identified_annotations,
                    plot_config=plot_config,
                    plotted_graph=plotted_graph,
                    t=t,
                    temporal_node_colours=temporal_node_colours,
                    temporal_node_opacity=temporal_node_opacity,
                )


# pylint: disable=R0801
@typechecked
def update_non_recursive_edge_colour(
    *,
    dash_figure: go.Figure,
    edge: Tuple[str, str],
    identified_annotations: List[NamedAnnotation],
    plot_config: Plot_config,
    plotted_graph: nx.DiGraph,
    t: int,
    temporal_node_colours: List,
    temporal_node_opacity: List,
) -> None:
    """Updates the colour of the non-recursive edges."""
    edge_annotation_colour = get_edge_colour(
        edge=edge,
        plotted_graph=plotted_graph,
        t=t,
        temporal_node_colours=temporal_node_colours,
    )
    edge_opacity = get_edge_opacity(
        edge=edge,
        plotted_graph=plotted_graph,
        t=t,
        temporal_node_opacity=temporal_node_opacity,
    )

    for i, id_anno in enumerate(identified_annotations):
        if id_anno.edge == edge:
            # TODO: determine why this does not update the dash plt.
            # id_anno.arrowcolor = the_edge_annotation_colour
            # id_anno.arrowcolor = the_edge_annotation_colour

            if plot_config.update_edge_colours:
                # TODO: find method to be sure the annotation
                dash_figure.layout.annotations[
                    i
                ].arrowcolor = edge_annotation_colour
            if plot_config.update_edge_opacity:
                dash_figure.layout.annotations[i].opacity = edge_opacity


@typechecked
def get_edge_colour(
    *,
    edge: Tuple[str, str],
    plotted_graph: nx.DiGraph,
    t: int,
    temporal_node_colours: List,
) -> str:
    """Returns the color of an edge arrow at time t."""
    for i, node_name in enumerate(
        list(some_node_name for some_node_name in plotted_graph.nodes())
    ):
        if node_name == edge[0]:
            return temporal_node_colours[i][t]
    # pylint: disable=W0631
    raise ValueError(f"Error, node_name:{node_name} not found.")


# Update the annotation colour.
def get_edge_opacity(
    *,
    edge: Tuple[str, str],
    plotted_graph: nx.DiGraph,
    t: int,
    temporal_node_opacity: List,
) -> str:
    """Returns the opacity of an edge arrow at time t."""
    for i, node_name in enumerate(
        list(some_node_name for some_node_name in plotted_graph.nodes())
    ):
        if node_name == edge[0]:
            return temporal_node_opacity[i][t]
    # pylint: disable=W0631
    raise ValueError(f"Error, node_name:{node_name} not found.")


@typechecked
def update_node_colour(
    *,
    dash_figure: go.Figure,
    plot_config: Plot_config,
    plotted_graph: nx.DiGraph,
    t: int,
) -> None:
    """Updates the colour of the non-recursive edges."""
    if plot_config.update_node_colours:
        dash_figure.data[0]["marker"]["color"] = list(
            f'{plotted_graph.nodes[n]["temporal_node_colours"][t]}'
            for n in plotted_graph.nodes()
        )


@typechecked
def limit_line_length(
    *, line_separation_chars: str, some_str: str, limit: int
) -> str:
    """Returns first <limit> lines of a string. Assumes new line character is:

     \n
    .
    """
    if some_str.count(line_separation_chars) <= limit:
        return some_str
    split_lines: List[str] = some_str.split(line_separation_chars)
    merged_lines: List[str] = []
    for i in range(0, min(limit, len(split_lines))):
        merged_lines.append(
            os.linesep.join([split_lines[i], line_separation_chars])
        )
    return os.linesep.join(merged_lines)


@typechecked
def update_node_hovertext(
    *,
    dash_figure: go.Figure,
    plot_config: Plot_config,
    plotted_graph: nx.DiGraph,
    t: int,
) -> None:
    """Updates the colour of the non-recursive edges."""
    if plot_config.update_node_labels:
        hovertexts: List[str] = []
        for n in plotted_graph.nodes():
            # Specify the text that is shown when mouse hovers over
            # node.
            hovertext: str = plotted_graph.nodes[n]["temporal_node_hovertext"][
                t
            ]

            limited_hovertext = limit_line_length(
                line_separation_chars="<br />",
                some_str=hovertext,
                limit=25,
            )

            # Add hovertext per node to hovertext list.
            hovertexts.append(limited_hovertext)

        dash_figure.data[0].update(
            hovertext=hovertexts,  # hoverlabel=dict(namelength=-1)
        )


@typechecked
def create_app_layout(
    *,
    app: dash.Dash,
    dash_figures: Dict[str, go.Figure],
    plotted_graphs: Dict[str, nx.DiGraph],
    temporal_node_colours_dict: Dict[str, List],
) -> dash.Dash:
    """Creates the app layout."""
    html_figures: List = []
    for graph_name in plotted_graphs.keys():
        # Create html figures with different id's.
        html_figures.append(
            dcc.Slider(
                id=f"color-set-slider{graph_name}",
                min=0,
                max=len(temporal_node_colours_dict[graph_name][0]) - 1,
                value=0,
                marks={
                    i: str(i)
                    for i in range(
                        len(temporal_node_colours_dict[graph_name][0])
                    )
                },
                step=None,
            )
        )
        html_figures.append(
            html.Div(
                dcc.Graph(
                    id=f"Graph{graph_name}", figure=dash_figures[graph_name]
                )
            )
        )

    # Store html graphs in layout.
    app.layout = html.Div(html_figures)
    return app


@typechecked
def support_updates(
    *,
    app: dash.Dash,
    dash_figures: Dict[str, go.Figure],
    identified_annotations_dict: Dict[str, List[NamedAnnotation]],
    plot_config: Plot_config,
    plotted_graphs: Dict[str, nx.DiGraph],
    temporal_node_colours_dict: Dict[str, List],
    temporal_node_opacity_dict: Dict[str, List],
) -> None:
    """Allows for updating of the various graphs."""
    # State variable to keep track of current color set
    initial_t = 0

    graph_name_one = "adapted_snn_graph"
    if graph_name_one in plotted_graphs.keys():

        @app.callback(
            Output(f"Graph{graph_name_one}", "figure"),
            [Input(f"color-set-slider{graph_name_one}", "value")],
        )
        def update_color_one(
            t: int,
        ) -> go.Figure:
            # ) -> None:
            """Updates the colour of the nodes and edges based on user
            input."""
            if len(temporal_node_colours_dict[graph_name_one][0]) == 0:
                raise ValueError(
                    "Not enough timesteps were found. probably took timestep "
                    + "of ignored node."
                )

            update_node_colour_and_opacity(
                dash_figure=dash_figures[graph_name_one],
                identified_annotations=identified_annotations_dict[
                    graph_name_one
                ],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_one],
                t=t,
                temporal_node_colours=temporal_node_colours_dict[
                    graph_name_one
                ],
                temporal_node_opacity=temporal_node_opacity_dict[
                    graph_name_one
                ],
            )

            update_node_colour(
                dash_figure=dash_figures[graph_name_one],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_one],
                t=t,
            )
            update_node_hovertext(
                dash_figure=dash_figures[graph_name_one],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_one],
                t=t,
            )
            return dash_figures[graph_name_one]

        update_color_one(
            t=initial_t,
        )

    # Manual copy
    graph_name_two = "rad_adapted_snn_graph"
    if graph_name_two in plotted_graphs.keys():

        @app.callback(
            Output(f"Graph{graph_name_two}", "figure"),
            [Input(f"color-set-slider{graph_name_two}", "value")],
        )
        def update_color_two(
            t: int,
        ) -> go.Figure:
            # ) -> None:
            """Updates the colour of the nodes and edges based on user
            input."""
            if len(temporal_node_colours_dict[graph_name_two][0]) == 0:
                raise ValueError(
                    "Not enough timesteps were found. probably took timestep "
                    + "of ignored node."
                )

            update_node_colour_and_opacity(
                dash_figure=dash_figures[graph_name_two],
                identified_annotations=identified_annotations_dict[
                    graph_name_two
                ],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_two],
                t=t,
                temporal_node_colours=temporal_node_colours_dict[
                    graph_name_two
                ],
                temporal_node_opacity=temporal_node_opacity_dict[
                    graph_name_two
                ],
            )

            update_node_colour(
                dash_figure=dash_figures[graph_name_two],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_two],
                t=t,
            )
            update_node_hovertext(
                dash_figure=dash_figures[graph_name_two],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_two],
                t=t,
            )
            return dash_figures[graph_name_two]

        update_color_two(
            t=initial_t,
        )
