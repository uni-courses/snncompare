"""Updates dash plots."""
import os
from typing import Dict, List, Optional, Tuple

import dash
import networkx as nx
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from typeguard import typechecked

from snncompare.export_plots.create_dash_fig_obj import NamedAnnotation
from snncompare.export_plots.Plot_config import Plot_config


@typechecked
def update_edge_colour_and_opacity(
    *,
    dash_figure: go.Figure,
    identified_annotations: List[NamedAnnotation],
    plot_config: Plot_config,
    plotted_graph: nx.DiGraph,
    t: int,
    temporal_node_colours: Dict[str, List],
    temporal_node_opacity: List,
) -> None:
    """Updates the node colour and opacities."""
    # Overwrite annotation with function instead of value.
    if plot_config.update_edge_colours:
        for edge in plotted_graph.edges():
            edge_annotation_colour: str = get_edge_colour(
                edge=edge,
                plotted_graph=plotted_graph,
                t=t,
                temporal_node_colours=temporal_node_colours,
            )
            edge_opacity: float = get_edge_opacity(
                edge=edge,
                plotted_graph=plotted_graph,
                t=t,
                temporal_node_opacity=temporal_node_opacity,
            )
            if edge[0] != edge[1]:
                # pylint: disable=R0801
                update_non_recursive_edge_colour(
                    dash_figure=dash_figure,
                    edge=edge,
                    edge_annotation_colour=edge_annotation_colour,
                    edge_opacity=edge_opacity,
                    identified_annotations=identified_annotations,
                    plot_config=plot_config,
                )


# pylint: disable=R0801
@typechecked
def update_non_recursive_edge_colour(
    *,
    dash_figure: go.Figure,
    edge: Tuple[str, str],
    edge_annotation_colour: str,
    edge_opacity: float,
    identified_annotations: List[NamedAnnotation],
    plot_config: Plot_config,
) -> None:
    """Updates the colour of the non-recursive edges."""

    for i, id_anno in enumerate(identified_annotations):
        if id_anno.edge == edge:
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
    temporal_node_colours: Dict[str, List],
) -> str:
    """Returns the color of an edge arrow at time t."""
    for node_name in list(
        some_node_name for some_node_name in plotted_graph.nodes()
    ):
        rad_color: str = get_radiation_colour(
            plotted_graph=plotted_graph,
            left_node_name=edge[0],
            right_node_name=edge[1],
            t=t,
        )
        if rad_color != "":
            return rad_color

        if node_name == edge[0]:
            return temporal_node_colours[node_name][t]
    # pylint: disable=W0631
    raise ValueError(f"Error, node_name:{node_name} not found.")


# Update the annotation colour.
def get_edge_opacity(
    *,
    edge: Tuple[str, str],
    plotted_graph: nx.DiGraph,
    t: int,
    temporal_node_opacity: List,
) -> float:
    """Returns the opacity of an edge arrow at time t."""
    for i, node_name in enumerate(
        list(some_node_name for some_node_name in plotted_graph.nodes())
    ):
        # Check if synaptic radiation is included and if yes, return that.
        if t in plotted_graph.graph["synaptic_rad_map"].keys():
            if edge in plotted_graph.graph["synaptic_rad_map"][t].keys():
                # TODO: do not hardcode synaptic rad opacity here but in
                # plot_config.
                return 0.8

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
    temporal_node_opacity: List,
) -> None:
    """Updates the colour of the non-recursive edges."""
    if plot_config.update_node_colours:
        dash_figure.data[0]["marker"]["color"] = list(
            f'{plotted_graph.nodes[n]["temporal_colour"][t]}'
            for n in plotted_graph.nodes()
        )
        # Update recursive edges.
        for i, node_name in enumerate(plotted_graph.nodes()):
            edge_opacity: float = get_edge_opacity(
                edge=(node_name, node_name),
                plotted_graph=plotted_graph,
                t=t,
                temporal_node_opacity=temporal_node_opacity,
            )

            l_col: str = get_radiation_colour(
                plotted_graph=plotted_graph,
                left_node_name=node_name,
                right_node_name=node_name,
                t=t,
                left_only=True,
            )
            if l_col == "":
                l_col = (
                    f'{plotted_graph.nodes[node_name]["temporal_colour"][t]}'
                )

            dash_figure.layout.shapes[i].update(
                line_color=l_col,
                opacity=edge_opacity,
            )

    else:
        print("Did not update node colours.")


@typechecked
def get_radiation_colour(
    *,
    plotted_graph: nx.DiGraph,
    left_node_name: str,
    right_node_name: str,
    t: int,
    left_only: Optional[bool] = False,
) -> str:
    """Returns first <limit> lines of a string. Assumes new line character is:

     \n
    .
    """
    # TODO: inspect why not all recursive nodes of radiated neuron become red.
    if plotted_graph.graph["radiation"].effect_type == "neuron_death":
        if t in plotted_graph.graph["synaptic_rad_map"].keys():
            if left_only:
                left_synapses: List[str] = list(
                    map(
                        lambda edge: edge[0],
                        plotted_graph.graph["synaptic_rad_map"][t].keys(),
                    )
                )
                if left_node_name in left_synapses:
                    return "rgb(255, 0, 0)"
            else:
                if (left_node_name, right_node_name) in plotted_graph.graph[
                    "synaptic_rad_map"
                ][t].keys():
                    # TODO: do not hardcode red as synaptic rad colour here
                    # but in plot_config.
                    return "rgb(255, 0, 0)"

    elif (
        plotted_graph.graph["radiation"].effect_type
        == "change_synaptic_weight"
    ):
        # Check if synaptic radiation is included and if yes, return that.
        if t in plotted_graph.graph["synaptic_rad_map"].keys():
            if (left_node_name, right_node_name) in plotted_graph.graph[
                "synaptic_rad_map"
            ][t].keys():
                # TODO: do not hardcode yellow as synaptic rad colour here
                # but in plot_config.
                return "rgb(255, 255, 0)"
    return ""


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
    temporal_node_colours_dict: Dict[str, Dict[str, List]],
) -> dash.Dash:
    """Creates the app layout."""
    html_figures: List = []

    for graph_name in plotted_graphs.keys():
        sim_duration: int = get_sim_duration_from_node_colours(
            graph_name=graph_name,
            temporal_node_colours_dict=temporal_node_colours_dict,
        )

        # Create html figures with different id's.
        html_figures.append(
            dcc.Slider(
                id=f"color-set-slider{graph_name}",
                min=0,
                max=sim_duration - 1,
                value=0,
                marks={i: str(i) for i in range(sim_duration)},
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
def get_sim_duration_from_node_colours(
    *, graph_name: str, temporal_node_colours_dict: Dict[str, Dict[str, List]]
) -> int:
    """Returns the simulation duration from the node colours dict."""
    # Get the simulation duration.
    first_node_name: str = list(temporal_node_colours_dict[graph_name].keys())[
        0
    ]
    nr_of_timesteps: int = len(
        temporal_node_colours_dict[graph_name][first_node_name]
    )
    return nr_of_timesteps


@typechecked
def support_updates(
    *,
    app: dash.Dash,
    dash_figures: Dict[str, go.Figure],
    identified_annotations_dict: Dict[str, List[NamedAnnotation]],
    plot_config: Plot_config,
    plotted_graphs: Dict[str, nx.DiGraph],
    temporal_node_colours_dict: Dict[str, Dict[str, List]],
    temporal_node_opacity_dict: Dict[str, List],
) -> None:
    """Allows for updating of the various graphs."""
    # State variable to keep track of current color set
    initial_t = 0

    graph_name_one = "snn_algo_graph"
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
            sim_duration: int = get_sim_duration_from_node_colours(
                graph_name=graph_name_one,
                temporal_node_colours_dict=temporal_node_colours_dict,
            )
            if sim_duration == 0:
                raise ValueError(
                    "Not enough timesteps were found. probably took timestep "
                    + "of ignored node."
                )

            update_edge_colour_and_opacity(
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
                temporal_node_opacity=temporal_node_opacity_dict[
                    graph_name_one
                ],
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

    graph_name_two = "adapted_snn_graph"
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
            sim_duration: int = get_sim_duration_from_node_colours(
                graph_name=graph_name_two,
                temporal_node_colours_dict=temporal_node_colours_dict,
            )
            if sim_duration == 0:
                raise ValueError(
                    "Not enough timesteps were found. probably took timestep "
                    + "of ignored node."
                )

            update_edge_colour_and_opacity(
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
                temporal_node_opacity=temporal_node_opacity_dict[
                    graph_name_two
                ],
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

    # Manual copy
    graph_name_four = "rad_adapted_snn_graph"
    if graph_name_four in plotted_graphs.keys():

        @app.callback(
            Output(f"Graph{graph_name_four}", "figure"),
            [Input(f"color-set-slider{graph_name_four}", "value")],
        )
        def update_color_four(
            t: int,
        ) -> go.Figure:
            # ) -> None:
            """Updates the colour of the nodes and edges based on user
            input."""
            sim_duration: int = get_sim_duration_from_node_colours(
                graph_name=graph_name_four,
                temporal_node_colours_dict=temporal_node_colours_dict,
            )
            if sim_duration == 0:
                raise ValueError(
                    "Not enough timesteps were found. probably took timestep "
                    + "of ignored node."
                )

            update_edge_colour_and_opacity(
                dash_figure=dash_figures[graph_name_four],
                identified_annotations=identified_annotations_dict[
                    graph_name_four
                ],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_four],
                t=t,
                temporal_node_colours=temporal_node_colours_dict[
                    graph_name_four
                ],
                temporal_node_opacity=temporal_node_opacity_dict[
                    graph_name_four
                ],
            )

            update_node_colour(
                dash_figure=dash_figures[graph_name_four],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_four],
                t=t,
                temporal_node_opacity=temporal_node_opacity_dict[
                    graph_name_four
                ],
            )
            update_node_hovertext(
                dash_figure=dash_figures[graph_name_four],
                plot_config=plot_config,
                plotted_graph=plotted_graphs[graph_name_four],
                t=t,
            )
            return dash_figures[graph_name_four]

        update_color_four(
            t=initial_t,
        )
