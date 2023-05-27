"""Computes what the failure modes were, and then stores this data in the
graphs."""
from typing import Dict, List

import dash
import dash_daq as daq
import pandas as pd
from dash import Input, Output, dcc, html
from dash.dcc.Markdown import Markdown
from pandas import DataFrame
from snnalgorithms.sparse.MDSA.alg_params import get_algorithm_setting_name
from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.process_results.helper import (
    convert_failure_modes_to_table_dict,
    convert_table_dict_to_table,
    get_adaptation_names,
)
from snncompare.process_results.Table_settings import (
    Failure_mode_entry,
    Table_settings,
)
from snncompare.run_config.Run_config import Run_config


# pylint: disable=R0914
@typechecked
def show_failures(
    *,
    exp_config: Exp_config,
    run_configs: List[Run_config],
    # snn_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> None:
    """Shows table with neuron differences between radiated and unradiated
    snns."""
    table_settings: Table_settings = Table_settings(
        exp_config=exp_config,
        run_configs=run_configs,
    )

    app = dash.Dash(__name__)
    app.scripts.config.serve_locally = True

    @typechecked
    def update_table(
        *,
        algorithm_setting: str,
        first_timestep_only: bool,
        seed: int,
        graph_size: int,
        table_settings: Table_settings,
        show_run_configs: bool,
        show_spike_failures: bool,
    ) -> DataFrame:
        """Updates the displayed table."""
        adaptation_names: List[str] = get_adaptation_names(
            run_configs=run_configs
        )

        failure_mode_entries: List[
            Failure_mode_entry
        ] = table_settings.get_failure_mode_entries(
            first_timestep_only,
            seed=seed,
            graph_size=graph_size,
            algorithm_setting=algorithm_setting,
            show_spike_failures=show_spike_failures,
        )

        table_dict: Dict[
            int, Dict[str, List[str]]
        ] = convert_failure_modes_to_table_dict(
            failure_mode_entries=failure_mode_entries,
            show_run_configs=show_run_configs,
        )

        table: List[List[str]] = convert_table_dict_to_table(
            adaptation_names=adaptation_names,
            table=table_dict,
        )
        updated_df = pd.DataFrame.from_records(table)
        return updated_df

    # table,columns=update_table(seed=8,graph_size=3)
    # df,columns=update_table(seed=8,graph_size=3)
    initial_df = update_table(
        algorithm_setting=get_algorithm_setting_name(
            algorithm_setting=run_configs[0].algorithm
        ),
        first_timestep_only=True,
        graph_size=run_configs[0].graph_size,
        table_settings=table_settings,
        seed=run_configs[0].seed,
        show_run_configs=False,
        show_spike_failures=True,
    )
    app.layout = html.Div(
        [
            # Include dropdown
            "Algorithm setting:",
            dcc.Dropdown(
                table_settings.algorithm_setts,
                table_settings.algorithm_setts[0],
                id="alg_setting_selector_id",
            ),
            "Pseudo-random seed:",
            dcc.Dropdown(
                table_settings.seeds,
                table_settings.seeds[0],
                id="seed_selector_id",
            ),
            html.Div(id="seed-selector"),
            "Graph size:",
            dcc.Dropdown(
                table_settings.graph_sizes,
                table_settings.graph_sizes[0],
                id="graph_size_selector_id",
            ),
            html.Div(id="graph-size-selector"),
            html.Div(
                [
                    html.Div(
                        [
                            html.P(
                                children=[
                                    html.Strong("Missing spike/u"),
                                    html.Span(
                                        " -   (w.r.t unradiated adapted snn)",
                                    ),
                                    html.Br(),
                                    html.U("Extra spike/u"),
                                    html.Span(
                                        " - (w.r.t unradiated adapted snn)",
                                    ),
                                    html.Br(),
                                    html.Span(
                                        "Radiated adaptation failed",
                                        style={"color": "red"},
                                    ),
                                    html.Br(),
                                    html.Span(
                                        "Radiated adaptation passed",
                                        style={"color": "green"},
                                    ),
                                    html.Br(),
                                ]
                            ),
                        ]
                    )
                ]
            ),
            html.Div(
                [
                    # pylint: disable=E1102
                    daq.BooleanSwitch(
                        id="show_run_configs",
                        label=(
                            "Show: run_config unique_id's (On) / neuron "
                            + "names (Off),"
                        ),
                        on=False,
                    ),
                    html.Div(id="show_run_configs_div"),
                ]
            ),
            html.Div(
                [
                    # pylint: disable=E1102
                    daq.BooleanSwitch(
                        id="show_spike_failures",
                        label=(
                            "Show neurons with: spike (on) / u (off) "
                            + "differences"
                        ),
                        on=True,
                    ),
                    html.Div(id="show_spike_failures_div"),
                ]
            ),
            html.Div(
                [
                    # pylint: disable=E1102
                    daq.BooleanSwitch(
                        id="first_timestep_only",
                        label="Show deviation of first timestep only",
                        on=True,
                    ),
                    html.Div(id="first_timestep_only_div"),
                ]
            ),
            html.Br(),
            html.Div(
                id="table",
                children=[
                    dcc.Markdown(
                        dangerously_allow_html=True,
                        children=initial_df.to_html(escape=False),
                    )
                ],
            ),
        ]
    )

    @app.callback(
        [
            Output("table", "children"),
            Output("show_run_configs_div", "children"),
        ],
        # [Output("table", "data")],
        [
            Input("alg_setting_selector_id", "value"),
            Input("seed_selector_id", "value"),
            Input("graph_size_selector_id", "value"),
            Input("show_run_configs", "on"),
            Input("show_spike_failures", "on"),
            Input("first_timestep_only", "on"),
        ],
    )
    @typechecked
    # pylint: disable=R0913
    def update_output(
        algorithm_setting: str,
        seed: int,
        graph_size: int,
        show_run_configs: bool,
        show_spike_failures: bool,
        first_timestep_only: bool,
    ) -> List[Markdown]:
        """Updates the table with failure modes based on the user settings."""
        print(
            f"algorithm_setting={algorithm_setting}"
            + f"seed={seed}, graph_size={graph_size}, show_run_configs="
            + f"{show_run_configs}, show_spike_failures={show_spike_failures}"
            + f"first_timestep_only={first_timestep_only}"
        )

        new_df = update_table(
            algorithm_setting=algorithm_setting,
            first_timestep_only=first_timestep_only,
            graph_size=graph_size,
            table_settings=table_settings,
            seed=seed,
            show_run_configs=show_run_configs,
            show_spike_failures=show_spike_failures,
        )
        return [
            dcc.Markdown(
                dangerously_allow_html=True,
                children=new_df.to_html(escape=False),
            ),
            show_run_configs,
        ]

    app.run_server(port=8053)
