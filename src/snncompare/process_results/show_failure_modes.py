"""Computes what the failure modes were, and then stores this data in the
graphs."""

from pprint import pprint
from typing import Dict, List, Optional, Tuple

import dash
import numpy as np
import pandas as pd
from dash import Input, Output, dcc, html

# from dash.dependencies import Input, Output
from dash_table import DataTable
from typeguard import typechecked

from snncompare.exp_config import Exp_config


# pylint: disable=R0914
@typechecked
def show_failures(
    *,
    exp_config: Exp_config,
    # snn_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> None:
    """Filler https://community.plotly.com/t/update-a-dash-datatable-with-
    callbacks/21382."""

    # Dropdown options.
    seeds = exp_config.seeds
    print("seeds")
    pprint(seeds)

    graph_sizes = list(
        map(
            lambda size_and_max_graphs: size_and_max_graphs[0],
            exp_config.size_and_max_graphs,
        )
    )
    print("graph_sizes")
    pprint(graph_sizes)

    algorithm_setts = []
    for algorithm_name, algo_specs in exp_config.algorithms.items():
        for algo_config in algo_specs:
            algorithm = {algorithm_name: algo_config}
            algorithm_setts.append(algorithm)
    print("algorithm_setts")
    pprint(algorithm_setts)

    adaptation_names = []
    for adaptation in exp_config.adaptations:
        adaptation_names.append(
            f"{adaptation.adaptation_type}_{adaptation.redundancy}"
        )
    print("adaptation_names")
    pprint(adaptation_names)

    # Columns:
    new_columns = ["t"] + adaptation_names

    print("new_columns")
    pprint(new_columns)

    data_one = [
        [0, "spike_once_0", "degree_receiver_0"],
        [6, "spike_once_1", "degree_receiver_1"],
        [7, "spike_once_2", "degree_receiver_2"],
        [9, "spike_once_3", "degree_receiver_3"],
        [55, "spike_once_5", "degree_receiver_5"],
    ]

    data_two = [
        [0, "swag", "degree_receiver_0"],
        [6, "swag", "degree_receiver_1"],
        [7, "swag", "degree_receiver_2"],
        [9, "swag", "degree_receiver_3"],
        [55, "swag", "degree_receiver_5"],
    ]

    app = dash.Dash(__name__)
    app.scripts.config.serve_locally = True

    app.layout = html.Div(
        [
            html.Button(["Update"], id="btn"),
            DataTable(id="table", data=[]),
            # Include dropdown
            dcc.Dropdown(["NYC", "MTL", "SF"], "NYC", id="seed_selector_id"),
            html.Div(id="seed-selector"),
            dcc.Dropdown(
                ["aaa", "bb", "cc"], "NYC", id="graph_size_selector_id"
            ),
            html.Div(id="graph-size-selector"),
        ]
    )

    columns = [
        {"id": 0, "name": "Complaint ID"},
        {"id": 1, "name": "Product"},
        {"id": 2, "name": "Sub-product"},
    ]

    @app.callback(
        [Output("table", "data"), Output("table", "columns")],
        [Input("btn", "n_clicks")],
    )
    @typechecked
    def updateTable(
        n_clicks: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        if n_clicks is None:
            dashboard = pd.DataFrame(data_one, columns=columns)
            print(f"type={type(dashboard.values[0:3])}")
            print(f"dashboard.values={dashboard.values[0:3]}")
            return dashboard.values[0:3], columns
        dashboard = pd.DataFrame(data_two, columns=columns)
        return dashboard.values[0:3], columns

    @app.callback(
        Output("seed-selector", "children"), Input("seed_selector_id", "value")
    )
    def update_output(value: str) -> str:
        return f"You have selected seed: {value}"

    @app.callback(
        Output("graph-size-selector", "children"),
        Input("graph_size_selector_id", "value"),
    )
    def update_swag(value: str) -> str:
        return f"You have selected graph_size: {value}"

    app.run_server(port=8053)
