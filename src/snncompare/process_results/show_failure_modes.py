"""Computes what the failure modes were, and then stores this data in the
graphs."""

from typing import Dict, List, Tuple, Union

import dash
import networkx as nx
from dash import Input, Output, dcc, html
from dash.dash_table import DataTable
from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.graph_generation.stage_1_create_graphs import (
    load_input_graph_from_file_with_init_props,
)
from snncompare.helper import get_snn_graph_name
from snncompare.import_results.load_stage_1_and_2 import load_simsnn_graphs
from snncompare.run_config.Run_config import Run_config

# from dash.dependencies import Input, Output


class Failure_mode_entry:
    """Contains a list of neuron names."""

    # pylint: disable=R0913
    # pylint: disable=R0903
    @typechecked
    def __init__(
        self,
        adaptation_name: str,
        incorrectly_spikes: bool,
        neuron_names: List[str],
        run_config: Run_config,
        timestep: int,
    ) -> None:
        """Stores a failure mode entry."""
        self.adaptation_name: str = adaptation_name
        self.incorrectly_spikes: bool = incorrectly_spikes
        self.neuron_names: List = neuron_names
        self.run_config: Run_config = run_config
        self.timestep: int = timestep


# pylint: disable=R0903
# pylint: disable=R0902
class Table_settings:
    """Creates the object with the settings for the Dash table."""

    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        run_configs: List[Run_config],
    ) -> None:
        self.exp_config: Exp_config = exp_config
        self.run_configs: List[Run_config] = run_configs
        # Dropdown options.
        self.seeds = exp_config.seeds

        self.graph_sizes = list(
            map(
                lambda size_and_max_graphs: size_and_max_graphs[0],
                exp_config.size_and_max_graphs,
            )
        )

        self.algorithm_setts = []
        for algorithm_name, algo_specs in exp_config.algorithms.items():
            for algo_config in algo_specs:
                algorithm = {algorithm_name: algo_config}
                self.algorithm_setts.append(algorithm)

        self.adaptation_names = []
        for adaptation in exp_config.adaptations:
            self.adaptation_names.append(
                f"{adaptation.adaptation_type}_{adaptation.redundancy}"
            )

        self.run_config_and_snns: List[
            Tuple[Run_config, Dict]
        ] = self.create_failure_mode_tables()

    @typechecked
    def create_failure_mode_tables(
        self,
    ) -> List[Tuple[Run_config, Dict]]:
        """Returns the failure mode data for the selected settings."""
        run_config_and_snns: List[Tuple[Run_config, Dict]] = []
        for run_config in self.run_configs:
            snn_graphs: Dict = {}
            input_graph: nx.Graph = load_input_graph_from_file_with_init_props(
                run_config=run_config
            )

            for with_adaptation in [False, True]:
                for with_radiation in [False, True]:
                    graph_name: str = get_snn_graph_name(
                        with_adaptation=with_adaptation,
                        with_radiation=with_radiation,
                    )
                    snn_graphs[graph_name] = load_simsnn_graphs(
                        run_config=run_config,
                        input_graph=input_graph,
                        with_adaptation=with_adaptation,
                        with_radiation=with_radiation,
                        stage_index=7,
                    )
            run_config_and_snns.append((run_config, snn_graphs))
        return run_config_and_snns

    @typechecked
    def get_failure_mode_entries(
        self,
        seed: int,
        graph_size: int,
        algorithm_setting: str,
        # ) -> List[Dict]:
    ) -> List[Failure_mode_entry]:
        """Returns the failure mode data for the selected settings."""
        failure_mode_entries: List[Failure_mode_entry] = []
        for run_config, snn_graphs in self.run_config_and_snns:
            failure_run_config, failure_mode = (
                run_config,
                snn_graphs["rad_adapted_snn_graph"].network.graph.graph[
                    "failure_modes"
                ],
            )
            if run_config != failure_run_config:
                raise ValueError("Error, run configs not equal.")

            # Check if the run config settings are desired.
            alg_name: str = list(run_config.algorithm.keys())[0]
            alg_param: int = list(run_config.algorithm.values())[0]["m_val"]
            adaptation_name: str = (
                f"{run_config.adaptation.adaptation_type}_"
                + f"{run_config.adaptation.redundancy}"
            )
            if (
                run_config.seed == seed
                and run_config.graph_size == graph_size
                and f"{alg_name}_{alg_param}" == algorithm_setting
            ):
                if "incorrectly_silent" in failure_mode.keys():
                    for timestep, neuron_list in failure_mode[
                        "incorrectly_silent"
                    ].items():
                        failure_mode_entry = Failure_mode_entry(
                            adaptation_name=adaptation_name,
                            incorrectly_spikes=False,
                            neuron_names=neuron_list,
                            run_config=run_config,
                            timestep=int(timestep),
                        )
                        failure_mode_entries.append(failure_mode_entry)
                if "incorrectly_spikes" in failure_mode.keys():
                    for timestep, neuron_list in failure_mode[
                        "incorrectly_spikes"
                    ].items():
                        failure_mode_entry = Failure_mode_entry(
                            adaptation_name=adaptation_name,
                            incorrectly_spikes=True,
                            neuron_names=neuron_list,
                            run_config=run_config,
                            timestep=int(timestep),
                        )
                        failure_mode_entries.append(failure_mode_entry)

        return failure_mode_entries


@typechecked
def get_adaptation_names(
    run_configs: List[Run_config],
    # ) -> List[Dict]:
) -> List[str]:
    """Returns the failure mode data for the selected settings."""
    adaptation_names: List[str] = []
    for run_config in run_configs:
        adaptation_name: str = (
            f"{run_config.adaptation.adaptation_type}_"
            + f"{run_config.adaptation.redundancy}"
        )
        if adaptation_name not in adaptation_names:
            adaptation_names.append(adaptation_name)
    return adaptation_names


# pylint: disable=R0914
@typechecked
def convert_failure_modes_to_table_dict(
    *,
    failure_mode_entries: List[Failure_mode_entry],
) -> Dict[int, Dict[str, List[List[str]]]]:
    """Converts the failure mode dicts into a table.

    It gets the list of timesteps. Then per timestep, creates a
    dictionary with the adaptation names as dictionary and the list of
    lists of neuron names as values.
    """
    failure_mode_entries.sort(key=lambda x: x.timestep, reverse=False)

    table: Dict[int, Dict[str, List[List[str]]]] = {}
    # Get timesteps
    for failure_mode in failure_mode_entries:
        table[failure_mode.timestep] = {}

    # Create the list of neuron_name lists per adaptation type.
    for failure_mode in failure_mode_entries:
        table[failure_mode.timestep][failure_mode.adaptation_name] = []

    # Create the list of neuron_name lists per adaptation type.
    for failure_mode in failure_mode_entries:
        # TODO: make more advanced, e.g. bold/italic,red,green etc.
        cell_list_element: List[str] = failure_mode.neuron_names
        table[failure_mode.timestep][failure_mode.adaptation_name].append(
            cell_list_element
        )
    return table


@typechecked
def convert_table_dict_to_table(
    *,
    adaptation_names: List[str],
    table: Dict[int, Dict[str, List[List[str]]]],
) -> List[List[str]]:
    """Converts a table dict to a table of lists of lists."""

    # Create 2d matrix.
    rows: List[List[str]] = []
    for timestep, failure_modes in table.items():
        new_row: List[str] = [str(timestep)]
        for adaptation_name in adaptation_names:
            if adaptation_name not in failure_modes.keys():
                new_row.append("")
            else:
                new_row.append(str(failure_modes[adaptation_name]))
        rows.append(new_row)
    return rows


# pylint: disable=R0914
@typechecked
def get_table_columns(
    *,
    adaptation_names: List[str],
) -> List[Dict[str, Union[int, str]]]:
    """Creates the table column header."""
    column_header: List[Dict[str, Union[int, str]]] = []

    header: List[str] = ["t"] + adaptation_names
    for i, column_head in enumerate(header):
        column_header.append({"id": i, "name": column_head})
    return column_header


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

    # TODO: get the pass/fail data per run config.
    # Make the text red if the SNN failed.
    # Make the text green if the SNN passed.

    # TODO: Make the text bold if the neuron spiked when it should not.
    # TODO: Make the text italic if the neuron did not spike when it should.

    app = dash.Dash(__name__)
    app.scripts.config.serve_locally = True

    app.layout = html.Div(
        [
            # html.Button(["Update"], id="btn"),
            DataTable(id="table", data=[]),
            # Include dropdown
            dcc.Dropdown(
                table_settings.seeds,
                table_settings.seeds[0],
                id="seed_selector_id",
            ),
            html.Div(id="seed-selector"),
            dcc.Dropdown(
                table_settings.graph_sizes,
                table_settings.graph_sizes[0],
                id="graph_size_selector_id",
            ),
            html.Div(id="graph-size-selector"),
        ]
    )

    @typechecked
    def update_table(
        seed: int, graph_size: int
    ) -> Tuple[List[List[str]], List[Dict[str, Union[int, str]]]]:
        """Updates the displayed table."""
        adaptation_names: List[str] = get_adaptation_names(
            run_configs=run_configs
        )

        failure_mode_entries: List[
            Failure_mode_entry
        ] = table_settings.get_failure_mode_entries(
            seed=seed,
            graph_size=graph_size,
            algorithm_setting="MDSA_0",
        )
        table_dict: Dict[
            int, Dict[str, List[List[str]]]
        ] = convert_failure_modes_to_table_dict(
            failure_mode_entries=failure_mode_entries,
        )

        table: List[List[str]] = convert_table_dict_to_table(
            adaptation_names=adaptation_names,
            table=table_dict,
        )
        columns: List[Dict[str, Union[int, str]]] = get_table_columns(
            adaptation_names=adaptation_names
        )
        return table, columns

    @app.callback(
        [Output("table", "data"), Output("table", "columns")],
        [
            Input("seed_selector_id", "value"),
            Input("graph_size_selector_id", "value"),
        ],
    )
    @typechecked
    def update_output(
        seed: int, graph_size: int
    ) -> Tuple[List[List[str]], List[Dict[str, Union[int, str]]]]:
        print(f"seed={seed}, graph_size={graph_size}")
        return update_table(graph_size=graph_size, seed=seed)

    app.run_server(port=8053)
