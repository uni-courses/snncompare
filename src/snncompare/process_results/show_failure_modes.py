"""Computes what the failure modes were, and then stores this data in the
graphs."""
import itertools
from collections import OrderedDict
from pprint import pprint
from typing import Dict, List, Tuple, Union

import dash
import dash_daq as daq
import networkx as nx
import pandas as pd
from dash import Input, Output, dcc, html
from dash.dcc.Markdown import Markdown
from pandas import DataFrame
from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.graph_generation.stage_1_create_graphs import (
    load_input_graph_from_file_with_init_props,
)
from snncompare.helper import get_snn_graph_name
from snncompare.import_results.load_stage4 import load_stage4_results
from snncompare.import_results.load_stage_1_and_2 import load_simsnn_graphs
from snncompare.run_config.Run_config import Run_config

# from dash.dependencies import Input, Output


# pylint: disable=R0902
class Failure_mode_entry:
    """Contains a list of neuron names."""

    # pylint: disable=R0913
    # pylint: disable=R0903
    @typechecked
    def __init__(
        self,
        adaptation_name: str,
        incorrectly_spikes: bool,
        incorrectly_silent: bool,
        incorrect_u_increase: bool,
        incorrect_u_decrease: bool,
        neuron_names: List[str],
        run_config: Run_config,
        timestep: int,
    ) -> None:
        """Stores a failure mode entry.

        Args:
            adaptation_name (str): The name of the adaptation.
            incorrectly_spikes (bool): Indicates if the neurons spiked
            incorrectly.
            neuron_names (List[str]): List of neuron names.
            run_config (Run_config): The run configuration.
            timestep (int): The timestep at which the failure mode occurred.
        """
        self.adaptation_name: str = adaptation_name
        self.incorrectly_spikes: bool = incorrectly_spikes
        self.incorrectly_silent: bool = incorrectly_silent
        self.incorrect_u_increase: bool = incorrect_u_increase
        self.incorrect_u_decrease: bool = incorrect_u_decrease
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
        """Stores the Dash failure-mode plot settings.

        Args:
            exp_config (Exp_config): The experiment configuration.
            run_configs (List[Run_config]): List of run configurations.
        """

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
        """Returns the failure mode data for the selected settings.

        Returns:
            A list of tuples containing the run configuration and the failure
            mode data.
        """
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

    # pylint: disable=R0912
    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def get_failure_mode_entries(
        self,
        first_timestep_only: bool,
        seed: int,
        graph_size: int,
        algorithm_setting: str,
        show_spike_failures: bool,
    ) -> List[Failure_mode_entry]:
        """Returns the failure mode data for the selected settings.

        Args:
            seed: The seed value.
            graph_size: The size of the graph.
            algorithm_setting: The algorithm setting.

        Returns:
            A list of failure mode entries.

        Raises:
            ValueError: If the run configurations are not equal.
        """
        failure_mode_entries: List[Failure_mode_entry] = []

        # Loop over the combination of run_config and accompanying snn graphs.
        for run_config, snn_graphs in self.run_config_and_snns:
            # Read the failure modes from the graph object.
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
                if show_spike_failures:
                    if "incorrectly_silent" in failure_mode.keys():
                        for timestep, neuron_list in failure_mode[
                            "incorrectly_silent"
                        ].items():
                            failure_mode_entry: Failure_mode_entry = (
                                Failure_mode_entry(
                                    adaptation_name=adaptation_name,
                                    incorrectly_spikes=False,
                                    incorrectly_silent=True,
                                    incorrect_u_increase=False,
                                    incorrect_u_decrease=False,
                                    neuron_names=neuron_list,
                                    run_config=run_config,
                                    timestep=int(timestep),
                                )
                            )
                            append_failure_mode(
                                first_timestep_only=first_timestep_only,
                                failure_mode_entry=failure_mode_entry,
                                failure_mode_entries=failure_mode_entries,
                            )
                    if "incorrectly_spikes" in failure_mode.keys():
                        for timestep, neuron_list in failure_mode[
                            "incorrectly_spikes"
                        ].items():
                            failure_mode_entry = Failure_mode_entry(
                                adaptation_name=adaptation_name,
                                incorrectly_spikes=True,
                                incorrectly_silent=False,
                                incorrect_u_increase=False,
                                incorrect_u_decrease=False,
                                neuron_names=neuron_list,
                                run_config=run_config,
                                timestep=int(timestep),
                            )
                            append_failure_mode(
                                first_timestep_only=first_timestep_only,
                                failure_mode_entry=failure_mode_entry,
                                failure_mode_entries=failure_mode_entries,
                            )
                else:
                    if "inhibitory_delta_u" in failure_mode.keys():
                        for timestep, neuron_list in failure_mode[
                            "inhibitory_delta_u"
                        ].items():
                            failure_mode_entry = Failure_mode_entry(
                                adaptation_name=adaptation_name,
                                incorrectly_spikes=False,
                                incorrectly_silent=False,
                                incorrect_u_increase=False,
                                incorrect_u_decrease=True,
                                neuron_names=neuron_list,
                                run_config=run_config,
                                timestep=int(timestep),
                            )
                            append_failure_mode(
                                first_timestep_only=first_timestep_only,
                                failure_mode_entry=failure_mode_entry,
                                failure_mode_entries=failure_mode_entries,
                            )
                    if "excitatory_delta_u" in failure_mode.keys():
                        for timestep, neuron_list in failure_mode[
                            "excitatory_delta_u"
                        ].items():
                            failure_mode_entry = Failure_mode_entry(
                                adaptation_name=adaptation_name,
                                incorrectly_spikes=False,
                                incorrectly_silent=False,
                                incorrect_u_increase=True,
                                incorrect_u_decrease=False,
                                neuron_names=neuron_list,
                                run_config=run_config,
                                timestep=int(timestep),
                            )
                            append_failure_mode(
                                first_timestep_only=first_timestep_only,
                                failure_mode_entry=failure_mode_entry,
                                failure_mode_entries=failure_mode_entries,
                            )

        return failure_mode_entries


@typechecked
def append_failure_mode(
    first_timestep_only: bool,
    failure_mode_entry: Failure_mode_entry,
    failure_mode_entries: List[Failure_mode_entry],
) -> None:
    """If first_timestep only, this function checks whether this timestep
    already contains a failure mode for the run_config within the failure mode.

    If yes, it does not add anything, otherwise it adds the failure mode
    to the list of failure modes.
    """
    if first_timestep_only:
        found_entry: bool = False
        for found_failure_mode in failure_mode_entries:
            if (
                failure_mode_entry.run_config.unique_id
                == found_failure_mode.run_config.unique_id
            ):
                found_entry = True
                break
        if not found_entry:
            print(f"add:{failure_mode_entry.run_config.adaptation.__dict__}")
            failure_mode_entries.append(failure_mode_entry)
    else:
        failure_mode_entries.append(failure_mode_entry)


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
    show_run_configs: bool,
) -> Dict[int, Dict[str, List[str]]]:
    """Converts the failure mode dicts into a table.

    It gets the list of timesteps. Then per timestep, creates a
    dictionary with the adaptation names as dictionary and the list of
    lists of neuron names as values.
    """
    failure_mode_entries.sort(key=lambda x: x.timestep, reverse=False)

    table: Dict[int, Dict[str, List[str]]] = {}
    # Get timesteps
    for failure_mode in failure_mode_entries:
        table[failure_mode.timestep] = {}

    # Create the list of neuron_name lists per adaptation type.
    for failure_mode in failure_mode_entries:
        table[failure_mode.timestep][failure_mode.adaptation_name] = []

    # Create the list of neuron_name lists per adaptation type.
    for failure_mode in failure_mode_entries:
        # Apply cell formatting.
        cell_element: str = apply_cell_formatting(
            failure_mode=failure_mode, show_run_configs=show_run_configs
        )

        print(
            f"t={failure_mode.timestep}, "
            + f"failure_mode.adaptation_name={failure_mode.adaptation_name}"
        )
        print(f"cell_element={cell_element}")
        table[failure_mode.timestep][failure_mode.adaptation_name].append(
            cell_element
        )
    return table


@typechecked
def apply_cell_formatting(
    *,
    failure_mode: Failure_mode_entry,
    show_run_configs: bool,
) -> str:
    """Formats the incoming list of neuron names based on the run properties
    and failure mode.

    - Removes the list artifacts (brackets and quotations and commas).
    - If the run was successful, the neuron names become green.
    - If the run was not successful, the neuron names become red.

    - If the neurons spiked when they should not, their names become
    underlined.
    - If the neurons did not spike when they should, their names are bold.
    """

    if not show_run_configs:
        # Determine failure mode formatting.
        if (
            failure_mode.incorrectly_spikes
            or failure_mode.incorrect_u_increase
        ):
            separator_name = "u"

        elif (
            failure_mode.incorrectly_silent
            or failure_mode.incorrect_u_decrease
        ):
            separator_name = "b"
        else:
            raise ValueError("Error, some incorrect behaviour expected.")
        separator = f"</{separator_name}> <br></br> <{separator_name}>"

        # Format the list of strings into an underlined or bold list of names,
        # separated by the newline character in html (<br>).
        cell_element: str = (
            f"<{separator_name}>"
            + separator.join(str(x) for x in failure_mode.neuron_names)
            + f"</{separator_name}>"
            + "<br></br>"
        )
    else:
        cell_element = failure_mode.run_config.unique_id

    # TODO: determine whether run passed or not.
    stage_4_results_dict = load_stage4_results(
        run_config=failure_mode.run_config,
        stage_4_results_dict=None,
    )
    results_dict: Dict[str, Union[float, bool]] = stage_4_results_dict[
        "rad_adapted_snn_graph"
    ].network.graph.graph["results"]
    if results_dict["passed"]:
        cell_element = f'<FONT COLOR="#008000">{cell_element}</FONT>'  # green
    else:
        cell_element = f'<FONT COLOR="#FF0000">{cell_element}</FONT>'  # red

    return cell_element


@typechecked
def convert_table_dict_to_table(
    *,
    adaptation_names: List[str],
    table: Dict[int, Dict[str, List[str]]],
) -> List[List[Union[List[str], str]]]:
    """Converts a table dict to a table in format: lists of lists.

    TODO: fix this.
    """
    print("incoming table=")
    pprint(table)
    # Create 2d matrix.
    rows: List[List[Union[List[str], str]]] = []
    for timestep, failure_modes in table.items():
        new_row: List[Union[List[str], str]] = [str(timestep)]
        for adaptation_name in adaptation_names:
            if adaptation_name not in failure_modes.keys():
                new_row.append("")
            else:
                new_row.append(failure_modes[adaptation_name])
        rows.append(new_row)
    print("outgoing rows=")
    pprint(rows)
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

    @typechecked
    def convert_table_to_ordered_dict(
        adaptation_names: List[str], table: List[List[str]]
    ) -> OrderedDict:
        """Converts the table with rows of: (row header, and a list of row
        data), into an OrderedDict with format:

        List of Tuples[column header, List of column values]
        """
        headers: List[str] = ["t"] + adaptation_names

        # Get remainder of columns.
        columns: Dict[int, List[str]] = {}
        for i in range(0, len(headers)):
            columns[i] = []
        merged_rows = []
        for row in table:
            merged_row = list(itertools.chain.from_iterable(row))
            merged_rows.append(merged_row)

        for row in merged_rows:
            for column_index in range(0, len(headers)):
                if len(row) > column_index:
                    element: str = row[column_index]
                else:
                    element = ""
                columns[column_index].append(element)

        ordered_dict_list = []
        for column_index, header in enumerate(headers):
            ordered_dict_list.append((header, columns[column_index]))
        data = OrderedDict(ordered_dict_list)
        return data

    @typechecked
    def update_table(
        *,
        first_timestep_only: bool,
        seed: int,
        graph_size: int,
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
            algorithm_setting="MDSA_0",
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

        # Convert the ordered dict into a pandas dataframe
        ordered_dict: OrderedDict = convert_table_to_ordered_dict(
            adaptation_names=table_settings.adaptation_names, table=table
        )
        updated_df = pd.DataFrame(ordered_dict)

        return updated_df

    # table,columns=update_table(seed=8,graph_size=3)
    # df,columns=update_table(seed=8,graph_size=3)
    initial_df = update_table(
        first_timestep_only=True,
        graph_size=3,
        seed=8,
        show_run_configs=True,
        show_spike_failures=False,
    )
    app.layout = html.Div(
        [
            # Include dropdown
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
                    # pylint: disable=E1102
                    daq.BooleanSwitch(
                        id="show_run_configs",
                        label="Show run_config unique_id's",
                        on=True,
                    ),
                    html.Div(id="show_run_configs_div"),
                ]
            ),
            html.Div(
                [
                    # pylint: disable=E1102
                    daq.BooleanSwitch(
                        id="show_spike_failures",
                        label="Show neurons with spike (or u) failures",
                        on=False,
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
            Input("seed_selector_id", "value"),
            Input("graph_size_selector_id", "value"),
            Input("show_run_configs", "on"),
            Input("show_spike_failures", "on"),
            Input("first_timestep_only", "on"),
        ],
    )
    @typechecked
    def update_output(
        seed: int,
        graph_size: int,
        show_run_configs: bool,
        show_spike_failures: bool,
        first_timestep_only: bool,
    ) -> List[Markdown]:
        """Updates the table with failure modes based on the user settings."""
        print(
            f"seed={seed}, graph_size={graph_size}, show_run_configs="
            + f"{show_run_configs}, show_spike_failures={show_spike_failures}"
            + f"first_timestep_only={first_timestep_only}"
        )

        new_df = update_table(
            first_timestep_only=first_timestep_only,
            graph_size=graph_size,
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
