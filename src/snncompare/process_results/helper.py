"""Computes what the failure modes were, and then stores this data in the
graphs."""
from typing import TYPE_CHECKING, Dict, List, Union

from typeguard import typechecked

from snncompare.import_results.load_stage4 import load_stage4_results
from snncompare.run_config.Run_config import Run_config

# from dash.dependencies import Input, Output
if TYPE_CHECKING:
    from snncompare.process_results.Table_settings import Failure_mode_entry


@typechecked
def append_failure_mode(
    first_timestep_only: bool,
    failure_mode_entry: "Failure_mode_entry",
    failure_mode_entries: List["Failure_mode_entry"],
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
    failure_mode_entries: List["Failure_mode_entry"],
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

        table[failure_mode.timestep][failure_mode.adaptation_name].append(
            cell_element
        )
    return table


@typechecked
def apply_cell_formatting(
    *,
    failure_mode: "Failure_mode_entry",
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
    """Converts a table dict to a table in format: lists of lists."""
    # Create 2d matrix.
    rows: List[List[Union[List[str], str]]] = []

    # Create the header. Assume adapted only.
    # TODO: build support for unadapted failure modes.
    header: List = ["Timestep"] + adaptation_names
    rows.append(header)

    # Fill the remaining set of the columns.
    for timestep, failure_modes in table.items():
        new_row: List[Union[List[str], str]] = [str(timestep)]
        for adaptation_name in adaptation_names:
            if adaptation_name not in failure_modes.keys():
                new_row.append("")
            else:
                new_row.append(failure_modes[adaptation_name])
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
