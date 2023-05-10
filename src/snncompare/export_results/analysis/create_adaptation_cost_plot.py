"""Creates the different plot data objects."""
from typing import Dict, List

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_results.analysis.adaptation_cost import (
    Adaptation_cost_plot_data,
)
from snncompare.export_results.analysis.get_adaptation_cost_settings import (
    Cost_plot_group,
    Raw_adap_cost_data,
    get_raw_adap_cost_datas,
)


@typechecked
def plot_raw_adap_cost_datas(exp_config: Exp_config) -> None:
    """Creates raw plot data objects."""
    raw_adap_cost_datas: List[
        Raw_adap_cost_data
    ] = get_used_raw_adap_cost_datas(
        exp_config=exp_config, plot_type="per_algorithm"
    )

    for cost_type in raw_adap_cost_datas[0].cost_types:
        adaptation_cost_plot_data: Adaptation_cost_plot_data = (
            get_adaptation_cost_groups_from_incoming_adaptation_cost_datas(
                cost_type=cost_type, raw_adap_cost_datas=raw_adap_cost_datas
            )
        )

        adaptation_cost_plot_data.plot_adaptation_costs()


@typechecked
def get_adaptation_types(
    raw_adap_cost_datas: List[Raw_adap_cost_data],
) -> List[str]:
    """Returns the adaptation types in a list of Raw_adap_cost_data."""
    adaptation_types: List[str] = []
    for raw_adap_cost_data in raw_adap_cost_datas:
        if raw_adap_cost_data.adaptation is None:
            adaptation_group: str = "no_adaptation"

        else:
            adaptation_group = (
                f"{raw_adap_cost_data.adaptation.adaptation_type}"
                + f"_{raw_adap_cost_data.adaptation.redundancy}"
            )
        if adaptation_group not in adaptation_types:
            adaptation_types.append(adaptation_group)
    return adaptation_types


@typechecked
def get_graph_sizes(
    raw_adap_cost_datas: List[Raw_adap_cost_data],
) -> List[int]:
    """Returns the adaptation types in a list of Raw_adap_cost_data."""
    graph_sizes: List[int] = []
    for raw_adap_cost_data in raw_adap_cost_datas:
        if raw_adap_cost_data.graph_size not in graph_sizes:
            graph_sizes.append(raw_adap_cost_data.graph_size)
    return graph_sizes


@typechecked
def get_adaptation_cost_groups_from_incoming_adaptation_cost_datas(
    raw_adap_cost_datas: List[Raw_adap_cost_data],
    cost_type: str,
) -> Adaptation_cost_plot_data:
    """Converts the list of incoming adaptation costs objects into groups of
    adaptation cost coordinates."""

    cost_plot_groups: Dict[str, Cost_plot_group] = {}
    print(f"cost_type={cost_type}")
    for adaptation_type in get_adaptation_types(
        raw_adap_cost_datas=raw_adap_cost_datas
    ):  # both type and value
        y_values: List[float] = []

        # Create a list of input_graph sizes, and per input_graph size, a list
        # of y-coordinates that represent the cost in [neurons/synapses/spikes]
        xy_coords_per_adaptation_type: Dict[float, List[float]] = {}
        for graph_size in get_graph_sizes(raw_adap_cost_datas):
            for raw_adap_cost_data in raw_adap_cost_datas:
                if raw_adap_cost_data.adaptation is None:
                    adaptation_name: str = "no_adaptation"
                else:
                    adaptation_name = (
                        f"{raw_adap_cost_data.adaptation.adaptation_type}"
                        + f"_{raw_adap_cost_data.adaptation.redundancy}"
                    )
                if adaptation_name == adaptation_type:
                    y_values.append(raw_adap_cost_data.costs[cost_type])
            xy_coords_per_adaptation_type[graph_size] = y_values

        cost_plot_groups[adaptation_type] = Cost_plot_group(
            adaptation_type=adaptation_type,
            xy_coords_per_adaptation_type=xy_coords_per_adaptation_type,
        )

    # Convert xy_coordinates dict to list of xy_coordinates.
    adaptation_costs: List = []
    for key, value in cost_plot_groups.items():
        adaptation_costs.append(value)
        print(f"{key}, {value.xy_coords_per_adaptation_type[3]}")

    # TODO: move into parent function (to add algorithm params).
    adaptation_cost_plot_data: Adaptation_cost_plot_data = (
        Adaptation_cost_plot_data(
            adaptation_costs=adaptation_costs,
            cost_type=cost_type,
            title=(
                f"Adaptation costs in terms of: {cost_type}" + " overhead."
            ),
            algorithm_name=raw_adap_cost_datas[0].algorithm_name,
            algorithm_parameter=raw_adap_cost_datas[0].algorithm_parameter,
        )
    )

    return adaptation_cost_plot_data


@typechecked
def get_used_raw_adap_cost_datas(
    *, exp_config: Exp_config, plot_type: str
) -> List[Raw_adap_cost_data]:
    """Gets the adaptation cost objects for all runs in the Experiment setup,
    and then filters them based on the desired plot type setting.

    TODO: build support for per algorithm parameter.
    """
    raw_adap_cost_datas: List[Raw_adap_cost_data] = get_raw_adap_cost_datas(
        exp_config=exp_config
    )
    if plot_type == "per_algorithm":
        used_raw_adap_cost_datas = per_algorithm(
            exp_config=exp_config,
            raw_adap_cost_datas=raw_adap_cost_datas,
        )
    if plot_type == "per_alg_param":
        per_alg_param()
    if plot_type == "per_adaptation_type":
        per_adaptation_type()
    if plot_type == "per_adaptation_value":
        per_adaptation_value()
    return used_raw_adap_cost_datas


@typechecked
def per_algorithm(
    exp_config: Exp_config, raw_adap_cost_datas: List[Raw_adap_cost_data]
) -> List[Raw_adap_cost_data]:
    """Creates a plot of the adaptation costs.

    The costs for all algorithm settings are put in a single plot.
    """
    # used_raw_adap_cost_datas: List[Raw_adap_cost_data]
    for algorithm_name in exp_config.algorithms.keys():
        used_raw_adap_cost_datas: List[Raw_adap_cost_data] = []
        for raw_adap_cost_data in raw_adap_cost_datas:
            if raw_adap_cost_data.algorithm_name == algorithm_name:
                used_raw_adap_cost_datas.append(raw_adap_cost_data)

    return used_raw_adap_cost_datas


@typechecked
def per_alg_param() -> None:
    """Creates a plot of the adaptation costs.

    The costs of adaptation are put in a separate plot per algorithm
    parameter.
    """


@typechecked
def per_adaptation_type() -> None:
    """Creates a plot of the adaptation costs.

    The costs of adaptation are put in a separate plot per adaptation
    type.
    """


@typechecked
def per_adaptation_value() -> None:
    """Creates a plot of the adaptation costs.

    The costs of adaptation are put in a separate plot per adaptation
    value
    """
