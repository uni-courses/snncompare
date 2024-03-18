"""Computes the adaptation costs in terms of:

- additional neurons for an adaptation type, per graph size.
- additional synapses for an adaptation type, per graph size.
- additional spikes for an adaptation type, per graph size.
Then plots this data (in 3 plots) for a given Experiment setting.
"""
from typing import Dict, List, Set, Union

from simplt.dotted_plot.dotted_plot import plot_multiple_dotted_groups
from typeguard import typechecked

from snncompare.export_results.analysis.get_adaptation_cost_settings import (
    Cost_plot_group,
)
from snncompare.import_results.helper import create_relative_path


# pylint: disable = R0903
class Adaptation_cost_plot_data:
    """Stores the data for an adaptation cost plot."""

    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        adaptation_costs: List[Cost_plot_group],
        cost_type: str,
        title: str,
        algorithm_name: str,
        algorithm_parameter: int,
    ) -> None:
        """The constructor for the cost plot object.

        Args:
        :adaptation_costs: (List[Cost_plot_group]), The list of cost
        plot groups.
        :cost_type: (str), The type of cost.
        :title: (str), The title of the plot.
        :algorithm_name: (str), The name of the algorithm used.
        :algorithm_parameter: (int), The parameter of the algorithm
        used.
        """

        self.adaptation_costs: List[Cost_plot_group] = adaptation_costs
        self.cost_type: str = cost_type
        self.title: str = title
        self.algorithm_name: str = algorithm_name
        self.algorithm_parameter: int = algorithm_parameter

    @typechecked
    def get_adaptation_types(
        self,
    ) -> Set[str]:
        """Get adaptation types."""
        adaptation_types: Set[str] = set()
        for adaptation_cost in self.adaptation_costs:
            adaptation_types.add(adaptation_cost.adaptation_type)
        return adaptation_types

    @typechecked
    def plot_adaptation_costs(
        self,
        filename: str,
    ) -> None:
        """Plots the adaptation costs.

        Create a plot (per MDSA value). with on the:
         - x-axis: graph size.
         - y-axis: nr of spikes/neurons/synapses
        as values: 1 dot per unique snn per run config.
        Give each adaptation type (and no adaptation) a separate colour.
        Output the plots to latex/graphs/adaptation_costs/
        <algorithm>/<mdsa_val>
        """
        adaptation_types: Set[str] = self.get_adaptation_types()
        y_series: Dict[Union[float, int, str], Dict[float, List[float]]] = {}
        for adaptation_type in adaptation_types:
            for adaptation_cost in self.adaptation_costs:
                if adaptation_cost.adaptation_type == adaptation_type:
                    y_series[
                        adaptation_type
                    ] = adaptation_cost.xy_coords_per_adaptation_type

        output_dir: str = f"latex/graphs/adaptation_costs/{self.cost_type}/"
        create_relative_path(some_path=output_dir)

        plot_multiple_dotted_groups(
            extensions=[".png"],
            filename=filename,
            label=list(adaptation_types),
            legendPosition=0,
            output_dir=output_dir,
            x_axis_label="input graph size [nodes]",
            y_axis_label=f"Adaptation cost [{self.cost_type}]",
            y_series=y_series,
        )
