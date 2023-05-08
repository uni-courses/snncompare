"""Computes the adaptation costs in terms of:

- additional neurons for an adaptation type, per graph size.
- additional synapses for an adaptation type, per graph size.
- additional spikes for an adaptation type, per graph size.
Then plots this data (in 3 plots) for a given Experiment setting.
"""
import json
import os
from typing import Dict, List, Set, Union

import networkx as nx
from simplt.dotted_plot.dotted_plot import plot_multiple_dotted_groups
from simsnn.core.simulators import Simulator
from snnadaptation.Adaptation import Adaptation
from snnalgorithms.get_input_graphs import (
    add_mdsa_initialisation_properties_to_input_graph,
)
from snnradiation.Rad_damage import Rad_damage
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_plots.Plot_config import get_default_plot_config
from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_rand_nrs_and_hash,
)
from snncompare.graph_generation.export_input_graphs import (
    load_input_graph_based_on_nr,
)
from snncompare.graph_generation.stage_1_create_graphs import (
    get_graphs_stage_1,
)
from snncompare.helper import get_snn_graph_from_graphs_dict
from snncompare.import_results.helper import (
    create_relative_path,
    get_isomorphic_graph_hash,
)
from snncompare.run_config.Run_config import Run_config


# pylint: disable = R0903
class Adaptation_cost:
    """Stores the adaptation costs for an adaptation type."""

    @typechecked
    def __init__(
        self,
        adaptation_type: str,
        adaptation_coords: Dict[float, List[float]],
    ) -> None:
        self.adaptation_type: str = adaptation_type
        self.adaptation_coords: Dict[float, List[float]] = adaptation_coords


# pylint: disable = R0903
class Adaptation_plot_data:
    """Stores the data for an adaptation cost plot."""

    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        adaptation_costs: List[Adaptation_cost],
        cost_type: str,
        title: str,
        algorithm_name: str,
        algorithm_parameter: int,
    ) -> None:
        self.adaptation_costs: List[Adaptation_cost] = adaptation_costs
        self.cost_type: str = cost_type
        self.title: str = title
        self.algorithm_name: str = algorithm_name
        self.algorithm_parameter: int = algorithm_parameter

    @typechecked
    def get_adaptation_types(
        self,
    ) -> Set[str]:
        """Get adaptation types.."""
        adaptation_types: Set[str] = set()
        for adaptation_cost in self.adaptation_costs:
            adaptation_types.add(adaptation_cost.adaptation_type)
        return adaptation_types

    @typechecked
    def plot_adaptation_costs(
        self,
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
                    ] = adaptation_cost.adaptation_coords

            filename: str = (
                f"{self.algorithm_name}_{self.algorithm_parameter}_"
                + f"{self.cost_type}"
            )

            output_dir: str = (
                f"latex/graphs/adaptation_costs/{self.cost_type}/"
            )
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


@typechecked
def create_adapation_cost_plots(*, exp_config: Exp_config) -> None:
    """Creates the adaptation cost plots for the experiment configuration."""
    # Assume required data is available.
    cost_types: List[str] = ["neuronal", "synaptic", "energy"]

    for cost_type in cost_types:
        # Per algorithm setting (e.g. MDSA value:)
        if "MDSA" in exp_config.algorithms.keys():
            algorithm_name: str = "MDSA"
            for m_val_dict in exp_config.algorithms[algorithm_name]:
                m_val: int = m_val_dict["m_val"]
                # for each adaptation type:
                # Get the original SNN for each run config.
                # Get the adapted SNN for each run config.
                # Remove duplicate SNNs from these two lists.

                # Per snn, count: the nr of spikes.
                # Per snn, count: the nr of neurons.
                # Per snn, count: the nr of synapses.
                for with_adaptation in [True, False]:
                    for adaptation in exp_config.adaptations:
                        adaptation_plot_data = Adaptation_plot_data(
                            adaptation_costs=[
                                get_adaptation_cost(
                                    adaptation=adaptation,
                                    algorithm_name=algorithm_name,
                                    algorithm_parameter=m_val,
                                    cost_type=cost_type,
                                    exp_config=exp_config,
                                    with_adaptation=with_adaptation,
                                )
                            ],
                            cost_type=cost_type,
                            title=(
                                f"Adaptation costs in terms of: {cost_type}"
                                + " overhead."
                            ),
                            algorithm_name=algorithm_name,
                            algorithm_parameter=m_val,
                        )
                        adaptation_plot_data.plot_adaptation_costs()
        else:
            raise NotImplementedError("Error, algorithm not (yet) supported.")


# pylint: disable=R0914
@typechecked
def get_adaptation_cost(
    *,
    adaptation: Adaptation,
    algorithm_name: str,
    algorithm_parameter: int,
    cost_type: str,
    exp_config: Exp_config,
    with_adaptation: bool,
) -> Adaptation_cost:
    """Returns the adaptation costs for this experiment setting.

    results/stage2/MDSA_0/no_adaptation/snns
    results/stage2/MDSA_0/no_adaptation/snns
    """

    # TODO: return both.

    dummy_rad_damage: Rad_damage = Rad_damage(
        amplitude=0,
        effect_type="neuron_death",
        excitatory=False,
        inhibitory=False,
        probability_per_t=0.0,
    )

    # Create a list of input_graph sizes, and per input_graph size, a list
    # of y-coordinates that represent the cost in [neurons/synapses/spikes]
    adaptation_coords: Dict[int, List[float]] = {}

    for graph_size, nr_of_graphs in exp_config.size_and_max_graphs:
        # Per graph size, compute the y values/cost.
        y_values: List[float] = []

        # For each graph of a specific size, multiple runs are performed
        # with different random weight- (and radiation) seeds.
        for seed in exp_config.seeds:
            for graph_nr in range(0, nr_of_graphs):
                # Load input graph from file.
                input_graph: nx.Graph = load_input_graph_based_on_nr(
                    graph_size=graph_size, graph_nr=graph_nr
                )
                add_mdsa_initialisation_properties_to_input_graph(
                    input_graph=input_graph, seed=seed
                )

                # Get snn graph with or without adaptation, based on input
                # graph.
                stage_1_graphs: Dict[
                    str, Union[nx.Graph, nx.DiGraph, Simulator]
                ] = get_graphs_stage_1(
                    plot_config=get_default_plot_config(),
                    run_config=Run_config(
                        adaptation=adaptation,
                        algorithm={
                            algorithm_name: {"m_val": algorithm_parameter}
                        },
                        graph_size=graph_size,
                        graph_nr=graph_nr,
                        radiation=dummy_rad_damage,
                        seed=seed,
                        simulator="simsnn",
                    ),
                )

                cost: int = get_snn_complexity(
                    adaptation=adaptation,
                    algorithm_name=algorithm_name,
                    algorithm_parameter=algorithm_parameter,
                    cost_type=cost_type,
                    input_graph=input_graph,
                    stage_1_graphs=stage_1_graphs,
                    with_adaptation=with_adaptation,
                )

                y_values.append(cost)
        adaptation_coords[graph_size] = y_values
    adaptation_costs: Adaptation_cost = Adaptation_cost(
        adaptation_type=adaptation.adaptation_type,
        adaptation_coords=adaptation_coords,
    )

    return adaptation_costs


@typechecked
def get_snn_complexity(
    *,
    adaptation: Adaptation,
    algorithm_name: str,
    algorithm_parameter: int,
    cost_type: str,
    input_graph: nx.Graph,
    stage_1_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    with_adaptation: bool,
) -> int:
    """Returns the complexity of a graph."""

    snn_graph: Union[nx.DiGraph, Simulator] = get_snn_graph_from_graphs_dict(
        with_adaptation=with_adaptation,
        with_radiation=False,
        graphs_dict=stage_1_graphs,
    )
    if cost_type == "energy":
        return get_nr_of_spikes_in_snn(
            adaptation=adaptation,
            algorithm_name=algorithm_name,
            algorithm_parameter=algorithm_parameter,
            input_graph=input_graph,
            stage_1_graphs=stage_1_graphs,
            with_adaptation=with_adaptation,
        )
    if cost_type == "neuronal":
        return len(snn_graph.network.nodes)
    if cost_type == "synaptic":
        return len(snn_graph.network.edges)
    raise NotImplementedError(f"Error:{cost_type} not implemented")


@typechecked
def get_nr_of_spikes_in_snn(
    *,
    adaptation: Adaptation,
    algorithm_name: str,
    algorithm_parameter: int,
    input_graph: nx.Graph,
    stage_1_graphs: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    with_adaptation: bool,
) -> int:
    """Returns the adaptation costs for this experiment setting."""
    _, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)

    # Get the isomorphic hash of the snn graphs.
    if with_adaptation:
        adaptation_name: str = (
            f"{adaptation.adaptation_type}:" + f"{adaptation.redundancy}"
        )
        isomorphic_hash: str = get_isomorphic_graph_hash(
            some_graph=stage_1_graphs["adapted_snn_graph"]
        )
    else:
        adaptation_name = "no_adaptation"
        isomorphic_hash = get_isomorphic_graph_hash(
            some_graph=stage_1_graphs["snn_algo_graph"]
        )

    # Get the filepaths of the snn graph propagation json.
    output_dir = (
        f"results/stage{2}/{algorithm_name}_"
        + f"{algorithm_parameter}/{adaptation_name}/snns/"
    )
    additional_hashes: str = ""
    additional_hashes = f"{additional_hashes}_rand_{rand_nrs_hash}"
    propagation_results_filepath: str = (
        f"{output_dir}{isomorphic_hash}{additional_hashes}" + ".json"
    )
    if not os.path.isfile(propagation_results_filepath):
        raise FileNotFoundError(
            f"Error, {propagation_results_filepath} not found."
        )

    # Read snn graph propagation JSON file into dict.
    with open(propagation_results_filepath, encoding="utf-8") as json_file:
        snn_propagation = json.load(json_file)
        json_file.close()
    nr_of_spikes: int = sum(snn_propagation["spikes"])
    print(f"nr_of_spikes={nr_of_spikes}")
    return nr_of_spikes
