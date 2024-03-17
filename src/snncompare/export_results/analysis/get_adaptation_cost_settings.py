"""Helps in computing the adaptation cost plot data."""
import json
from typing import Dict, List, Set, Union

import networkx as nx
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
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.run_config.Run_config import Run_config


# pylint: disable = R0903
class Cost_plot_group:
    """Stores the adaptation costs for an adaptation type."""

    @typechecked
    def __init__(
        self,
        adaptation_type: str,
        xy_coords_per_adaptation_type: Dict[float, List[float]],
    ) -> None:
        """Defines an adaptation object that stores the coordinates of the
        adaptations of a tree.

        Args:
        :adaptation_type: (str), The type of adaptation.
        :xy_coords_per_adaptation_type: (Dict[float, List[float]]), A
        dictionary containing the GPS coordinates of the adaptations of
        a tree.
        """

        self.adaptation_type: str = adaptation_type
        self.xy_coords_per_adaptation_type: Dict[
            float, List[float]
        ] = xy_coords_per_adaptation_type

    @typechecked
    def get_nth_y_coord_of_all_sizes(self, n: int) -> List[float]:
        """Returns the nth y-value of each x-coordinate's list of y-values."""
        nth_values: List[float] = []
        for values in self.xy_coords_per_adaptation_type.values():
            nth_values.append(values[n])
        return nth_values


# pylint: disable = R0902
# pylint: disable = R0903
class Raw_adap_cost_data:
    """Stores the raw data to make adaptation costs plots."""

    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        adaptation: Union[None, Adaptation],
        algorithm_name: str,
        algorithm_parameter: int,
        graph_size: int,
        graph_nr: int,
        input_graph: nx.Graph,
        dummy_run_config: Run_config,
        seed: int,
        snn_graph: Union[nx.DiGraph, Simulator],
    ) -> None:
        """Initializes the ``NeuralNetwork`` class.

        Args:
        :adaptation: (Union[None, Adaptation]), The adaptation to apply
        to the neural network.
        :algorithm_name: (str), The name of the algorithm to use for
        training the neural network.
        :algorithm_parameter: (int), The parameter to use for the
        training algorithm.
        :graph_size: (int), The size of the graph to use for training
        the neural network.
        :graph_nr: (int), The number of graphs to use for training the
        neural network.
        :input_graph: (nx.Graph), The input graph to use for training
        the neural network.
        :dummy_run_config: (Run_config), The dummy run configuration to
        use for training the neural network.
        :seed: (int), The seed to use for the random number generator.
        :snn_graph: (Union[nx.DiGraph, Simulator]), The SNN graph to use
        for training the neural network.
        Returns:
        None
        """

        self.adaptation: Union[None, Adaptation] = adaptation
        self.algorithm_name: str = algorithm_name
        self.algorithm_parameter: int = algorithm_parameter
        self.graph_size: int = graph_size
        self.graph_nr: int = graph_nr
        self.input_graph: nx.Graph = input_graph
        self.dummy_run_config: Run_config = dummy_run_config
        self.seed: int = seed
        self.snn_graph: Union[nx.DiGraph, Simulator] = snn_graph
        self.costs: Dict[str, int] = {}
        self.cost_types: List[str] = ["neuronal", "synaptic", "spikes"]
        for cost_type in self.cost_types:
            self.costs[cost_type] = self.get_cost_value(cost_type=cost_type)

    @typechecked
    def get_cost_value(self, cost_type: str) -> int:
        """Returns the adaptation cost belonging to the snn for the desired
        cost_type."""
        if cost_type == "spikes":
            if self.adaptation is None:
                with_adaptation = False
            else:
                with_adaptation = True
            return get_nr_of_spikes_in_snn(
                input_graph=self.input_graph,
                run_config=self.dummy_run_config,
                with_adaptation=with_adaptation,
            )
        if cost_type == "neuronal":
            return len(self.snn_graph.network.nodes)
        if cost_type == "synaptic":
            return len(self.snn_graph.network.synapses)
        raise NotImplementedError(f"Error, {cost_type} not supported.")


@typechecked
def get_raw_adap_cost_datas(
    *,
    exp_config: Exp_config,
) -> List[Raw_adap_cost_data]:
    """Generates the list of raw adaptation cost data objects for a given
    exp_config."""
    base_cost_settings: List[
        Dict
    ] = get_experiment_configurations_for_adaptation_settings(
        exp_config=exp_config,
    )
    raw_plot_datas: List[Raw_adap_cost_data] = []
    for base_cost_setting in base_cost_settings:
        # Get the adaptations that were used for these settings.
        for adaptation in exp_config.adaptations:
            # Get the input graph for these adaptations.
            input_graph: nx.Graph = get_input_graph(
                graph_size=base_cost_setting["graph_size"],
                graph_nr=base_cost_setting["graph_nr"],
                seed=base_cost_setting["seed"],
                with_rand_initialisation=True,
            )
            # Get the dummy run_config for these adaptations.
            dummy_run_config: Run_config = get_run_config(
                adaptation=adaptation,
                algorithm_name=base_cost_setting["algorithm_name"],
                algorithm_parameter=base_cost_setting["algorithm_param_val"],
                graph_size=base_cost_setting["graph_size"],
                graph_nr=base_cost_setting["graph_nr"],
                seed=base_cost_setting["seed"],
            )

            # Get the snns for these adaptations.
            # pprint(dummy_run_config.__dict__)
            adapted_snn = get_snn(
                dummy_run_config=dummy_run_config,
                with_adaptation=True,
            )
            # print(len(adapted_snn.network.graph.nodes))

            raw_plot_data: Raw_adap_cost_data = Raw_adap_cost_data(
                adaptation=adaptation,
                algorithm_name=base_cost_setting["algorithm_name"],
                algorithm_parameter=base_cost_setting["algorithm_param_val"],
                graph_size=base_cost_setting["graph_size"],
                graph_nr=base_cost_setting["graph_nr"],
                input_graph=input_graph,
                dummy_run_config=dummy_run_config,
                seed=base_cost_setting["seed"],
                snn_graph=adapted_snn,
            )
            raw_plot_datas.append(raw_plot_data)

        # Also get an unadapted snn for each base-setting, using an arbitrary
        # adaptation.
        un_adapted_snn = get_snn(
            dummy_run_config=dummy_run_config,
            with_adaptation=False,
        )
        raw_plot_data_without_adaptation = Raw_adap_cost_data(
            # adaptation=adaptation,
            adaptation=None,
            algorithm_name=base_cost_setting["algorithm_name"],
            algorithm_parameter=base_cost_setting["algorithm_param_val"],
            graph_size=base_cost_setting["graph_size"],
            graph_nr=base_cost_setting["graph_nr"],
            input_graph=input_graph,
            dummy_run_config=dummy_run_config,
            seed=base_cost_setting["seed"],
            snn_graph=un_adapted_snn,
        )
        raw_plot_datas.append(raw_plot_data_without_adaptation)
    return raw_plot_datas


# pylint: disable=R0914
@typechecked
def get_experiment_configurations_for_adaptation_settings(
    *,
    exp_config: Exp_config,
) -> List[Dict]:
    """Returns all relevant run config settings that are needed to generate the
    adaptation cost plots for the given exp_config."""
    base_cost_settings: List[Dict] = []

    # Per algorithm setting (e.g. MDSA value:)
    # pylint: disable=R1702
    if "MDSA" in exp_config.algorithms.keys():
        algorithm_name: str = "MDSA"
        for m_val_dict in exp_config.algorithms[algorithm_name]:
            m_val: int = m_val_dict["m_val"]

            for graph_size, nr_of_graphs in exp_config.size_and_max_graphs:
                # Per graph size, compute the y values/cost.

                # For each graph of a specific size, multiple runs are
                # performed with different random weight- (and radiation)
                # seeds.
                for seed in exp_config.seeds:
                    for graph_nr in range(0, nr_of_graphs):
                        # Load input graph from file.

                        base_cost_setting: Dict = {
                            "algorithm_name": algorithm_name,
                            "algorithm_param_name": "m_val",
                            "algorithm_param_val": m_val,
                            "graph_size": graph_size,
                            "seed": seed,
                            "graph_nr": graph_nr,
                        }
                        if base_cost_setting in base_cost_settings:
                            raise ValueError(
                                f"Error, {base_cost_setting} already in list."
                            )
                        base_cost_settings.append(base_cost_setting)
        return base_cost_settings
    raise NotImplementedError("Error, algorithm not (yet) supported.")


@typechecked
def get_adaptations_in_exp_config(
    *,
    exp_config: Exp_config,
) -> Set[Dict]:
    """Returns the adaptation settings."""

    adaptation_settings: Set[Dict] = set({})
    for adaptation in exp_config.adaptations:
        adaptation_setting: Dict = {
            "adaptation_type": adaptation.adaptation_type,
            "adaptation_redundancy": adaptation.redundancy,
        }
        adaptation_settings.add(adaptation_setting)
    return adaptation_settings


@typechecked
def get_input_graph(
    *,
    graph_nr: int,
    graph_size: int,
    seed: int,
    with_rand_initialisation: bool,
) -> nx.Graph:
    """Returns the input graph."""
    input_graph: nx.Graph = load_input_graph_based_on_nr(
        graph_size=graph_size, graph_nr=graph_nr
    )
    if with_rand_initialisation:
        add_mdsa_initialisation_properties_to_input_graph(
            input_graph=input_graph, seed=seed
        )
    return input_graph


@typechecked
def get_run_config(
    *,
    adaptation: Adaptation,
    algorithm_name: str,
    algorithm_parameter: int,
    graph_nr: int,
    graph_size: int,
    seed: int,
) -> Run_config:
    """Returns the input graph."""
    dummy_rad_damage: Rad_damage = Rad_damage(
        amplitude=0,
        effect_type="neuron_death",
        excitatory=False,
        inhibitory=False,
        probability_per_t=0.0,
    )

    dummy_run_config: Run_config = Run_config(
        adaptation=adaptation,
        algorithm={algorithm_name: {"m_val": algorithm_parameter}},
        graph_size=graph_size,
        graph_nr=graph_nr,
        radiation=dummy_rad_damage,
        seed=seed,
        simulator="simsnn",
    )
    return dummy_run_config


@typechecked
def get_snn(
    *,
    dummy_run_config: Run_config,
    with_adaptation: bool,
) -> Union[nx.DiGraph, Simulator]:
    """Get dummy run config."""
    # Get snn graph with or without adaptation, based on input
    # graph.
    stage_1_graphs: Dict[
        str, Union[nx.Graph, nx.DiGraph, Simulator]
    ] = get_graphs_stage_1(
        plot_config=get_default_plot_config(),
        run_config=dummy_run_config,
    )

    snn_graph: Union[nx.DiGraph, Simulator] = get_snn_graph_from_graphs_dict(
        with_adaptation=with_adaptation,
        with_radiation=False,
        graphs_dict=stage_1_graphs,
    )
    return snn_graph


@typechecked
def get_nr_of_spikes_in_snn(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
    with_adaptation: bool,
) -> int:
    """Returns the adaptation costs for this experiment setting."""
    _, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)

    (
        simsnn_exists,
        simsnn_filepath,
    ) = simsnn_files_exists_and_get_path(
        output_category="snns",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=with_adaptation,
        stage_index=2,
        rad_affected_neurons_hash=None,
        rand_nrs_hash=rand_nrs_hash,
    )

    if not simsnn_exists:
        raise FileNotFoundError(f"Error, {simsnn_filepath} not found.")

    # Read snn graph propagation JSON file into dict.
    with open(simsnn_filepath, encoding="utf-8") as json_file:
        snn_propagation = json.load(json_file)
        json_file.close()
    # TODO: determine why spikes is list in list, remove [0] if desirable.
    nr_of_spikes: int = sum(snn_propagation["spikes"][0])
    return nr_of_spikes
