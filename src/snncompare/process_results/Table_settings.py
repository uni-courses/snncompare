"""Computes what the failure modes were, and then stores this data in the
graphs."""
from typing import Dict, List, Tuple

import networkx as nx
from snnalgorithms.sparse.MDSA.alg_params import get_algorithm_setting_name
from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.graph_generation.stage_1_create_graphs import (
    load_input_graph_from_file_with_init_props,
)
from snncompare.helper import get_snn_graph_name
from snncompare.import_results.load_stage_1_and_2 import load_simsnn_graphs
from snncompare.process_results.helper import graph_of_run_config_passed
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
                self.algorithm_setts.append(
                    get_algorithm_setting_name(
                        algorithm_setting={algorithm_name: algo_config}
                    )
                )

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
        print("Creating failure mode table.")
        for i, run_config in enumerate(self.run_configs):
            snn_graphs: Dict = {}
            input_graph: nx.Graph = load_input_graph_from_file_with_init_props(
                run_config=run_config
            )
            print(
                f"{i}/{len(self.run_configs)},{run_config.adaptation.__dict__}"
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
            run_config_algorithm_name: str = get_algorithm_setting_name(
                algorithm_setting=run_config.algorithm
            )
            adaptation_name: str = (
                f"{run_config.adaptation.adaptation_type}_"
                + f"{run_config.adaptation.redundancy}"
            )

            if (
                run_config.seed == seed
                and run_config.graph_size == graph_size
                and run_config_algorithm_name == algorithm_setting
                and not graph_of_run_config_passed(
                    graph_name="rad_adapted_snn_graph",
                    run_config=run_config,
                )
            ):
                get_failure_mode_obj(
                    adaptation_name=adaptation_name,
                    failure_mode=failure_mode,
                    failure_mode_entries=failure_mode_entries,
                    first_timestep_only=first_timestep_only,
                    run_config=run_config,
                    show_spike_failures=show_spike_failures,
                )

        return failure_mode_entries


@typechecked
def get_failure_mode_obj(
    *,
    adaptation_name: str,
    failure_mode: Dict,
    failure_mode_entries: List[Failure_mode_entry],
    first_timestep_only: bool,
    run_config: Run_config,
    show_spike_failures: bool,
) -> None:
    """Parses input data to return a Failure mode object, containing the
    difference between the radiated and unradiated SNN."""

    @typechecked
    def create_failure_mode_entry(
        incorrectly_spikes: bool,
        incorrectly_silent: bool,
        incorrect_u_increase: bool,
        incorrect_u_decrease: bool,
        neuron_list: List[str],
    ) -> Failure_mode_entry:
        """Returns a Failure mode object, containing the difference between the
        radiated and unradiated SNN."""
        return Failure_mode_entry(
            adaptation_name=adaptation_name,
            incorrectly_spikes=incorrectly_spikes,
            incorrectly_silent=incorrectly_silent,
            incorrect_u_increase=incorrect_u_increase,
            incorrect_u_decrease=incorrect_u_decrease,
            neuron_names=neuron_list,
            run_config=run_config,
            timestep=int(timestep),
        )

    @typechecked
    def append_failure_mode_entry(
        failure_mode_entry: Failure_mode_entry,
    ) -> None:
        if first_timestep_only:
            failure_mode_entries.append(failure_mode_entry)
        else:
            # Optionally handle first timestep only logic here
            print("passing")

    if show_spike_failures:
        if "incorrectly_silent" in failure_mode:
            for timestep, neuron_list in failure_mode[
                "incorrectly_silent"
            ].items():
                failure_mode_entry = create_failure_mode_entry(
                    False, True, False, False, neuron_list
                )
                append_failure_mode_entry(failure_mode_entry)

        if "incorrectly_spikes" in failure_mode:
            for timestep, neuron_list in failure_mode[
                "incorrectly_spikes"
            ].items():
                failure_mode_entry = create_failure_mode_entry(
                    True, False, False, False, neuron_list
                )
                append_failure_mode_entry(failure_mode_entry)
    else:
        if "inhibitory_delta_u" in failure_mode:
            for timestep, neuron_list in failure_mode[
                "inhibitory_delta_u"
            ].items():
                failure_mode_entry = create_failure_mode_entry(
                    False, False, False, True, neuron_list
                )
                append_failure_mode_entry(failure_mode_entry)

        if "excitatory_delta_u" in failure_mode:
            for timestep, neuron_list in failure_mode[
                "excitatory_delta_u"
            ].items():
                failure_mode_entry = create_failure_mode_entry(
                    False, False, True, False, neuron_list
                )
