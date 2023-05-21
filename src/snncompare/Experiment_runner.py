"""Contains the object that runs the entire experiment. Also Contains a single
setting of the experiment configuration settings.

(The values of the settings may vary, yet the types should be the same.)
"""

import copy
import multiprocessing
from typing import Dict, List, Optional, Union

import customshowme
import networkx as nx
from simsnn.core.simulators import Simulator
from snnalgorithms.get_input_graphs import (
    create_mdsa_input_graphs_from_exp_config,
)
from snnbackends.verify_nx_graphs import verify_results_nx_graphs
from typeguard import typechecked

from snncompare.create_configs import generate_run_configs
from snncompare.exp_config.Exp_config import (
    Exp_config,
    Supported_experiment_settings,
)
from snncompare.export_plots.create_dash_plot import create_svg_plot
from snncompare.export_plots.Plot_config import (
    Plot_config,
    get_default_plot_config,
)
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists
from snncompare.export_plots.temp_default_output_creation import (
    create_default_hover_info,
    create_default_output_config,
)
from snncompare.export_results.analysis.create_adaptation_cost_plot import (
    plot_raw_adap_cost_datas,
)
from snncompare.export_results.analysis.create_performance_plots import (
    create_performance_plots,
    get_completed_and_missing_run_configs,
)
from snncompare.export_results.output_stage1_configs_and_input_graph import (
    output_stage_1_configs_and_input_graphs,
)
from snncompare.export_results.output_stage1_snn_graphs import (
    output_stage_1_snns,
)
from snncompare.export_results.output_stage2_snns import output_stage_2_snns
from snncompare.export_results.output_stage4_results import output_snn_results
from snncompare.graph_generation.export_input_graphs import store_pickle
from snncompare.helper import (
    add_stage_completion_to_graph,
    get_snn_graph_names,
)
from snncompare.import_results.load_stage_1_and_2 import (
    assert_has_outputted_stage_1,
    has_outputted_stage_1,
    load_stage1_simsnn_graphs,
)
from snncompare.optional_config.Output_config import (
    Hover_info,
    Output_config,
    Zoom,
)
from snncompare.process_results.get_failure_modes import (
    add_failure_modes_to_graph,
)
from snncompare.process_results.show_failure_modes import show_failures
from snncompare.progress_report.has_completed_stage2_or_4 import (
    assert_has_outputted_stage_2_or_4,
    has_outputted_stage_2_or_4,
)
from snncompare.run_config.Run_config import Run_config
from snncompare.simulation.add_radiation_graphs import (
    ensure_empty_rad_snns_exist,
)

from .graph_generation.stage_1_create_graphs import (
    get_graphs_stage_1,
    load_input_graph_from_file_with_init_props,
)

# from .import_results.load_stage1_results import load_results_stage_1
from .process_results.process_results import set_results
from .simulation.stage2_sim import sim_graphs


class Experiment_runner:
    """Experiment manager.

    First prepares the environment for running the experiment, and then
    calls a private method that executes the experiment consisting of 4
    stages.
    """

    # pylint: disable=R0903
    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        output_config: Output_config,
        reverse: bool,
        perform_run: Optional[bool] = True,
        specific_run_config: Optional[Run_config] = None,
    ) -> None:
        # Ensure output directories are created for stages 1 to 4.
        create_root_dir_if_not_exists(root_dir_name="results")

        output_config.hover_info = create_default_hover_info(
            exp_config=exp_config
        )

        # Store the experiment configuration settings.
        self.exp_config = exp_config

        # Load the ranges of supported settings.
        self.supp_exp_config = Supported_experiment_settings()
        create_mdsa_input_graphs_from_exp_config(exp_config=exp_config)
        self.run_configs = generate_run_configs(
            exp_config=exp_config, specific_run_config=specific_run_config
        )

        if reverse:
            self.run_configs.reverse()

        if perform_run:  # Used to get quick Experiment_runner for testing.
            print("Performing run.\n\n")
            self.__perform_run(
                exp_config=self.exp_config,
                output_config=output_config,
                run_configs=self.run_configs,
            )

        if 5 in output_config.output_json_stages:
            print("Generating boxplot results.\n\n")
            create_performance_plots(
                completed_run_configs=self.run_configs,
                exp_config=exp_config,
            )

        if 6 in output_config.output_json_stages:
            plot_raw_adap_cost_datas(exp_config=self.exp_config)

        if output_config.extra_storing_config.show_failure_modes:
            show_failures(
                exp_config=self.exp_config, run_configs=self.run_configs
            )

    # pylint: disable=W0238
    @typechecked
    def __perform_run(
        self,
        exp_config: Exp_config,
        output_config: Output_config,
        run_configs: List[Run_config],
    ) -> None:
        """Private method that performs a run of the experiment.

        The 2 underscores indicate it is private. This method executes
        the run in the way the processed configuration settings specify.
        """
        plot_config = get_default_plot_config()
        results_nx_graphs: Dict
        for i, run_config in enumerate(run_configs):
            print(f"\n{i+1}/{len(run_configs)} [runs]")
            run_config.print_run_config_dict()
            results_nx_graphs = self.perform_run_stage_1(
                exp_config=exp_config,
                output_config=output_config,
                plot_config=plot_config,
                run_config=run_config,
            )

            results_nx_graphs = self.__perform_run_stage_2(
                results_nx_graphs=results_nx_graphs,
                output_config=output_config,
                run_config=run_config,
            )

            self.__perform_run_stage_3(
                exp_config=exp_config,
                output_config=output_config,
                results_nx_graphs=results_nx_graphs,
                run_config=run_config,
            )

            self.__perform_run_stage_4(
                exp_config=exp_config,
                output_config=output_config,
                results_nx_graphs=results_nx_graphs,
                run_config=run_config,
            )
            # Store run results in dict of Experiment_runner.
            self.results_nx_graphs: Dict = {
                run_config.unique_id: results_nx_graphs  # type:ignore[index]
            }

    @customshowme.time
    @typechecked
    def perform_run_stage_1(
        self,
        exp_config: Exp_config,
        output_config: Output_config,
        plot_config: Plot_config,
        run_config: Run_config,
    ) -> Dict:
        """Performs the run for stage 1 or loads the data from file depending
        on the run configuration.

        Stage 1 applies a conversion that the user specified to run an
        SNN algorithm. This is done by taking an input graph, and
        generating an SNN (graph) that runs the intended algorithm.
        """

        input_graph: nx.Graph = load_input_graph_from_file_with_init_props(
            run_config=run_config
        )

        add_stage_completion_to_graph(snn=input_graph, stage_index=1)

        results_nx_graphs = {
            "exp_config": exp_config,
            "run_config": run_config,
            "graphs_dict": {"input_graph": input_graph},
        }

        # Check if stage 1 is performed. If not, perform it.
        if (
            not has_outputted_stage_1(
                input_graph=input_graph,
                run_config=run_config,
            )
            or 1 in output_config.recreate_stages
        ):
            # Run first stage of experiment, get input graph.
            stage_1_graphs: Dict[
                str, Union[nx.Graph, nx.DiGraph, Simulator]
            ] = get_graphs_stage_1(
                plot_config=plot_config, run_config=run_config
            )

            # Indicate the graphs have completed stage 1.
            for snn in stage_1_graphs.values():
                add_stage_completion_to_graph(snn=snn, stage_index=1)

            results_nx_graphs["graphs_dict"] = stage_1_graphs

            output_stage_1_configs_and_input_graphs(
                exp_config=exp_config,
                run_config=run_config,
                graphs_dict=results_nx_graphs["graphs_dict"],
            )

            for with_adaptation in [False, True]:
                output_stage_1_snns(
                    run_config=run_config,
                    graphs_dict=results_nx_graphs["graphs_dict"],
                    with_adaptation=with_adaptation,
                )

        else:
            results_nx_graphs["graphs_dict"] = load_stage1_simsnn_graphs(
                run_config=run_config,
                stage_1_graphs_dict=results_nx_graphs["graphs_dict"],
            )

            # self.equalise_loaded_run_config(
            # loaded_from_json=results_nx_graphs["run_config"],
            # incoming=run_config,
            # )

        assert_has_outputted_stage_1(run_config=run_config)
        return results_nx_graphs

    @customshowme.time
    @typechecked
    def __perform_run_stage_2(
        self,
        output_config: Output_config,
        results_nx_graphs: Dict,
        run_config: Run_config,
    ) -> Dict:
        """Performs the run for stage 2 or loads the data from file depending
        on the run configuration.

        Stage two simulates the SNN graphs over time and, if desired,
        exports each timestep of those SNN graphs to a json dictionary.
        """

        # Verify incoming results dict.
        if run_config.simulator == "nx":
            verify_results_nx_graphs(
                results_nx_graphs=results_nx_graphs, run_config=run_config
            )

        ensure_empty_rad_snns_exist(
            run_config=run_config,
            stage_1_graphs=results_nx_graphs["graphs_dict"],
        )

        # Run simulation on networkx or lava backend.
        sim_graphs(
            output_config=output_config,
            run_config=run_config,
            stage_1_graphs=results_nx_graphs["graphs_dict"],
        )

        # TODO: include check to se if stage 2 output is skipped.
        output_stage_2_snns(
            graphs_dict=results_nx_graphs["graphs_dict"],
            output_config=output_config,
            run_config=run_config,
        )
        return results_nx_graphs

    @typechecked
    def __perform_run_stage_3(
        self,
        run_config: Run_config,
        exp_config: Exp_config,
        output_config: Output_config,
        results_nx_graphs: Dict,
    ) -> None:
        """Performs the run for stage 3, which visualises the behaviour of the
        SNN graphs over time. This behaviour is shown as a sequence of images.

        The behaviour is described with:
        - Green neuron: means a neuron spikes in that timestep.
        - Green synapse: means the input neuron of that synapse spiked at that
        timestep.
        - Red neuron: radiation has damaged/killed the neuron, it won't spike.
        - Red synapse: means the input neuron of that synapse has died and will
        not spike at that timestep.
        - White/transparent neuron: works as expected, yet does not spike at
        that timestep.
        - A circular synapse: a recurrent connection of a neuron into itself.
        """
        if output_config.export_types:
            if "hover_info" not in output_config.__dict__.keys():
                output_config = create_default_output_config(
                    exp_config=exp_config,
                )
            # Override output config from exp_config.
            output_config.extra_storing_config.show_images = True
            output_config.hover_info.neuron_properties = [
                "spikes",
                "a_in_next",
                "bias",
                "du",
                "u",
                "dv",
                "v",
                "vth",
            ]

            # Generate Dash plots using multiprocessing.
            jobs = []
            for i, graph_name in enumerate(get_snn_graph_names()):
                if output_config.dash_port is None:
                    output_config.dash_port = 8050 + i
                else:
                    output_config.dash_port += i
                if graph_name in output_config.graph_types:
                    p = multiprocessing.Process(
                        target=create_svg_plot,
                        args=(
                            [graph_name],
                            results_nx_graphs["graphs_dict"],
                            output_config,
                            run_config,
                        ),
                    )
                    jobs.append(p)
                    p.start()
            for proc in jobs:
                proc.join()
            input("Proceeding to next visualisation.")

    @customshowme.time
    @typechecked
    def __perform_run_stage_4(
        self,
        exp_config: Exp_config,
        output_config: Output_config,
        results_nx_graphs: Dict,
        run_config: Run_config,
    ) -> None:
        """Performs the run for stage 4.

        Stage 4 computes the results of the SNN against the
        default/Neumann implementation. Then stores this result in the
        last entry of each graph.
        """

        if (
            not has_outputted_stage_2_or_4(
                graphs_dict=results_nx_graphs["graphs_dict"],
                run_config=run_config,
                stage_index=4,
            )
            or 4 in output_config.recreate_stages
        ):
            set_results(
                exp_config=exp_config,
                output_config=output_config,
                run_config=run_config,
                stage_2_graphs=results_nx_graphs["graphs_dict"],
            )

            output_data_types = ["results"]
            if output_config.extra_storing_config.export_failure_modes:
                output_data_types += [
                    "failure_modes",
                ]

                # Set failure modes.
                add_failure_modes_to_graph(
                    snn_graphs=results_nx_graphs["graphs_dict"],
                    run_config=run_config,
                )

            for output_data_type in output_data_types:
                if output_data_type == "results":
                    stage_index: int = 4
                else:
                    stage_index = 7
                output_snn_results(
                    output_data_type=output_data_type,
                    run_config=run_config,
                    graphs_dict=results_nx_graphs["graphs_dict"],
                    stage_index=stage_index,
                )

            assert_has_outputted_stage_2_or_4(
                graphs_dict=results_nx_graphs["graphs_dict"],
                run_config=run_config,
                stage_index=4,
            )

    def load_pickled_boxplot_data(
        self,
        exp_config: Exp_config,
        output_config: Output_config,
        run_configs: List[Run_config],
    ) -> None:
        """Loads the data that is needed for a boxplot for the given experiment
        config, from a pickled file."""
        pickle_run_configs_filepath: str = (
            "latex/Images/completed_run_configs.pickle"
        )

        # Get list of missing run_configs based on the incoming run_configs.
        (
            _,
            missing_run_configs,
        ) = get_completed_and_missing_run_configs(run_configs=run_configs)

        # Create duplicate Output_config that is used to generate the data
        # belonging to each run config, using the Experiment runner.
        boxplot_output_config = Output_config(
            recreate_stages=[],
            export_types=[],
            hover_info=Hover_info(
                node_names=True,
                outgoing_synapses=True,
                incoming_synapses=False,
                neuron_properties=["v", "vth"],
                synapse_properties=["weight"],
                neuron_models=exp_config.neuron_models,
                synaptic_models=exp_config.synaptic_models,
            ),
            zoom=Zoom(
                create_zoomed_image=False,
                left_right=None,
                bottom_top=None,
            ),
            output_json_stages=[1, 2, 4],
            extra_storing_config=copy.deepcopy(
                output_config.extra_storing_config
            ),
        )

        # Generate the data/run the experiments for the missing run_configs.
        for missing_run_config in missing_run_configs:
            # Execute those run_configs
            Experiment_runner(
                exp_config=exp_config,
                output_config=boxplot_output_config,
                reverse=True,
                specific_run_config=missing_run_config,
                perform_run=True,
            )

        # Store the run configs into a file to save them as being "completed."
        store_pickle(
            run_configs=run_configs,
            filepath=pickle_run_configs_filepath,
        )
