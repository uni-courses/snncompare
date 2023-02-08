"""Contains the object that runs the entire experiment. Also Contains a single
setting of the experiment configuration settings.

(The values of the settings may vary, yet the types should be the same.)
"""

# import showme
import copy
from pprint import pprint
from typing import Dict, List, Optional

import customshowme
from snnbackends.verify_nx_graphs import (
    results_nx_graphs_contain_expected_stages,
    verify_results_nx_graphs,
    verify_results_nx_graphs_contain_expected_stages,
)
from typeguard import typechecked

from snncompare.create_configs import generate_run_configs
from snncompare.exp_config.Exp_config import (
    Exp_config,
    Supported_experiment_settings,
)
from snncompare.export_plots.Plot_config import (
    Plot_config,
    get_default_plot_config,
)
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists
from snncompare.export_results.analysis.create_performance_plots import (
    create_performance_plots,
    get_completed_and_missing_run_configs,
    store_pickle,
)
from snncompare.helper import dicts_are_equal
from snncompare.optional_config.Output_config import Output_config, Zoom
from snncompare.run_config.Run_config import Run_config

from .export_results.Output_stage_12 import output_files_stage_1_and_2
from .export_results.Output_stage_34 import output_stage_files_3_and_4
from .graph_generation.stage_1_create_graphs import get_used_graphs
from .import_results.check_completed_stages import has_outputted_stage_jsons
from .import_results.stage_1_load_input_graphs import load_results_stage_1
from .process_results.process_results import (
    export_results_to_json,
    set_results,
)
from .run_config.verify_run_completion import (
    assert_stage_3_is_completed,
    assert_stage_is_completed,
)
from .simulation.stage2_sim import sim_graphs


class Experiment_runner:
    """Experiment manager.

    First prepares the environment for running the experiment, and then
    calls a private method that executes the experiment consisting of 4
    stages.
    """

    # pylint: disable=R0903

    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        output_config: Output_config,
        perform_run: bool = True,
        specific_run_config: Optional[Run_config] = None,
    ) -> None:

        # Ensure output directories are created for stages 1 to 4.
        create_root_dir_if_not_exists(root_dir_name="results")

        # Store the experiment configuration settings.
        self.exp_config = exp_config

        # Load the ranges of supported settings.
        self.supp_exp_config = Supported_experiment_settings()

        self.run_configs = generate_run_configs(
            exp_config=exp_config, specific_run_config=specific_run_config
        )

        if perform_run:  # Used to get quick Experiment_runner for testing.
            self.__perform_run(
                exp_config=self.exp_config,
                output_config=output_config,
                run_configs=self.run_configs,
            )

        if 5 in output_config.output_json_stages:
            self.load_pickled_boxplot_data(
                exp_config=self.exp_config,
                run_configs=self.run_configs,
                output_config=output_config,
            )

            create_performance_plots(
                completed_run_configs=self.run_configs,
                exp_config=exp_config,
                output_config=output_config,
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
            pprint(run_config.__dict__)
            results_nx_graphs = self.__perform_run_stage_1(
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
                output_config=output_config,
                results_nx_graphs=results_nx_graphs,
                run_config=run_config,
            )

            self.__perform_run_stage_4(
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
    def __perform_run_stage_1(
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

        # Check if stage 1 is performed. If not, perform it.
        if (
            not has_outputted_stage_jsons(
                expected_stages=[1], run_config=run_config, stage_index=1
            )
            or 1 in output_config.recreate_stages
        ):
            # Run first stage of experiment, get input graph.
            stage_1_graphs: Dict = get_used_graphs(
                plot_config=plot_config, run_config=run_config
            )
            results_nx_graphs = {
                "exp_config": exp_config,
                "run_config": run_config,
                "graphs_dict": stage_1_graphs,
            }

            # Exports results, including graphs as dict.
            output_files_stage_1_and_2(
                results_nx_graphs=results_nx_graphs, stage_index=1
            )
        else:
            results_nx_graphs = load_results_stage_1(run_config=run_config)
        self.equalise_loaded_run_config(
            loaded_from_json=results_nx_graphs["run_config"],
            incoming=run_config,
        )

        assert_stage_is_completed(
            expected_stages=[1],
            run_config=run_config,
            stage_index=1,
        )
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
        verify_results_nx_graphs(
            results_nx_graphs=results_nx_graphs, run_config=run_config
        )

        if (
            not has_outputted_stage_jsons(
                expected_stages=[1, 2], run_config=run_config, stage_index=2
            )
            or 2 in output_config.recreate_stages
        ):
            # Only stage I should be loaded.
            results_nx_graphs = load_results_stage_1(run_config=run_config)
            self.equalise_loaded_run_config(
                loaded_from_json=results_nx_graphs["run_config"],
                incoming=run_config,
            )
            # TODO: remove stage 2 artifacts from loaded data.

            # Run simulation on networkx or lava backend.
            sim_graphs(
                run_config=run_config,
                stage_1_graphs=results_nx_graphs["graphs_dict"],
            )
            output_files_stage_1_and_2(
                results_nx_graphs=results_nx_graphs, stage_index=2
            )
        else:
            # TODO: verify loading is required.
            if not results_nx_graphs_contain_expected_stages(
                results_nx_graphs=results_nx_graphs,
                stage_index=2,
                expected_stages=[
                    1,
                    2,
                ],
            ):
                # Load results of stage 1 and 2 from file.
                results_nx_graphs = load_results_stage_1(run_config=run_config)
                self.equalise_loaded_run_config(
                    loaded_from_json=results_nx_graphs["run_config"],
                    incoming=run_config,
                )
        verify_results_nx_graphs_contain_expected_stages(
            results_nx_graphs=results_nx_graphs,
            stage_index=2,
            expected_stages=[
                1,
                2,
            ],
        )

        assert_stage_is_completed(
            expected_stages=[1, 2],
            run_config=run_config,
            stage_index=2,
        )

        return results_nx_graphs

    @customshowme.time
    @typechecked
    def __perform_run_stage_3(
        self,
        output_config: Output_config,
        results_nx_graphs: Dict,
        run_config: Run_config,
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
            # Generate output json dicts (and plots) of propagated graphs.
            output_stage_files_3_and_4(
                output_config=output_config,
                results_nx_graphs=results_nx_graphs,
                stage_index=3,
            )

            # TODO: assert gif file exists
            assert_stage_3_is_completed(
                results_nx_graphs=results_nx_graphs,
                run_config=run_config,
            )

    @typechecked
    def __perform_run_stage_4(
        self,
        output_config: Output_config,
        results_nx_graphs: Dict,
        run_config: Run_config,
    ) -> None:
        """Performs the run for stage 4.

        Stage 4 computes the results of the SNN against the
        default/Neumann implementation. Then stores this result in the
        last entry of each graph.
        """
        verify_results_nx_graphs_contain_expected_stages(
            results_nx_graphs=results_nx_graphs,
            stage_index=2,
            expected_stages=[
                1,
                2,
            ],
        )

        if set_results(
            output_config=output_config,
            run_config=run_config,
            stage_2_graphs=results_nx_graphs["graphs_dict"],
        ):
            export_results_to_json(
                output_config=output_config,
                results_nx_graphs=results_nx_graphs,
                stage_index=4,
            )

        assert_stage_is_completed(
            expected_stages=[1, 2, 4],  # TODO: determine if 3 should be in.
            run_config=run_config,
            stage_index=4,
        )

    @typechecked
    def equalise_loaded_run_config(
        self,
        loaded_from_json: Run_config,
        incoming: Run_config,
    ) -> None:
        """Ensures the non-impactfull run config that is loaded from json are
        identical to those of the  incoming run_config."""
        for key, val in incoming.__dict__.items():

            if loaded_from_json.__dict__[key] != val:
                loaded_from_json.__dict__[key] = val
        if not dicts_are_equal(
            left=loaded_from_json.__dict__,
            right=incoming.__dict__,
            without_unique_id=False,
        ):
            pprint(loaded_from_json.__dict__)
            pprint(incoming.__dict__)
            raise AttributeError(
                "Run_config and loaded run_config from json are not equal."
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
                specific_run_config=missing_run_config,
                perform_run=True,
            )

        # Store the run configs into a file to save them as being "completed."
        store_pickle(
            run_configs=run_configs,
            filepath=pickle_run_configs_filepath,
        )
