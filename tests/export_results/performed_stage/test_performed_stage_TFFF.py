"""Performs tests check whether the has_outputted_stage function correctly
determines which stages have been completed and not for:
Stage1=Done
Stage2=Not yet done.
Stage3=Not yet done.
Stage4=Not yet done.
."""
import os
import shutil
import typing
import unittest

from snnbackends.plot_graphs import create_root_dir_if_not_exists
from typeguard import typechecked

from snncompare.exp_setts.default_setts.create_default_settings import (
    default_experiment_config,
)
from snncompare.Experiment_runner import (
    Experiment_runner,
    determine_what_to_run,
)
from snncompare.export_results.helper import run_config_to_filename
from snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from snncompare.graph_generation.stage_1_get_input_graphs import (
    get_input_graph,
)
from snncompare.import_results.check_completed_stages import (
    has_outputted_stage,
)
from snncompare.import_results.read_json import load_results_from_json
from tests.tests_helper import (
    create_result_file_for_testing,
    get_n_random_run_configs,
)


# pylint: disable=R0902:
# pylint: disable=R0801:
class Test_stage_1_output_json(unittest.TestCase):
    """Tests whether the function output_stage_json() creates valid output json
    files."""

    # Initialize test object
    @typechecked
    def __init__(self, *args: str, **kwargs: str):
        super().__init__(*args, **kwargs)

        # Remove results directory if it exists.
        if os.path.exists("results"):
            shutil.rmtree("results")
        if os.path.exists("latex"):
            shutil.rmtree("latex")
        create_root_dir_if_not_exists("latex/Images/graphs")

        # Initialise experiment settings, and run experiment.
        self.experiment_config: typing.Dict[
            str, typing.Union[str, int]
        ] = default_experiment_config()
        # self.input_graph = get_networkx_graph_of_2_neurons()

        self.expected_completed_stages = [1]
        self.export_images = False  # Expect the test to export snn pictures.
        # Instead of the Experiment_runner.
        self.experiment_config["show_snn"] = False
        self.experiment_config["export_images"] = self.export_images
        self.experiment_runner = Experiment_runner(
            self.experiment_config,
        )
        # TODO: verify the to_run is computed correctly.

        # Pick (first) run config and get the output locations for testing.
        # TODO: make random, and make it loop through all/random run configs.
        nr_of_tested_configs: int = 10
        seed: int = 42
        self.run_configs = get_n_random_run_configs(
            self.experiment_runner.run_configs, nr_of_tested_configs, seed
        )

    # Loop through (random) run configs.

    # Test:
    @typechecked
    def test_output_json_contains_(self) -> None:
        """Tests whether deleting all results and creating an artificial json
        with only stage 1 completed, results in has_outputted_stage() returning
        that only stage 1 is completed, and that stages 2,3 and 4 are not yet
        completed."""

        for run_config in self.experiment_runner.run_configs:
            to_run = determine_what_to_run(run_config)
            json_filepath = (
                f"results/{run_config_to_filename(run_config)}.json"
            )

            # TODO: determine per stage per run config which graph names are
            # expected.
            stage_1_graph_names = get_expected_stage_1_graph_names(run_config)
            create_result_file_for_testing(
                json_filepath,
                stage_1_graph_names,
                self.expected_completed_stages,
                get_input_graph(run_config),
                run_config,
            )

            # Read output JSON file into dict.
            stage_1_output_dict = load_results_from_json(
                json_filepath, run_config
            )

            # Verify the 3 dicts are in the result dict.
            self.assertIn("experiment_config", stage_1_output_dict)
            self.assertIn("run_config", stage_1_output_dict)
            self.assertIn("graphs_dict", stage_1_output_dict)
            self.assertIn(
                "alg_props",
                stage_1_output_dict["graphs_dict"]["input_graph"].graph,
            )

            # Verify the right graphs are within the graphs_dict.
            for graph_name in stage_1_graph_names:
                self.assertIn(
                    graph_name, stage_1_output_dict["graphs_dict"].keys()
                )

            # Verify each graph has the right completed stages attribute.
            for graph_name in stage_1_output_dict["graphs_dict"].keys():
                self.assertEqual(
                    stage_1_output_dict["graphs_dict"][graph_name].graph[
                        "completed_stages"
                    ],
                    self.expected_completed_stages,
                )

            # Test whether the performed stage function returns False for the
            # uncompleted stages in the graphs.
            print("first test")
            self.assertTrue(has_outputted_stage(run_config, 1, to_run))

            # Test for stage 1, 2, and 4.
            self.assertFalse(has_outputted_stage(run_config, 2, to_run))
            self.assertEqual(
                has_outputted_stage(run_config, 3, to_run), self.export_images
            )
            self.assertFalse(has_outputted_stage(run_config, 4, to_run))

            # TODO: write test for stage 3.
