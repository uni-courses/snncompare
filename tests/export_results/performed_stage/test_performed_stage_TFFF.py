"""Performs tests check whether the has_outputted_stage_jsons( function
correctly determines which stages have been completed and not for:
Stage1=Done
Stage2=Not yet done.
Stage3=Not yet done.
Stage4=Not yet done.
."""
import os
import shutil
import unittest

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists
from snncompare.export_results.helper import run_config_to_filename
from snncompare.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from snncompare.graph_generation.stage_1_create_graphs import (
    get_input_graph_of_run_config,
)
from snncompare.import_results.check_completed_stages import (
    has_outputted_stage_jsons,
)
from snncompare.import_results.read_json import load_results_from_json
from snncompare.json_configurations.create_default_settings import (
    default_exp_config,
)
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
        create_root_dir_if_not_exists(root_dir_name="latex/Images/graphs")

        # Initialise experiment settings, and run experiment.
        self.exp_config: Exp_config = default_exp_config()
        # self.input_graph = get_networkx_graph_of_2_neurons()

        self.expected_completed_stages = [1]
        self.export_images = False  # Expect the test to export snn pictures.
        # Instead of the Experiment_runner.
        self.exp_config.export_images = self.export_images
        self.experiment_runner = Experiment_runner(
            exp_config=self.exp_config,
        )

        # Pick (first) run config and get the output locations for testing.
        # TODO: make random, and make it loop through all/random run configs.
        nr_of_tested_configs: int = 10
        seed: int = 42
        self.run_configs = get_n_random_run_configs(
            run_configs=self.experiment_runner.run_configs,
            n=nr_of_tested_configs,
            seed=seed,
        )

    # Loop through (random) run configs.

    # Test:
    @typechecked
    def test_output_json_contains_(self) -> None:
        """Tests whether deleting all results and creating an artificial json
        with only stage 1 completed, results in has_outputted_stage_jsons()
        returning that only stage 1 is completed, and that stages 2,3 and 4 are
        not yet completed."""

        for run_config in self.experiment_runner.run_configs:
            filename: str = run_config_to_filename(
                run_config_dict=run_config.__dict__
            )
            json_filepath = "results/" + f"{filename}" + ".json"

            # TODO: determine per stage per run config which graph names are
            # expected.
            stage_1_graph_names = get_expected_stage_1_graph_names(
                run_config=run_config
            )
            create_result_file_for_testing(
                json_filepath=json_filepath,
                graph_names=stage_1_graph_names,
                completed_stages=self.expected_completed_stages,
                input_graph=get_input_graph_of_run_config(
                    run_config=run_config
                ),
                run_config=run_config,
            )

            # Read output JSON file into dict.
            stage_1_output_dict = load_results_from_json(
                json_filepath=json_filepath, run_config=run_config
            )

            # Verify the 3 dicts are in the result dict.
            self.assertIn("exp_config", stage_1_output_dict)
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
            # TODO: update expected stages.
            self.assertTrue(
                has_outputted_stage_jsons(
                    expected_stages=[
                        1,
                    ],
                    run_config=run_config,
                    stage_index=1,
                )
            )

            # Test for stage 1, 2, and 4.
            self.assertFalse(
                has_outputted_stage_jsons(
                    expected_stages=[1, 2],
                    run_config=run_config,
                    stage_index=2,
                )
            )
            self.assertTrue(
                has_outputted_stage_jsons(
                    expected_stages=[
                        1,
                        2,
                        3,
                    ],
                    run_config=run_config,
                    stage_index=3,
                )
            )
            self.assertFalse(
                has_outputted_stage_jsons(
                    expected_stages=[1, 2, 3, 4],
                    run_config=run_config,
                    stage_index=4,
                )
            )

            # TODO: write test for stage 3.
