"""Performs tests check whether the performed_stage function correctly
determines which stages have been completed and not for:
Stage1=Done
Stage2=Done
Stage3=Done
Stage4=TODO
."""

import os
import shutil
import unittest

from src.experiment_settings.Experiment_runner import (
    Experiment_runner,
    example_experi_config,
)
from src.export_results.helper import run_config_to_filename
from src.export_results.plot_graphs import create_root_dir_if_not_exists
from src.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)
from src.graph_generation.get_graph import get_networkx_graph_of_2_neurons
from src.helper import get_extensions_list
from src.import_results.stage_1_load_input_graphs import (
    load_results_from_json,
    performed_stage,
)
from tests.tests_helper import (
    create_dummy_output_images_stage_3,
    create_result_file_for_testing,
    get_n_random_run_configs,
)


# pylint: disable=R0902:
# pylint: disable=R0801:
class Test_stage_1_output_json(unittest.TestCase):
    """Tests whether the function output_stage_json() creates valid output json
    files."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Remove results directory if it exists.
        if os.path.exists("results"):
            shutil.rmtree("results")
        if os.path.exists("latex"):
            shutil.rmtree("latex")
        create_root_dir_if_not_exists("latex/Images/graphs")

        # Initialise experiment settings, and run experiment.
        self.experi_config: dict = example_experi_config()
        self.input_graph = get_networkx_graph_of_2_neurons()

        self.expected_completed_stages = [1, 2, 3, 4]
        self.export_snns = False  # Expect the test to export snn pictures.
        # Instead of the Experiment_runner.
        self.experiment_runner = Experiment_runner(
            self.experi_config, export_snns=self.export_snns, show_snns=False
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

    # Test: Deleting all results says none of the stages have been performed.
    def test_output_json_contains_(self):
        """Tests whether the output function creates a json that can be read as
        a dict that contains an experi_config, a graphs_dict, and a
        run_config."""

        for run_config in self.experiment_runner.run_configs:
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
                self.input_graph,
                run_config,
            )
            create_dummy_output_images_stage_3(
                stage_1_graph_names,
                self.input_graph,
                run_config,
                get_extensions_list(run_config, 3),
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
            self.assertTrue(performed_stage(run_config, 1))

            # Test for stage 1, 2, and 4.
            self.assertTrue(performed_stage(run_config, 2))
            self.assertTrue(performed_stage(run_config, 3))
            self.assertTrue(performed_stage(run_config, 4))
