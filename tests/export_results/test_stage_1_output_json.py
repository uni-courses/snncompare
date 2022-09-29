"""Performs tests that verifies the json files created in stage 1 are valid."""

import json
import os
import pathlib
import shutil
import unittest

from src.experiment_settings.Experiment_runner import (
    Experiment_runner,
    example_experi_config,
)
from src.export_results.helper import run_config_to_filename
from src.export_results.Output import output_files_stage_1
from src.graph_generation.stage_1_get_input_graphs import get_used_graphs
from src.helper import delete_file_if_exists
from tests.tests_helper import assertIsFile, assertIsNotFile


# pylint: disable=R0902:
class Test_stage_1_output_json(unittest.TestCase):
    """Tests whether the function output_stage_json() creates valid output json
    files."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Remove results directory if it exists.
        if os.path.exists("results"):
            shutil.rmtree("results")

        # Initialise experiment settings, and run experiment.
        self.experi_config: dict = example_experi_config()
        experiment_runner = Experiment_runner(
            self.experi_config, show_snns=False, export_snns=False
        )
        # TODO: verify the to_run is computed correctly.

        # Pick (first) run config and get the output locations for testing.
        # TODO: make random, and make it loop through all/random run configs.
        self.first_run_config = experiment_runner.run_configs[0]
        self.stage_1_graphs: dict = get_used_graphs(self.first_run_config)
        self.filename: str = run_config_to_filename(self.first_run_config)
        self.json_filepath = f"results/{self.filename}.json"
        self.stage_index: int = 1

    def test_output_json_contains_(self):
        """Tests whether the output function creates a json that can be read as
        a dict that contains an experi_config, a graphs_dict, and a
        run_config."""
        filepath = pathlib.Path(self.json_filepath)

        # Delete the expected/tested output files.
        delete_file_if_exists(self.json_filepath)

        # Verify the output files are deleted.
        assertIsNotFile(filepath)

        # Run function that is being tested.
        output_files_stage_1(
            self.experi_config, self.first_run_config, self.stage_1_graphs
        )

        # Verify output JSON file exists.
        assertIsFile(filepath)

        # Read output JSON file into dict.
        with open(self.json_filepath, encoding="utf-8") as json_file:
            stage_1_output_dict = json.load(json_file)

        self.assertIn("experiment_config", stage_1_output_dict)
        self.assertIn("run_config", stage_1_output_dict)
        self.assertIn("graphs_dict", stage_1_output_dict)

        # TODO: Assert the right graphs are within the graphs_dict.
        print(f'stage_1_output_dict={stage_1_output_dict["graphs_dict"]}')
        for key in stage_1_output_dict["graphs_dict"].keys():
            print(f"key={key}")

        # TODO: Assert graphs in graphs_dict contain correct stage_index.
