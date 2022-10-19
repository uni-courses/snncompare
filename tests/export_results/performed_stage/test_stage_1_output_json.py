"""Performs tests that verifies the json files created in stage 1 are valid."""

import os
import pathlib
import shutil
import unittest

from src.experiment_settings.Experiment_runner import (
    Experiment_runner,
    example_experi_config,
)
from src.export_results.helper import run_config_to_filename
from src.graph_generation.stage_1_get_input_graphs import get_used_graphs
from src.import_results.stage_1_load_input_graphs import load_results_from_json
from tests.tests_helper import assertIsFile


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
            self.experi_config, export_snns=False, show_snns=False
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

        # Verify output JSON file exists.
        filepath = pathlib.Path(self.json_filepath)
        assertIsFile(filepath)

        # Read output JSON file into dict.
        # with open(self.json_filepath, encoding="utf-8") as json_file:
        #    stage_1_output_dict = json.load(json_file)
        stage_1_output_dict = load_results_from_json(
            self.json_filepath, self.first_run_config
        )

        self.assertIn("experiment_config", stage_1_output_dict)
        self.assertIn("run_config", stage_1_output_dict)
        self.assertIn("graphs_dict", stage_1_output_dict)

        # TODO: Assert the right graphs are within the graphs_dict.
        print(f'stage_1_output_dict={stage_1_output_dict["graphs_dict"]}')
        for key in stage_1_output_dict["graphs_dict"].keys():
            # g = nx.DiGraph(d)
            print(f"key={key}")

        # TODO: Assert graphs in graphs_dict contain correct stage_index.
