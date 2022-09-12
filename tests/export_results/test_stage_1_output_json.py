"""Performs tests that verifies the json files created in stage 1 are valid."""

import json
import pathlib as pl
import unittest

from src.experiment_settings.Experiment_runner import (
    example_experi_config,
    experiment_config_to_run_configs,
)
from src.export_results.helper import run_config_to_filename
from src.export_results.Output import output_files_stage_1
from src.graph_generation.stage_1_get_input_graphs import get_used_graphs
from src.helper import delete_file_if_exists


# pylint: disable=R0902:
class Test_stage_1_output_json(unittest.TestCase):
    """Tests whether the function output_stage_json() creates valid output json
    files."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialise:
        self.experi_config: dict = example_experi_config()
        # Run minimum steps to produce valid output graph.
        self.experi_config["show_snns"] = False
        self.experi_config["export_snns"] = True

        self.run_configs: dict = experiment_config_to_run_configs(
            self.experi_config
        )
        # Pick first run config.
        self.first_run_config = self.run_configs[0]
        self.stage_1_graphs: dict = get_used_graphs(self.first_run_config)
        self.filename: str = run_config_to_filename(self.first_run_config)
        self.json_filepath = f"results/{self.filename}.json"
        self.stage_index: int = 1

    def assertIsFile(self, path):
        """Asserts a file exists.

        Throws error if a file does not exist.
        """
        if not pl.Path(path).resolve().is_file():
            # pylint: disable=C0209
            raise AssertionError("File does not exist: %s" % str(path))

    def assertIsNotFile(self, path):
        """Asserts a file does not exists.

        Throws error if the file does exist.
        """
        if pl.Path(path).resolve().is_file():
            # pylint: disable=C0209
            raise AssertionError("File exist: %s" % str(path))

    def test_output_json_contains_(self):
        """Tests whether the output function creates a json that can be read as
        a dict that contains an experi_config, a graphs_dict, and a
        run_config."""
        filepath = pl.Path(self.json_filepath)

        # Delete the expected/tested output files.
        delete_file_if_exists(self.json_filepath)

        # Verify the output files are deleted.
        self.assertIsNotFile(filepath)

        # Run function that is being tested.
        output_files_stage_1(
            self.experi_config, self.first_run_config, self.stage_1_graphs
        )

        # Verify output JSON file exists.
        self.assertIsFile(filepath)

        # Read output JSON file into dict.
        with open(self.json_filepath, encoding="utf-8") as json_file:
            stage_1_output_dict = json.load(json_file)

        self.assertIn("experiment_config", stage_1_output_dict)
        self.assertIn("run_config", stage_1_output_dict)
        self.assertIn("graphs_dict", stage_1_output_dict)
