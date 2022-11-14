"""Tests whether the run_config gets the same unique_id at all times if its
content is the same, and that it gets different unique ids for different
run_config settings."""
# pylint: disable=R0801
import copy
import unittest
from typing import List

from typeguard import typechecked

from snncompare.exp_setts.custom_setts.run_configs.algo_test import (
    experiment_config_for_mdsa_testing,
)
from snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from snncompare.Experiment_runner import experiment_config_to_run_configs


class Test_setting_unique_id_run_config(unittest.TestCase):
    """Tests whether the run_config gets the same unique_id at all times if its
    content is the same, and that it gets different unique ids for different
    run_config settings."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        # Generate default experiment config.
        self.exp_setts: dict = experiment_config_for_mdsa_testing()

        self.exp_setts["show_snns"] = False
        self.exp_setts["export_images"] = False
        verify_experiment_config(
            Supported_experiment_settings(),
            self.exp_setts,
            has_unique_id=False,
            allow_optional=True,
        )

    @typechecked
    def test_same_run_config_same_unique_id(self) -> None:
        """Verifies the same run config gets the same unique_id."""

        # Create deepcopy of configuration settings.
        exp_setts = copy.deepcopy(self.exp_setts)

        # Generate run configurations.
        run_configs: List[dict] = experiment_config_to_run_configs(exp_setts)
        self.assertGreaterEqual(len(run_configs), 2)

        no_ids = copy.deepcopy(run_configs)
        for run_config in no_ids:
            run_config.pop("unique_id")

        # Verify the run configs are all different, (exclude the unique_id from
        # the comparison.)
        for row, row_run_config in enumerate(run_configs):
            for col, col_run_config in enumerate(copy.deepcopy(run_configs)):
                if row == col:
                    self.assertTrue(row_run_config == col_run_config)
                else:
                    self.assertFalse(row_run_config == col_run_config)

        # Verify the run configs are all different, (exclude the unique_id from
        # the comparison.)
        for index, run_config in enumerate(run_configs):
            if index == 0:
                self.assertEqual(
                    run_config["unique_id"],
                    "5fd1b88797031c152cb287c4d52182a8c7cd9830a870eea3882e8c"
                    + "3d0429bf58",
                )
            if index == 1:
                self.assertEqual(
                    run_config["unique_id"],
                    "ac4a2bf02c75d20f6d4d7039375e1e3f662e0aeaee97ccfbd08bf0"
                    + "6c6ac0b1c8",
                )
