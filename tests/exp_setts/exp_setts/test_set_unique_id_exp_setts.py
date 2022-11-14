"""Tests whether the exp_setts gets the same unique_id at all times if its
content is the same, and that it gets different unique ids for different
exp_setts settings."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from snncompare.exp_setts.custom_setts.run_configs.algo_test import (
    experiment_config_for_mdsa_testing,
    get_exp_setts_mdsa_size5_m4,
)
from snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)


class Test_setting_unique_id_exp_setts(unittest.TestCase):
    """Tests whether the exp_setts gets the same unique_id at all times if its
    content is the same, and that it gets different unique ids for different
    exp_setts settings."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        # Generate default experiment config.
        self.exp_setts_list: dict = [
            experiment_config_for_mdsa_testing(),
            get_exp_setts_mdsa_size5_m4(),
        ]

        for exp_setts in self.exp_setts_list:
            exp_setts["show_snns"] = False
            exp_setts["export_images"] = False
            verify_experiment_config(
                Supported_experiment_settings(),
                exp_setts,
                has_unique_id=False,
                allow_optional=True,
            )

    @typechecked
    def test_same_exp_setts_same_unique_id(self) -> None:
        """Verifies the same run config gets the same unique_id."""

        # Generate run configurations.
        self.assertGreaterEqual(len(self.exp_setts_list), 2)

        # Verify the run configs are all different, (exclude the unique_id from
        # the comparison.)
        for row, row_exp_setts in enumerate(self.exp_setts_list):
            for col, col_exp_setts in enumerate(
                copy.deepcopy(self.exp_setts_list)
            ):
                if row == col:
                    self.assertTrue(row_exp_setts == col_exp_setts)
                else:
                    self.assertFalse(row_exp_setts == col_exp_setts)

        # Verify the run configs are all different, (exclude the unique_id from
        # the comparison.)
        for index, exp_setts in enumerate(self.exp_setts_list):
            supp_setts = Supported_experiment_settings()
            supp_setts.append_unique_experiment_config_id(
                exp_setts, allow_optional=True
            )
            if index == 0:
                self.assertEqual(
                    exp_setts["unique_id"],
                    "77352e201ee23bab9958b0a6b9e2bae409e9f9234cb79e33"
                    + "c181855cfeeb7a98",
                )
            if index == 1:
                self.assertEqual(
                    exp_setts["unique_id"],
                    "471ce9c875abffe41f6b23bea5bc5ed67d869c1962acf0"
                    + "5cf7937cf2348826b9",
                )
