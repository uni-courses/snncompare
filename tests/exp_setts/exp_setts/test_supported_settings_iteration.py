"""Verifies the Supported_experiment_settings object catches invalid iterations
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.exp_setts.exp_setts.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_setts,
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_iterations_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config function catches invalid
    iterations settings."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.supp_exp_setts = Supported_experiment_settings()
        self.maxDiff = None  # Display full error message.

        self.invalid_iterations_value = {
            "iterations": "invalid value of type string iso list of floats",
        }

        self.supp_exp_setts = supp_exp_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        # Overwrite default setting for testing purposes.
        self.with_adaptation_with_radiation["iterations"] = list(
            range(0, 3, 1)
        )
        self.valid_iterations = self.supp_exp_setts.iterations

    def test_error_is_thrown_if_iterations_key_is_missing(self):
        """Verifies an exception is thrown if the iteration key is missing from
        the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        experiment_config.pop("iterations")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            # "'iterations'",
            "Error:iterations is not in the configuration"
            + f" settings:{experiment_config.keys()}",
            str(context.exception),
        )

    def test_iterations_value_is_invalid_type(self):
        """Verifies an exception is thrown if the iteration dictionary value,
        is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_exp_setts.iterations)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            experiment_config["iterations"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                experiment_config,
                expected_type,
                self,
            )

    def test_catch_empty_iterations_value_list(self):
        """Verifies an exception is thrown if the iteration dictionary value is
        a list without elements."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        experiment_config["iterations"] = []

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_catch_iterations_value_too_low(self):
        """Verifies an exception is thrown if the iteration dictionary value is
        lower than the supported range of iteration values permits."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        experiment_config["iterations"] = [-2]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, iterations was expected to be in range:"
            + f'{self.with_adaptation_with_radiation["iterations"]}.'
            + f" Instead, it contains:{-2}.",
            str(context.exception),
        )

    def test_catch_iterations_value_too_high(self):
        """Verifies an exception is thrown if the iteration dictionary value is
        higher than the supported range of iteration values permits."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        experiment_config["iterations"] = [50]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, iterations was expected to be in range:"
            + f'{self.with_adaptation_with_radiation["iterations"]}.'
            + f" Instead, it contains:{50}.",
            str(context.exception),
        )
