"""Verifies the Supported_experiment_settings object catches invalid iterations
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.experiment_settings.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_experi_setts,
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_iterations_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config function catches invalid
    iterations settings."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.supp_experi_setts = Supported_experiment_settings()
        self.maxDiff = None  # Display full error message.

        self.invalid_iterations_value = {
            "iterations": "invalid value of type string iso list of floats",
        }

        self.supp_experi_setts = supp_experi_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_iterations = self.supp_experi_setts.iterations

    def test_error_is_thrown_if_iterations_key_is_missing(self):
        """Verifies an exception is thrown if the iteration key is missing from
        the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        experi_config.pop("iterations")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experi_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            # "'iterations'",
            "Error:iterations is not in the configuration"
            + f" settings:{experi_config.keys()}",
            str(context.exception),
        )

    def test_iterations_value_is_invalid_type(self):
        """Verifies an exception is thrown if the iteration dictionary value,
        is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_experi_setts.iterations)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            experi_config["iterations"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                experi_config,
                expected_type,
                self,
            )

    def test_catch_empty_iterations_value_list(self):
        """Verifies an exception is thrown if the iteration dictionary value is
        a list without elements."""
        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        experi_config["iterations"] = []

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experi_config,
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
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        experi_config["iterations"] = [-2]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experi_config,
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
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        experi_config["iterations"] = [50]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experi_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, iterations was expected to be in range:"
            + f'{self.with_adaptation_with_radiation["iterations"]}.'
            + f" Instead, it contains:{50}.",
            str(context.exception),
        )
