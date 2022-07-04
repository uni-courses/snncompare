"""Verifies the Supported_settings object catches invalid iterations
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.verify_supported_settings import (
    verify_configuration_settings,
)
from tests.experiment_settings.test_generic_configuration import (
    adap_sets,
    rad_sets,
    supp_sets,
    with_adaptation_with_radiation,
)


class Test_iterations_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings function catches invalid
    iterations settings."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.supp_sets = Supported_settings()

        self.invalid_iterations_value = {
            "iterations": "invalid value of type string iso list of floats",
        }

        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_iterations = self.supp_sets.iterations

    # TODO: write test_iterations is None
    # TODO: write test_iterations is of invalid type.

    def test_catch_invalid_iterations_value_type_too_low(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        config_settings["iterations"] = [-2]

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, iterations was expected to be in range:"
            + f'{self.with_adaptation_with_radiation["iterations"]}.'
            + f" Instead, it contains:{-2}.",
            str(context.exception),
        )

    def test_catch_invalid_iterations_value_type_too_high(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        config_settings["iterations"] = [50]

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, iterations was expected to be in range:"
            + f'{self.with_adaptation_with_radiation["iterations"]}.'
            + f" Instead, it contains:{50}.",
            str(context.exception),
        )

    def test_catch_empty_iterations_value_list(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of iterations in copy.
        config_settings["iterations"] = []

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_empty_iterations(self):
        """Verifies an exception is thrown if an empty iterations dict is
        thrown."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        config_settings.pop("iterations")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "'iterations'",
            str(context.exception),
        )
