"""Verifies the Supported_experiment_settings object catches invalid
max_max_graphs specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.experiment_settings.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.experiment_settings.test_generic_configuration import (
    adap_sets,
    rad_sets,
    supp_sets,
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_max_max_graphs_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid max_max_graphs settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_sets = Supported_experiment_settings()
        self.valid_max_max_graphs = self.supp_sets.max_max_graphs

        self.invalid_max_max_graphs_value = {
            "max_max_graphs": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_error_is_thrown_if_max_max_graphs_key_is_missing(self):
        """Verifies an exception is thrown if the max_max_graphs key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        config_settings.pop("max_max_graphs")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            # "'max_max_graphs'",
            "Error:max_max_graphs is not in the configuration"
            + f" settings:{config_settings.keys()}",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_max_max_graphs_value_type(self):
        """Verifies an exception is thrown if the max_max_graphs dictionary
        value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_sets.max_max_graphs)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            config_settings["max_max_graphs"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                config_settings,
                expected_type,
                self,
            )

    def test_catch_invalid_max_max_graphs_value_type_too_low(self):
        """Verifies an exception is thrown if the max_max_graphs dictionary
        value is lower than the supported range of max_max_graphs values
        permits."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_max_graphs in copy.
        config_settings["max_max_graphs"] = -2

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, setting expected to be at least "
            + f"{self.supp_sets.min_max_graphs}. "
            + f"Instead, it is:{-2}",
            str(context.exception),
        )

    def test_catch_max_max_graphs_is_smaller_than_min_max_graphs(self):
        """Verifies an exception is thrown if the max_max_graphs value is
        smaller than the min_max_graphs value."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_max_graphs in copy.
        config_settings["min_max_graphs"] = (
            config_settings["min_max_graphs"] + 1
        )
        config_settings["max_max_graphs"] = (
            config_settings["min_max_graphs"] - 1
        )

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            f'Lower bound:{config_settings["min_max_graphs"]} is larger than'
            f' upper bound:{config_settings["max_max_graphs"]}.',
            str(context.exception),
        )

    def test_catch_max_max_graphs_value_too_high(self):
        """Verifies an exception is thrown if the max_max_graphs dictionary
        value is higher than the supported range of max_max_graphs values
        permits."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_max_graphs in copy.
        config_settings["max_max_graphs"] = 50

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, setting expected to be at most "
            + f"{self.supp_sets.max_max_graphs}. Instead, it is:"
            + "50",
            str(context.exception),
        )
