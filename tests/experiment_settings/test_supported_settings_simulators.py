"""Verifies the Supported_settings object catches invalid simulators
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
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_simulators_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings_types function catches
    invalid simulators settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.supp_sets = Supported_settings()

        self.invalid_simulators_value = {
            "simulators": "invalid value of type string iso list of floats",
        }

        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_simulators = self.supp_sets.simulators

    def test_catch_invalid_simulators_value(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of simulators in copy.
        config_settings["simulators"] = [
            "nx",
            "invalid_simulator_name",
            "lava",
        ]

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, simulators was expected to be in range:"
            + f"{self.supp_sets.simulators}."
            + " Instead, it contains:invalid_simulator_name.",
            str(context.exception),
        )

    def test_catch_empty_simulators_value_list(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of simulators in copy.
        config_settings["simulators"] = []

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_returns_valid_m(self):
        """Verifies a valid simulators is returned."""
        returned_dict = verify_configuration_settings(
            self.supp_sets,
            self.with_adaptation_with_radiation,
            has_unique_id=False,
        )
        self.assertIsInstance(returned_dict, dict)

    def test_empty_simulators(self):
        """Verifies an exception is thrown if an empty simulators dict is
        thrown."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        config_settings.pop("simulators")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            # "'simulators'",
            "Error:simulators is not in the configuration"
            + f" settings:{config_settings.keys()}",
            str(context.exception),
        )

    def test_simulators_value_is_invalid_type(self):
        """Verifies an exception is thrown if the configuration setting:

        simulators is of invalid type.
        """

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_sets.simulators)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            config_settings["simulators"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                config_settings,
                expected_type,
                self,
            )
