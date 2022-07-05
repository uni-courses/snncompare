"""Verifies the Supported_settings object catches invalid m specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.Supported_settings import Supported_settings
from src.experiment_settings.verify_supported_settings import (
    verify_configuration_settings,
)
from tests.experiment_settings.test_generic_configuration import (
    adap_sets,
    rad_sets,
    supp_sets,
    verify_type_error_is_thrown_on_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_m_vals_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings_types function catches
    invalid m settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_sets = Supported_settings()
        self.valid_m_vals = self.supp_sets.algorithms["MDSA"].m_vals

        self.invalid_m_vals_value = {
            "m_vals": "invalid value of type string iso list of floats",
        }

        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_m_vals_is_none(self):
        """Verifies an error is thrown if configuration settings do not contain
        this setting.."""

        with self.assertRaises(Exception) as context:

            # Configuration Settings of type None throw error.
            verify_configuration_settings(
                self.supp_sets, None, has_unique_id=False
            )

        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f"{type(None)}, yet it was expected to be of"
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_m_vals_type(self):
        """."""

        with self.assertRaises(Exception) as context:
            # m dictionary of type None throws error.
            verify_configuration_settings(
                self.supp_sets, "string_instead_of_dict", has_unique_id=False
            )

        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f'{type("")}, yet it was expected to be of'
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_m_vals_value_type_too_low(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of m in copy.
        config_settings["algorithms"]["MDSA"]["m_vals"] = [-2]

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        expected_m_vals = self.with_adaptation_with_radiation["algorithms"][
            "MDSA"
        ]["m_vals"]
        self.assertEqual(
            "Error, m_vals was expected to be in range:"
            + f"{expected_m_vals}."
            + f" Instead, it contains:{-2}.",
            str(context.exception),
        )

    def test_catch_invalid_m_vals_value_type_too_high(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of m in copy.
        config_settings["algorithms"]["MDSA"]["m_vals"] = [50]

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        expected_m_vals = self.with_adaptation_with_radiation["algorithms"][
            "MDSA"
        ]["m_vals"]
        self.assertEqual(
            "Error, m_vals was expected to be in range:"
            + f"{expected_m_vals}."
            + f" Instead, it contains:{50}.",
            str(context.exception),
        )

    def test_catch_empty_m_vals_value_list(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of m in copy.
        config_settings["algorithms"]["MDSA"]["m_vals"] = []

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_returns_valid_m_vals(self):
        """Verifies a valid m is returned."""
        returned_dict = verify_configuration_settings(
            self.supp_sets,
            self.with_adaptation_with_radiation,
            has_unique_id=False,
        )
        self.assertIsInstance(returned_dict, dict)

    def test_empty_m_vals(self):
        """Verifies an exception is thrown if an empty m dict is thrown."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        config_settings["algorithms"]["MDSA"].pop("m_vals")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "'m_vals'",
            str(context.exception),
        )

    def test_m_vals_value_is_invalid_type(self):
        """Verifies an exception is thrown if the configuration setting:

        m_vals is of invalid type.
        """

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_sets.algorithms["MDSA"].m_vals)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            config_settings["algorithms"]["MDSA"][
                "m_vals"
            ] = invalid_config_setting_value
            verify_type_error_is_thrown_on_configuration_setting_value(
                invalid_config_setting_value,
                config_settings,
                expected_type,
                self,
            )
