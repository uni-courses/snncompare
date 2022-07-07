"""Verifies the Supported_experiment_settings object catches invalid m
specifications."""
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
    supp_experi_setts,
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_m_vals_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid m settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_experi_setts = Supported_experiment_settings()
        self.valid_m_vals = self.supp_experi_setts.algorithms["MDSA"].m_vals

        self.invalid_m_vals_value = {
            "m_vals": "invalid value of type string iso list of floats",
        }

        self.supp_experi_setts = supp_experi_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_error_is_thrown_if_m_vals_key_is_missing(self):
        """Verifies an exception is thrown if the m_vals key is missing from
        the MDSA algorithm settings dictionary of the supported algorithms
        dictionary of the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)

        # Remove key and value of m.
        config_settings["algorithms"]["MDSA"].pop("m_vals")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "'m_vals'",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_m_vals_value_type(self):
        """Verifies an exception is thrown if the m_vals dictionary value, is
        of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_experi_setts.algorithms["MDSA"].m_vals)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            config_settings["algorithms"]["MDSA"][
                "m_vals"
            ] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                config_settings,
                expected_type,
                self,
            )

    def test_catch_empty_m_vals_value_list(self):
        """Verifies an exception is thrown if the m_vals dictionary value is a
        list without elements."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)

        # Set negative value of m in copy.
        config_settings["algorithms"]["MDSA"]["m_vals"] = []

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_catch_m_vals_value_too_low(self):
        """Verifies an exception is thrown if the m_vals dictionary value is
        lower than the supported range of m_vals values permits."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)

        # Set negative value of m in copy.
        config_settings["algorithms"]["MDSA"]["m_vals"] = [-2]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts, config_settings, has_unique_id=False
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

    def test_catch_m_vals_value_too_high(self):
        """Verifies an exception is thrown if the m_vals dictionary value is
        higher than the supported range of m_vals values permits."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)

        # Set negative value of m in copy.
        config_settings["algorithms"]["MDSA"]["m_vals"] = [50]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts, config_settings, has_unique_id=False
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
