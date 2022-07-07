"""Verifies the Supported_settings object catches invalid seed
specifications."""
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
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_seed_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings_types function catches
    invalid seed settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_sets = Supported_settings()
        self.valid_seed = self.supp_sets.seed

        self.invalid_seed_value = {
            "seed": "invalid value of type string iso list of" + " floats",
        }

        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_error_is_thrown_if_seed_key_is_missing(self):
        """Verifies an exception is thrown if the seed key is missing from the
        configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        config_settings.pop("seed")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            # "'seed'",
            "Error:seed is not in the configuration"
            + f" settings:{config_settings.keys()}",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_seed_value_type(self):
        """Verifies an exception is thrown if the seed dictionary value, is of
        invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_sets.seed)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            config_settings["seed"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                config_settings,
                expected_type,
                self,
            )
