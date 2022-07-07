"""Verifies the Supported_experiment_settings object catches invalid
overwrite_sim_results specifications."""
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


class Test_overwrite_sim_results_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid overwrite_sim_results settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_experi_setts = Supported_experiment_settings()
        self.valid_overwrite_sim_results = (
            self.supp_experi_setts.overwrite_sim_results
        )

        self.invalid_overwrite_sim_results_value = {
            "overwrite_sim_results": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_experi_setts = supp_experi_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_error_is_thrown_if_overwrite_sim_results_key_is_missing(self):
        """Verifies an exception is thrown if the overwrite_sim_results key is
        missing from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)

        # Remove key and value of m.
        config_settings.pop("overwrite_sim_results")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts, config_settings, has_unique_id=False
            )

        self.assertEqual(
            # "'overwrite_sim_results'",
            "Error:overwrite_sim_results is not in the configuration"
            + f" settings:{config_settings.keys()}",
            str(context.exception),
        )

    def test_overwrite_sim_results_value_is_invalid_type(self):
        """Verifies an exception is thrown if the overwrite_sim_results
        dictionary value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_experi_setts.overwrite_sim_results)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            config_settings[
                "overwrite_sim_results"
            ] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                config_settings,
                expected_type,
                self,
            )
