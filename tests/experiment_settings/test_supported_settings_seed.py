"""Verifies the Supported_experiment_settings object catches invalid seed
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.snn_algo_compare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.snn_algo_compare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.experiment_settings.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_experi_setts,
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_seed_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid seed settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_experi_setts = Supported_experiment_settings()
        self.valid_seed = self.supp_experi_setts.seed

        self.invalid_seed_value = {
            "seed": "invalid value of type string iso list of" + " floats",
        }

        self.supp_experi_setts = supp_experi_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_error_is_thrown_if_seed_key_is_missing(self):
        """Verifies an exception is thrown if the seed key is missing from the
        configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        experiment_config.pop("seed")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            # "'seed'",
            "Error:seed is not in the configuration"
            + f" settings:{experiment_config.keys()}",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_seed_value_type(self):
        """Verifies an exception is thrown if the seed dictionary value, is of
        invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_experi_setts.seed)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            experiment_config["seed"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                experiment_config,
                expected_type,
                self,
            )
