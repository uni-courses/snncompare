"""Verifies the Supported_experiment_settings object catches invalid
recreate_s4 specifications."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from snncompare.exp_config.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_config.verify_experiment_settings import verify_exp_config
from tests.exp_config.exp_config.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_config,
    verify_invalid_config_sett_val_throws_error,
    with_adaptation_with_radiation,
)


class Test_recreate_s4_settings(unittest.TestCase):
    """Tests whether the verify_exp_config_types function catches invalid
    recreate_s4 settings.."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_exp_config = Supported_experiment_settings()
        self.valid_recreate_s4 = self.supp_exp_config.recreate_s4

        self.invalid_recreate_s4_value = {
            "recreate_s4": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_exp_config = supp_exp_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    @typechecked
    def test_error_is_thrown_if_recreate_s4_key_is_missing(
        self,
    ) -> None:
        """Verifies an exception is thrown if the recreate_s4 key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)

        # Remove key and value of m.
        exp_config.pop("recreate_s4")

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            # "'recreate_s4'",
            "Error:recreate_s4 is not in the configuration"
            + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_recreate_s4_value_is_invalid_type(
        self,
    ) -> None:
        """Verifies an exception is thrown if the recreate_s4 dictionary value,
        is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_exp_config.recreate_s4)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            exp_config.recreate_s4 = invalid_config_setting_value
            verify_invalid_config_sett_val_throws_error(
                invalid_config_setting_value,
                exp_config,
                expected_type,
                self,
                alternative_var_name="bool_setting",
            )
