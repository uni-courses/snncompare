"""Verifies the Supported_experiment_settings object catches invalid
min_graph_size specifications."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from snncompare.exp_config.Exp_config import (
    Supported_experiment_settings,
    verify_exp_config,
)
from tests.exp_config.exp_config.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_config,
    verify_invalid_config_sett_val_throws_error,
    with_adaptation_with_radiation,
)


class Test_min_graph_size_settings(unittest.TestCase):
    """Tests whether the verify_exp_config_types function catches invalid
    min_graph_size settings.."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_exp_config = Supported_experiment_settings()
        self.valid_min_graph_size = self.supp_exp_config.min_graph_size

        self.invalid_min_graph_size_value = {
            "min_graph_size": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_exp_config = supp_exp_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    @typechecked
    def test_error_is_thrown_if_min_graph_size_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the min_graph_size key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        exp_config.pop("min_graph_size")

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                supp_exp_config=self.supp_exp_config,
                exp_config=exp_config,
            )

        self.assertEqual(
            # "'min_graph_size'",
            "Error:min_graph_size is not in the configuration"
            + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_min_graph_size_value_type(
        self,
    ) -> None:
        """Verifies an exception is thrown if the min_graph_size dictionary
        value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_exp_config.min_graph_size)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            exp_config.min_graph_size = invalid_config_setting_value
            verify_invalid_config_sett_val_throws_error(
                invalid_config_setting_value=invalid_config_setting_value,
                exp_config=exp_config,
                expected_type=expected_type,
                test_object=self,
            )

    # TODO: test_catch_empty_min_graph_size_value_list

    @typechecked
    def test_catch_min_graph_size_value_too_low(self) -> None:
        """Verifies an exception is thrown if the min_graph_size dictionary
        value is lower than the supported range of min_graph_size values
        permits."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of min_graph_size in copy.
        exp_config.min_graph_size = -2

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                supp_exp_config=self.supp_exp_config,
                exp_config=exp_config,
            )

        self.assertEqual(
            "Error, setting expected to be at least "
            + f"{self.supp_exp_config.min_graph_size}. "
            + f"Instead, it is:{-2}",
            str(context.exception),
        )

    @typechecked
    def test_catch_min_graph_size_value_too_high(self) -> None:
        """Verifies an exception is thrown if the min_graph_size dictionary
        value is higher than the supported range of min_graph_size values
        permits."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of min_graph_size in copy.
        exp_config.min_graph_size = 50

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                supp_exp_config=self.supp_exp_config,
                exp_config=exp_config,
            )

        self.assertEqual(
            "Error, setting expected to be at most "
            + f"{self.supp_exp_config.max_graph_size}. Instead, it is:"
            + "50",
            str(context.exception),
        )
