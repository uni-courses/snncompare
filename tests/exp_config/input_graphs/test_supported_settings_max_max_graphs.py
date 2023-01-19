"""Verifies the Supported_experiment_settings object catches invalid
max_max_graphs specifications."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from snncompare.exp_config.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_config.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.exp_config.exp_config.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_config,
    verify_invalid_config_sett_val_throws_error,
    with_adaptation_with_radiation,
)


class Test_max_max_graphs_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid max_max_graphs settings.."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_exp_config = Supported_experiment_settings()
        self.valid_max_max_graphs = self.supp_exp_config.max_max_graphs

        self.invalid_max_max_graphs_value = {
            "max_max_graphs": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_exp_config = supp_exp_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    @typechecked
    def test_error_is_thrown_if_max_max_graphs_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the max_max_graphs key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        experiment_config.pop("max_max_graphs")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_config,
                experiment_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            # "'max_max_graphs'",
            "Error:max_max_graphs is not in the configuration"
            + f" settings:{experiment_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_max_max_graphs_value_type(
        self,
    ) -> None:
        """Verifies an exception is thrown if the max_max_graphs dictionary
        value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_exp_config.max_max_graphs)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            experiment_config["max_max_graphs"] = invalid_config_setting_value
            verify_invalid_config_sett_val_throws_error(
                invalid_config_setting_value,
                experiment_config,
                expected_type,
                self,
            )

    @typechecked
    def test_catch_invalid_max_max_graphs_value_type_too_low(self) -> None:
        """Verifies an exception is thrown if the max_max_graphs dictionary
        value is lower than the supported range of max_max_graphs values
        permits."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_max_graphs in copy.
        experiment_config["max_max_graphs"] = -2

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_config,
                experiment_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            "Error, setting expected to be at least "
            + f"{self.supp_exp_config.min_max_graphs}. "
            + f"Instead, it is:{-2}",
            str(context.exception),
        )

    @typechecked
    def test_catch_max_max_graphs_is_smaller_than_min_max_graphs(self) -> None:
        """Verifies an exception is thrown if the max_max_graphs value is
        smaller than the min_max_graphs value."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_max_graphs in copy.
        experiment_config["min_max_graphs"] = (
            experiment_config["min_max_graphs"] + 1
        )
        experiment_config["max_max_graphs"] = (
            experiment_config["min_max_graphs"] - 1
        )

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_config,
                experiment_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            f'Lower bound:{experiment_config["min_max_graphs"]} is larger than'
            f' upper bound:{experiment_config["max_max_graphs"]}.',
            str(context.exception),
        )

    @typechecked
    def test_catch_max_max_graphs_value_too_high(self) -> None:
        """Verifies an exception is thrown if the max_max_graphs dictionary
        value is higher than the supported range of max_max_graphs values
        permits."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_max_graphs in copy.
        experiment_config["max_max_graphs"] = 50

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_config,
                experiment_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            "Error, setting expected to be at most "
            + f"{self.supp_exp_config.max_max_graphs}. Instead, it is:"
            + "50",
            str(context.exception),
        )
