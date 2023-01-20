"""Verifies the Supported_experiment_settings object catches invalid simulators
specifications."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from snncompare.exp_config.Exp_config import verify_exp_config
from tests.exp_config.exp_config.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_config,
    verify_invalid_config_sett_val_throws_error,
    with_adaptation_with_radiation,
)


class Test_simulators_settings(unittest.TestCase):
    """Tests whether the verify_exp_config_types function catches invalid
    simulators settings.."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        # self.supp_exp_config = Supported_experiment_settings()

        self.invalid_simulators_value = {
            "simulators": "invalid value of type string iso list of floats",
        }

        self.supp_exp_config = supp_exp_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_simulators = self.supp_exp_config.simulators

    @typechecked
    def test_error_is_thrown_if_simulators_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the simulators key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        exp_config.pop("simulators")

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            # "'simulators'",
            "Error:simulators is not in the configuration"
            + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_simulators_value_type(self) -> None:
        """Verifies an exception is thrown if the simulators dictionary value,
        is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_exp_config.simulators)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            exp_config.simulators = invalid_config_setting_value
            verify_invalid_config_sett_val_throws_error(
                invalid_config_setting_value,
                exp_config,
                expected_type,
                self,
                True,
            )

    @typechecked
    def test_catch_empty_simulators_value_list(self) -> None:
        """Verifies an exception is thrown if the simulators dictionary value
        is a list without elements."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of simulators in copy.
        exp_config.simulators = []

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    @typechecked
    def test_catch_invalid_simulators_value(self) -> None:
        """Verifies an exception is thrown if the simulators dictionary value
        is not supported by the permissible simulators values."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of simulators in copy.
        exp_config.simulators = [
            "nx",
            "invalid_simulator_name",
            "lava",
        ]

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            "Error, simulators was expected to be in range:"
            + f"{self.supp_exp_config.simulators}."
            + " Instead, it contains:invalid_simulator_name.",
            str(context.exception),
        )
