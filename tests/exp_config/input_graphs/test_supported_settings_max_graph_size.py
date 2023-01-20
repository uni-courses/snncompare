"""Verifies the Supported_experiment_settings object catches invalid
max_graph_size specifications."""
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
    with_adaptation_with_radiation,
)


class Test_max_graph_size_settings(unittest.TestCase):
    """Tests whether the verify_exp_config_types function catches invalid
    max_graph_size settings.."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_exp_config = Supported_experiment_settings()
        self.valid_max_graph_size = self.supp_exp_config.max_graph_size

        self.invalid_max_graph_size_value = {
            "max_graph_size": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_exp_config = supp_exp_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    @typechecked
    def test_error_is_thrown_if_max_graph_size_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the max_graph_size key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        exp_config.pop("max_graph_size")

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            # "'max_graph_size'",
            "Error:max_graph_size is not in the configuration"
            + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_max_graph_size_value_type(
        self,
    ) -> None:
        """Verifies an exception is thrown if the max_graph_size dictionary
        value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        exp_config.max_graph_size = None

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            # "Error, expected type:<class 'int'>, yet it was:"
            # + f"{type(None)} for:{None}",
            'type of argument "integer_setting" must be int; got NoneType '
            "instead",
            str(context.exception),
        )

    @typechecked
    def test_catch_max_graph_size_value_too_low(self) -> None:
        """Verifies an exception is thrown if the max_graph_size dictionary
        value is lower than the supported range of max_graph_size values
        permits."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        exp_config.max_graph_size = -2

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            "Error, setting expected to be at least "
            + f"{self.supp_exp_config.min_graph_size}. "
            + f"Instead, it is:{-2}",
            str(context.exception),
        )

    @typechecked
    def test_catch_max_graph_size_is_smaller_than_min_graph_size(self) -> None:
        """To state the obvious, this also tests whether min_graph_size is
        larger than max_graph size throws an exception."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        exp_config.min_graph_size = exp_config.min_graph_size + 1
        exp_config.max_graph_size = exp_config.min_graph_size - 1

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            f"Lower bound:{exp_config.min_graph_size} is larger than"
            f" upper bound:{exp_config.max_graph_size}.",
            str(context.exception),
        )

    @typechecked
    def test_catch_max_graph_size_value_too_high(self) -> None:
        """Verifies an exception is thrown if the max_graph_size dictionary
        value is higher than the supported range of max_graph_size values
        permits."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        exp_config.max_graph_size = 50

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                self.supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            "Error, setting expected to be at most "
            + f"{self.supp_exp_config.max_graph_size}. Instead, it is:"
            + "50",
            str(context.exception),
        )
