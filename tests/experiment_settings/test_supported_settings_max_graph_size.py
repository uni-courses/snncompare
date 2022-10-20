"""Verifies the Supported_experiment_settings object catches invalid
max_graph_size specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.experiment_settings.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.experiment_settings.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_experi_setts,
    with_adaptation_with_radiation,
)


class Test_max_graph_size_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid max_graph_size settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_experi_setts = Supported_experiment_settings()
        self.valid_max_graph_size = self.supp_experi_setts.max_graph_size

        self.invalid_max_graph_size_value = {
            "max_graph_size": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_experi_setts = supp_experi_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_error_is_thrown_if_max_graph_size_key_is_missing(self):
        """Verifies an exception is thrown if the max_graph_size key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        experiment_config.pop("max_graph_size")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            # "'max_graph_size'",
            "Error:max_graph_size is not in the configuration"
            + f" settings:{experiment_config.keys()}",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_max_graph_size_value_type(self):
        """Verifies an exception is thrown if the max_graph_size dictionary
        value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        experiment_config["max_graph_size"] = None

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, expected type:<class 'int'>, yet it was:"
            + f"{type(None)} for:{None}",
            str(context.exception),
        )

    def test_catch_max_graph_size_value_too_low(self):
        """Verifies an exception is thrown if the max_graph_size dictionary
        value is lower than the supported range of max_graph_size values
        permits."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        experiment_config["max_graph_size"] = -2

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, setting expected to be at least "
            + f"{self.supp_experi_setts.min_graph_size}. "
            + f"Instead, it is:{-2}",
            str(context.exception),
        )

    def test_catch_max_graph_size_is_smaller_than_min_graph_size(self):
        """To state the obvious, this also tests whether min_graph_size is
        larger than max_graph size throws an exception."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        experiment_config["min_graph_size"] = (
            experiment_config["min_graph_size"] + 1
        )
        experiment_config["max_graph_size"] = (
            experiment_config["min_graph_size"] - 1
        )

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            f'Lower bound:{experiment_config["min_graph_size"]} is larger than'
            f' upper bound:{experiment_config["max_graph_size"]}.',
            str(context.exception),
        )

    def test_catch_max_graph_size_value_too_high(self):
        """Verifies an exception is thrown if the max_graph_size dictionary
        value is higher than the supported range of max_graph_size values
        permits."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of max_graph_size in copy.
        experiment_config["max_graph_size"] = 50

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, setting expected to be at most "
            + f"{self.supp_experi_setts.max_graph_size}. Instead, it is:"
            + "50",
            str(context.exception),
        )
