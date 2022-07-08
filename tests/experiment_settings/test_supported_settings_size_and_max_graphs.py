"""Verifies the Supported_experiment_settings object catches invalid
size_and_max_graphs specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.experiment_settings.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_experi_config,
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_size_and_max_graphs_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid size_and_max_graphs settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.supp_experi_config = Supported_experiment_settings()

        self.invalid_size_and_max_graphs_value = {
            "size_and_max_graphs": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_experi_config = supp_experi_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_size_and_max_graphs = (
            self.supp_experi_config.size_and_max_graphs
        )

    def test_error_is_thrown_if_size_and_max_graphs_key_is_missing(self):
        """Verifies an exception is thrown if the size_and_max_graphs key is
        missing from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.
        experi_config.pop("size_and_max_graphs")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_config, experi_config, has_unique_id=False
            )

        self.assertEqual(
            # "'size_and_max_graphs'",
            "Error:size_and_max_graphs is not in the configuration"
            + f" settings:{experi_config.keys()}",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_size_and_max_graphs_value_type(self):
        """Verifies an exception is thrown if the size_and_max_graphs
        dictionary value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_experi_config.size_and_max_graphs)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            experi_config["size_and_max_graphs"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                experi_config,
                expected_type,
                self,
            )

    def test_catch_empty_size_and_max_graphs_value_list(self):
        """Verifies an exception is thrown if the size_and_max_graphs
        dictionary value is a list without elements."""
        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of size_and_max_graphs in copy.
        experi_config["size_and_max_graphs"] = []

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_config, experi_config, has_unique_id=False
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_catch_size_and_max_graphs_value_too_low(self):
        """Verifies an exception is thrown if the size_and_max_graphs
        dictionary value is lower than the supported range of
        size_and_max_graphs values permits."""
        # Create deepcopy of configuration settings.
        experi_config_first = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        experi_config_second = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        # Set negative value of size_and_max_graphs in copy.
        experi_config_first["size_and_max_graphs"] = [
            (2, self.supp_experi_config.max_max_graphs),
            (
                self.supp_experi_config.min_graph_size,
                self.supp_experi_config.max_max_graphs,
            ),
        ]
        experi_config_second["size_and_max_graphs"] = [
            (
                self.supp_experi_config.min_graph_size,
                self.supp_experi_config.max_max_graphs,
            ),
            (-2, self.supp_experi_config.max_max_graphs),
        ]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_config,
                experi_config_first,
                has_unique_id=False,
            )

        self.assertEqual(
            "Error, setting expected to be at least"
            + f" {self.supp_experi_config.min_graph_size}. Instead, it is:"
            + f"{2}",
            str(context.exception),
        )

        # Verify it catches the too large graph_size at the second tuple as
        # well.
        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_config,
                experi_config_second,
                has_unique_id=False,
            )

        self.assertEqual(
            "Error, setting expected to be at least"
            + f" {self.supp_experi_config.min_graph_size}. Instead, it is:"
            + f"{-2}",
            str(context.exception),
        )

    def test_catch_size_and_max_graphs_value_too_high(self):
        """Verifies an exception is thrown if the size_and_max_graphs
        dictionary value is higher than the supported range of
        size_and_max_graphs values permits."""
        # Create deepcopy of configuration settings.
        experi_config_first = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        experi_config_second = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        # Set the desired graph size to 50, which is larger than allowed in
        # self.supp_experi_config.max_graph_size. The max_graphs is set to the
        # maximum which is acceptable.
        experi_config_first["size_and_max_graphs"] = [
            (50, self.supp_experi_config.max_max_graphs),
            (
                self.supp_experi_config.min_graph_size,
                self.supp_experi_config.max_max_graphs,
            ),
        ]
        experi_config_second["size_and_max_graphs"] = [
            (
                self.supp_experi_config.min_graph_size,
                self.supp_experi_config.max_max_graphs,
            ),
            (42, self.supp_experi_config.max_max_graphs),
        ]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_config,
                experi_config_first,
                has_unique_id=False,
            )

        self.assertEqual(
            "Error, setting expected to be at most"
            + f" {self.supp_experi_config.max_graph_size}. Instead, it is:"
            + f"{50}",
            str(context.exception),
        )

        # Verify it catches the too large graph_size at the second tuple as
        # well.
        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_config,
                experi_config_second,
                has_unique_id=False,
            )

        self.assertEqual(
            "Error, setting expected to be at most"
            + f" {self.supp_experi_config.max_graph_size}. Instead, it is:"
            + f"{42}",
            str(context.exception),
        )
