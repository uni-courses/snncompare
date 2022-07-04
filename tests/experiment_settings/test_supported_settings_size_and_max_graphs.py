"""Verifies the Supported_settings object catches invalid size_and_max_graphs
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.verify_supported_settings import (
    verify_configuration_settings,
)
from tests.experiment_settings.test_supported_settings_iteration import (
    adap_sets,
    rad_sets,
    supp_sets,
    with_adaptation_with_radiation,
)


class Test_size_and_max_graphs_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings_types function catches
    invalid size_and_max_graphs settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.supp_sets = Supported_settings()

        self.invalid_size_and_max_graphs_value = {
            "size_and_max_graphs": "invalid value of type string iso list of"
            + " floats",
        }

        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_size_and_max_graphs = self.supp_sets.size_and_max_graphs

    def test_size_and_max_graphs_is_none(self):
        """Verifies an error is thrown if configuration settings do not contain
        m."""

        with self.assertRaises(Exception) as context:
            # Configuration Settings of type None throw error.
            verify_configuration_settings(
                self.supp_sets, None, has_unique_id=False
            )

        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f"{type(None)}, yet it was expected to be of"
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_size_and_max_graphs_type(self):
        """."""

        with self.assertRaises(Exception) as context:
            # size_and_max_graphs dictionary of type None throws error.
            verify_configuration_settings(
                self.supp_sets, "string_instead_of_dict", has_unique_id=False
            )
        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f'{type("")}, yet it was expected to be of'
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_size_and_max_graphs_value_type_too_low(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings_first = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        config_settings_second = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        # Set negative value of size_and_max_graphs in copy.
        config_settings_first["size_and_max_graphs"] = [
            (2, self.supp_sets.max_max_graphs),
            (self.supp_sets.min_graph_size, self.supp_sets.max_max_graphs),
        ]
        config_settings_second["size_and_max_graphs"] = [
            (self.supp_sets.min_graph_size, self.supp_sets.max_max_graphs),
            (-2, self.supp_sets.max_max_graphs),
        ]

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings_first, has_unique_id=False
            )

        self.assertEqual(
            "Error, setting expected to be at least"
            + f" {self.supp_sets.min_graph_size}. Instead, it is:"
            + f"{2}",
            str(context.exception),
        )

        # Verify it catches the too large graph_size at the second tuple as
        # well.
        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings_second, has_unique_id=False
            )

        self.assertEqual(
            "Error, setting expected to be at least"
            + f" {self.supp_sets.min_graph_size}. Instead, it is:"
            + f"{-2}",
            str(context.exception),
        )

    def test_catch_invalid_size_and_max_graphs_value_size_too_high(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings_first = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        config_settings_second = copy.deepcopy(
            self.with_adaptation_with_radiation
        )
        # Set the desired graph size to 50, which is larger than allowed in
        # self.supp_sets.max_graph_size. The max_graphs is set to the maximum
        # which is acceptable.
        config_settings_first["size_and_max_graphs"] = [
            (50, self.supp_sets.max_max_graphs),
            (self.supp_sets.min_graph_size, self.supp_sets.max_max_graphs),
        ]
        config_settings_second["size_and_max_graphs"] = [
            (self.supp_sets.min_graph_size, self.supp_sets.max_max_graphs),
            (42, self.supp_sets.max_max_graphs),
        ]

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings_first, has_unique_id=False
            )

        self.assertEqual(
            "Error, setting expected to be at most"
            + f" {self.supp_sets.max_graph_size}. Instead, it is:"
            + f"{50}",
            str(context.exception),
        )

        # Verify it catches the too large graph_size at the second tuple as
        # well.
        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings_second, has_unique_id=False
            )

        self.assertEqual(
            "Error, setting expected to be at most"
            + f" {self.supp_sets.max_graph_size}. Instead, it is:"
            + f"{42}",
            str(context.exception),
        )

    def test_catch_empty_size_and_max_graphs_value_list(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of size_and_max_graphs in copy.
        config_settings["size_and_max_graphs"] = []

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_returns_valid_m(self):
        """Verifies a valid size_and_max_graphs is returned."""
        returned_dict = verify_configuration_settings(
            self.supp_sets,
            self.with_adaptation_with_radiation,
            has_unique_id=False,
        )
        self.assertIsInstance(returned_dict, dict)

    def test_empty_size_and_max_graphs(self):
        """Verifies an exception is thrown if an empty size_and_max_graphs dict
        is thrown."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.
        config_settings.pop("size_and_max_graphs")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "'size_and_max_graphs'",
            str(context.exception),
        )
