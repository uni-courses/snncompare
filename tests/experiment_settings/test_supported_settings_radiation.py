"""Verifies The Supported_settings object catches invalid radiation
specifications."""
# pylint: disable=R0801
import unittest

from src.experiment_settings.Supported_settings import Supported_settings
from src.experiment_settings.verify_supported_settings import (
    verify_adap_and_rad_settings,
)


class Test_radiation_settings(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_sets = Supported_settings()
        self.valid_radiation = self.supp_sets.radiation

        self.invalid_radiation_value = {
            "neuron_death": "invalid value of type string iso list",
        }

        self.invalid_radiation_key = {"non-existing-key": 5}

    # TODO: write test that verifies an error is thrown if the radiation key
    # is not set.
    def test_catch_radiation_is_none(self):
        """Verifies if an error is thrown if the value belonging to the
        radiation key in the configuration settings has value: None.

        (The value should be a dict.)
        """

        with self.assertRaises(Exception) as context:
            # radiation dictionary of type None throws error.
            verify_adap_and_rad_settings(self.supp_sets, None, "radiation")

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {type(None)}.",
            str(context.exception),
        )

    def test_catch_invalid_radiation_type(self):
        """Verifies if an error is thrown if the value belonging to the
        adaptation key in the configuration settings has a value of type
        string.

        (The value should be a dict.)
        """
        with self.assertRaises(Exception) as context:
            # radiation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                self.supp_sets, "string_instead_of_dict", "radiation"
            )

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {str}.",
            str(context.exception),
        )

    def test_catch_invalid_radiation_key(self):
        """."""

        with self.assertRaises(Exception) as context:
            # radiation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                self.supp_sets, self.invalid_radiation_key, "radiation"
            )

        self.assertEqual(
            "Error, property.key:non-existing-key is not in the supported "
            + f"property keys:{self.supp_sets.radiation.keys()}.",
            str(context.exception),
        )

    def test_catch_invalid_radiation_value_type(self):
        """."""
        with self.assertRaises(Exception) as context:
            # radiation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                self.supp_sets, self.invalid_radiation_value, "radiation"
            )

        self.assertEqual(
            "Error, the radiation value is of type:"
            + f"{str}, yet it was expected to be"
            + " float or dict.",
            str(context.exception),
        )

    def test_returns_valid_radiation(self):
        """Verifies a valid radiation is returned."""
        returned_dict = verify_adap_and_rad_settings(
            self.supp_sets, self.valid_radiation, "radiation"
        )
        self.assertIsInstance(returned_dict, dict)

    def test_empty_radiation(self):
        """Verifies an exception is thrown if an empty radiation dict is
        thrown."""
        with self.assertRaises(Exception) as context:
            # radiation dictionary of type None throws error.
            verify_adap_and_rad_settings(self.supp_sets, {}, "radiation")

        self.assertEqual(
            "Error, property dict: radiation was empty.",
            str(context.exception),
        )
