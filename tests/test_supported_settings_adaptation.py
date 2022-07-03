"""Verifies The Supported_settings object catches invalid adaptation
specifications."""
import unittest

from src.Supported_settings import Supported_settings


class Test_adaptation_settings(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_settings = Supported_settings()
        self.valid_adaptation = {
            "redundancy": [1.0, 2.0],  # Create 1 and 2 redundant neuron(s) per
            # neuron.
            "population": [
                10.0
            ],  # Create a population of 10 neurons to represent a
            # single neuron.
            "rate_coding": [
                5.0
            ],  # Multiply firing frequency with 5 to limit spike decay
            # impact.
        }

        self.invalid_adaptation_value = {
            "redundancy": "invalid value of type string iso list",
        }

        self.invalid_adaptation_key = {"non-existing-key": 5}

    def test_catch_adaptation_is_none(self):
        """."""

        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            self.supported_settings.verify_adap_and_rad_settings(
                None, "adaptation"
            )

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {type(None)}.",
            str(context.exception),
        )

    def test_catch_invalid_adaptation_type(self):
        """."""
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            self.supported_settings.verify_adap_and_rad_settings(
                "string_instead_of_dict",
                "adaptation",
            )

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {str}.",
            str(context.exception),
        )

    def test_catch_invalid_adaptation_key(self):
        """."""

        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            self.supported_settings.verify_adap_and_rad_settings(
                self.invalid_adaptation_key, "adaptation"
            )

        self.assertEqual(
            "Error, property.key:non-existing-key is not in the supported "
            + f"property keys:{self.supported_settings.adaptation.keys()}.",
            str(context.exception),
        )

    def test_catch_invalid_adaptation_value_type(self):
        """."""
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            self.supported_settings.verify_adap_and_rad_settings(
                self.invalid_adaptation_value, "adaptation"
            )

        self.assertEqual(
            'Error, value of adaptation["redundancy"]='
            + f"invalid value of type string iso list, (which has type:{str})"
            + ", is of different type than the expected and supported type: "
            + f"{list}",
            str(context.exception),
        )

    def test_returns_valid_adaptation(self):
        """TODO: verify dict is returned for valid adaptation."""
        returned_dict = self.supported_settings.verify_adap_and_rad_settings(
            self.valid_adaptation, "adaptation"
        )
        self.assertIsInstance(returned_dict, dict)

    def test_empty_adaptation(self):
        """Verifies an exception is thrown if an empty adaptation dict is
        thrown."""
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            self.supported_settings.verify_adap_and_rad_settings(
                {}, "adaptation"
            )

        self.assertEqual(
            "Error, property dict: adaptation was empty.",
            str(context.exception),
        )
