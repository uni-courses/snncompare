"""Verifies the Supported_settings object catches invalid m specifications."""
# pylint: disable=R0801
import unittest

from src.experiment_settings import Adaptation_settings, Radiation_settings
from src.Supported_settings import Supported_settings


class Test_m_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings_types function catches
    invalid m settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_settings = Supported_settings()
        self.valid_m = self.supported_settings.m

        self.invalid_m_value = {
            "m": "invalid value of type string iso list of floats",
        }

        self.supported_settings = Supported_settings()
        self.adaptation_settings = Adaptation_settings()
        self.radiation_settings = Radiation_settings()
        self.with_adaptation_with_radiation = {
            "m": list(range(0, 1, 1)),
            "iterations": list(range(0, 3, 1)),
            "size,max_graphs": [(3, 15), (4, 15)],
            "adaptation": self.supported_settings.verify_config_setting(
                self.adaptation_settings.with_adaptation, "adaptation"
            ),
            "radiation": self.supported_settings.verify_config_setting(
                self.radiation_settings.with_radiation, "radiation"
            ),
            "overwrite": True,
            "simulators": ["nx"],
        }

    def test_m_is_none(self):
        """Verifies an error is thrown if configuration settings do not contain
        m."""

        with self.assertRaises(Exception) as context:
            # radiation dictionary of type None throws error.
            self.supported_settings.verify_configuration_settings(
                None, "radiation"
            )

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {type(None)}.",
            str(context.exception),
        )

    def test_catch_invalid_m_type(self):
        """."""
        with self.assertRaises(Exception) as context:
            # m dictionary of type None throws error.
            self.supported_settings.verify_config_setting(
                "string_instead_of_dict", "m"
            )

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {str}.",
            str(context.exception),
        )

    def test_catch_invalid_m_value_type(self):
        """."""
        with self.assertRaises(Exception) as context:
            # m dictionary of type None throws error.
            self.supported_settings.verify_config_setting(
                self.invalid_m_value, "m"
            )

        self.assertEqual(
            "Error, the m value is of type:"
            + f"{str}, yet it was expected to be"
            + " float or dict.",
            str(context.exception),
        )

    def test_returns_valid_m(self):
        """Verifies a valid m is returned."""
        returned_dict = self.supported_settings.verify_config_setting(
            self.valid_m, "m"
        )
        self.assertIsInstance(returned_dict, dict)

    def test_empty_m(self):
        """Verifies an exception is thrown if an empty m dict is thrown."""
        with self.assertRaises(Exception) as context:
            # m dictionary of type None throws error.
            self.supported_settings.verify_config_setting({}, "m")

        self.assertEqual(
            "Error, property dict: m was empty.",
            str(context.exception),
        )
