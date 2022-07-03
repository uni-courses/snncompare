"""Verifies the Supported_settings object catches invalid m specifications."""
# pylint: disable=R0801
import copy
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
            # Configuration Settings of type None throw error.
            self.supported_settings.verify_configuration_settings(
                None, "radiation"
            )

        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f"{type(None)}, yet it was expected to be of"
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_m_type(self):
        """."""

        with self.assertRaises(Exception) as context:
            # m dictionary of type None throws error.
            self.supported_settings.verify_configuration_settings(
                "string_instead_of_dict", False
            )
        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f'{type("")}, yet it was expected to be of'
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_m_value_type_too_low(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of m in copy.
        config_settings["m"] = [-2]

        with self.assertRaises(Exception) as context:
            self.supported_settings.verify_configuration_settings(
                config_settings, False
            )

        self.assertEqual(
            "Error, m was expected to be in range:"
            + f'{self.with_adaptation_with_radiation["m"]}.'
            + f" Instead, it contains:{-2}.",
            str(context.exception),
        )

    def test_catch_invalid_m_value_type_too_high(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of m in copy.
        config_settings["m"] = [50]

        with self.assertRaises(Exception) as context:
            self.supported_settings.verify_configuration_settings(
                config_settings, False
            )

        self.assertEqual(
            "Error, m was expected to be in range:"
            + f'{self.with_adaptation_with_radiation["m"]}.'
            + f" Instead, it contains:{50}.",
            str(context.exception),
        )

    def test_catch_empty_m_value_list(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of m in copy.
        config_settings["m"] = []

        with self.assertRaises(Exception) as context:
            self.supported_settings.verify_configuration_settings(
                config_settings, False
            )

        self.assertEqual(
            "Error, m was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_returns_valid_m(self):
        """Verifies a valid m is returned."""
        returned_dict = self.supported_settings.verify_configuration_settings(
            self.with_adaptation_with_radiation, "m"
        )
        self.assertIsInstance(returned_dict, dict)

    def test_empty_m(self):
        """Verifies an exception is thrown if an empty m dict is thrown."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.
        print(f"Before config_settings={config_settings}")
        config_settings.pop("m")
        print(f"After config_settings={config_settings}")

        with self.assertRaises(Exception) as context:
            # m dictionary of type None throws error.
            # TODO: rename verify_config_setting to: something else.
            # TODO: rename something else to verify_configuration_settings
            # in this function.
            # self.supported_settings.verify_config_setting({}, "m")
            self.supported_settings.verify_configuration_settings(
                config_settings, False
            )

        self.assertEqual(
            "'m'",
            str(context.exception),
        )
