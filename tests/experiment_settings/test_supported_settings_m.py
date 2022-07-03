"""Verifies the Supported_settings object catches invalid m specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.experiment_settings import (
    Adaptation_settings,
    Radiation_settings,
)
from src.experiment_settings.Supported_settings import Supported_settings
from src.experiment_settings.verify_supported_settings import (
    verify_adap_and_rad_settings,
    verify_configuration_settings,
)


class Test_m_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings_types function catches
    invalid m settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_sets = Supported_settings()
        self.valid_m = self.supp_sets.m

        self.invalid_m_value = {
            "m": "invalid value of type string iso list of floats",
        }

        self.supp_sets = Supported_settings()
        self.adap_sets = Adaptation_settings()
        self.rad_sets = Radiation_settings()
        self.with_adaptation_with_radiation = {
            "m": list(range(0, 1, 1)),
            "iterations": list(range(0, 3, 1)),
            "size,max_graphs": [(3, 15), (4, 15)],
            "adaptation": verify_adap_and_rad_settings(
                self.supp_sets, self.adap_sets.with_adaptation, "adaptation"
            ),
            "radiation": verify_adap_and_rad_settings(
                self.supp_sets, self.rad_sets.with_radiation, "radiation"
            ),
            "overwrite": True,
            "simulators": ["nx"],
        }

    def test_m_is_none(self):
        """Verifies an error is thrown if configuration settings do not contain
        m."""

        with self.assertRaises(Exception) as context:
            # Configuration Settings of type None throw error.
            verify_configuration_settings(self.supp_sets, None, "radiation")

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
            verify_configuration_settings(
                self.supp_sets, "string_instead_of_dict", False
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
            verify_configuration_settings(
                self.supp_sets, config_settings, False
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
        print(f"config_settings={config_settings}")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, False
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
            verify_configuration_settings(
                self.supp_sets, config_settings, False
            )

        self.assertEqual(
            # "Error, m was expected contain at least 1 integer."
            # + f" Instead, it has length:{0}",
            f"Error, obj={[]}, its type is:[]," + f" expected type:{[int]}",
            str(context.exception),
        )

    def test_returns_valid_m(self):
        """Verifies a valid m is returned."""
        returned_dict = verify_configuration_settings(
            self.supp_sets, self.with_adaptation_with_radiation, "m"
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
            verify_configuration_settings(
                self.supp_sets, config_settings, False
            )

        self.assertEqual(
            "'m'",
            str(context.exception),
        )
