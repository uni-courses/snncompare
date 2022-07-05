"""Verifies the Supported_settings object catches invalid
overwrite_visualisation specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.Supported_settings import Supported_settings
from src.experiment_settings.verify_supported_settings import (
    verify_configuration_settings,
)
from tests.experiment_settings.test_generic_configuration import (
    adap_sets,
    rad_sets,
    supp_sets,
    with_adaptation_with_radiation,
)


class Test_overwrite_visualisation_settings(unittest.TestCase):
    """Tests whether the verify_configuration_settings_types function catches
    invalid overwrite_visualisation settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_sets = Supported_settings()
        self.valid_overwrite_visualisation = (
            self.supp_sets.overwrite_visualisation
        )

        self.invalid_overwrite_visualisation_value = {
            "overwrite_visualisation": "invalid value of type string iso list"
            + " of floats",
        }

        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    def test_catch_empty_overwrite_visualisation_value(self):
        """."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of overwrite_visualisation in copy.
        config_settings["overwrite_visualisation"] = None

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            "Error, expected type:<class 'bool'>, yet it was:"
            + f"{type(None)} for:{None}",
            str(context.exception),
        )

    def test_empty_overwrite_visualisation(self):
        """Verifies an exception is thrown if an empty overwrite_visualisation
        dict is thrown."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        config_settings.pop("overwrite_visualisation")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            # "'overwrite_visualisation'",
            "Error:overwrite_visualisation is not in the configuration"
            + f" settings:{config_settings.keys()}",
            str(context.exception),
        )
