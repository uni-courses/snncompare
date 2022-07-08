"""Verifies the Supported_experiment_settings object catches invalid simulators
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.experiment_settings.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_experi_setts,
    verify_error_is_thrown_on_invalid_configuration_setting_value,
    with_adaptation_with_radiation,
)


class Test_simulators_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid simulators settings.."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.supp_experi_setts = Supported_experiment_settings()

        self.invalid_simulators_value = {
            "simulators": "invalid value of type string iso list of floats",
        }

        self.supp_experi_setts = supp_experi_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_simulators = self.supp_experi_setts.simulators

    def test_error_is_thrown_if_simulators_key_is_missing(self):
        """Verifies an exception is thrown if the simulators key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        experi_config.pop("simulators")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experi_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            # "'simulators'",
            "Error:simulators is not in the configuration"
            + f" settings:{experi_config.keys()}",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_simulators_value_type(self):
        """Verifies an exception is thrown if the simulators dictionary value,
        is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """

        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        expected_type = type(self.supp_experi_setts.simulators)

        # Verify it throws an error on None and string.
        for invalid_config_setting_value in [None, ""]:
            experi_config["simulators"] = invalid_config_setting_value
            verify_error_is_thrown_on_invalid_configuration_setting_value(
                invalid_config_setting_value,
                experi_config,
                expected_type,
                self,
            )

    def test_catch_empty_simulators_value_list(self):
        """Verifies an exception is thrown if the simulators dictionary value
        is a list without elements."""
        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of simulators in copy.
        experi_config["simulators"] = []

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experi_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{0}",
            str(context.exception),
        )

    def test_catch_invalid_simulators_value(self):
        """Verifies an exception is thrown if the simulators dictionary value
        is not supported by the permissible simulators values."""
        # Create deepcopy of configuration settings.
        experi_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of simulators in copy.
        experi_config["simulators"] = [
            "nx",
            "invalid_simulator_name",
            "lava",
        ]

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_experi_setts,
                experi_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "Error, simulators was expected to be in range:"
            + f"{self.supp_experi_setts.simulators}."
            + " Instead, it contains:invalid_simulator_name.",
            str(context.exception),
        )
