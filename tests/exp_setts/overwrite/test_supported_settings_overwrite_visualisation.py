"""Verifies the Supported_experiment_settings object catches invalid
overwrite_visualisation specifications."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from src.snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.exp_setts.exp_setts.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_setts,
    with_adaptation_with_radiation,
)


class Test_overwrite_visualisation_settings(unittest.TestCase):
    """Tests whether the verify_experiment_config_types function catches
    invalid overwrite_visualisation settings.."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_exp_setts = Supported_experiment_settings()
        self.valid_overwrite_visualisation = (
            self.supp_exp_setts.overwrite_visualisation
        )

        self.invalid_overwrite_visualisation_value = {
            "overwrite_visualisation": "invalid value of type string iso list"
            + " of floats",
        }

        self.supp_exp_setts = supp_exp_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    @typechecked
    def test_error_is_thrown_if_overwrite_visualisation_key_is_missing(
        self,
    ) -> None:
        """Verifies an exception is thrown if the overwrite_visualisation key
        is missing from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        experiment_config.pop("overwrite_visualisation")

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            # "'overwrite_visualisation'",
            "Error:overwrite_visualisation is not in the configuration"
            + f" settings:{experiment_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_overwrite_visualisation_value_type(
        self,
    ) -> None:
        """Verifies an exception is thrown if the overwrite_visualisation
        dictionary value, is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of overwrite_visualisation in copy.

        # TODO: generalise to also check if an error is thrown if it contains a
        # string or integer, using the generic test file.
        # verify_error_is_thrown_on_invalid_configuration_setting_value
        experiment_config["overwrite_visualisation"] = None

        with self.assertRaises(Exception) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            # "Error, expected type:<class 'bool'>, yet it was:"
            # + f"{type(None)} for:{None}",
            'type of argument "bool_setting" must be bool; got NoneType '
            "instead",
            str(context.exception),
        )
