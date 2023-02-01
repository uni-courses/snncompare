"""Verifies the Supported_experiment_settings object catches invalid
recreate_s3 specifications."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from snncompare.exp_config.Exp_config import (
    Supported_experiment_settings,
    verify_exp_config,
)
from tests.exp_config.exp_config.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_config,
    with_adaptation_with_radiation,
)


class Test_recreate_s3_settings(unittest.TestCase):
    """Tests whether the verify_exp_config_types function catches invalid
    recreate_s3 settings.."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_exp_config = Supported_experiment_settings()
        self.valid_recreate_s3 = self.supp_exp_config.recreate_s3

        self.invalid_recreate_s3_value = {
            "recreate_s3": "invalid value of type string iso list"
            + " of floats",
        }

        self.supp_exp_config = supp_exp_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

    @typechecked
    def test_error_is_thrown_if_recreate_s3_key_is_missing(
        self,
    ) -> None:
        """Verifies an exception is thrown if the recreate_s3 key is missing
        from the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Remove key and value of m.

        exp_config.pop("recreate_s3")

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                supp_exp_config=self.supp_exp_config,
                exp_config=exp_config,
            )

        self.assertEqual(
            # "'recreate_s3'",
            "Error:recreate_s3 is not in the configuration"
            + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_recreate_s3_value_type(
        self,
    ) -> None:
        """Verifies an exception is thrown if the recreate_s3 dictionary value,
        is of invalid type.

        (Invalid types None, and string are tested, a list with floats
        is expected).
        """
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)
        # Set negative value of recreate_s3 in copy.

        # TODO: generalise to also check if an error is thrown if it contains a
        # string or integer, using the generic test file.
        # verify_invalid_config_sett_val_throws_error
        exp_config.recreate_s3 = None

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                supp_exp_config=self.supp_exp_config,
                exp_config=exp_config,
            )

        self.assertEqual(
            # "Error, expected type:<class 'bool'>, yet it was:"
            # + f"{type(None)} for:{None}",
            'type of argument "bool_setting" must be bool; got NoneType '
            "instead",
            str(context.exception),
        )
