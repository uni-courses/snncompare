"""Verifies The Supported_experiment_settings object catches invalid radiations
specifications."""
# pylint: disable=R0801
import copy
import unittest

from typeguard import typechecked

from snncompare.exp_config.Exp_config import (
    verify_adap_and_rad_settings,
    verify_exp_config,
)
from tests.exp_config.exp_config.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_config,
    with_adaptation_with_radiation,
)


class Test_radiations_settings(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_exp_config = supp_exp_config
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation

        self.valid_radiations = self.supp_exp_config.radiations

        self.invalid_radiations_value = {
            "neuron_death": "invalid value of type string iso list",
        }

        self.invalid_radiations_key = {"non-existing-key": 5}

    @typechecked
    def test_error_is_thrown_if_radiations_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the radiations key is missing
        from the MDSA algorithm settings dictionary of the supported algorithms
        dictionary of the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)

        # Remove key and value of m.
        exp_config.pop("radiations")

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                supp_exp_config=self.supp_exp_config,
                exp_config=exp_config,
            )

        self.assertEqual(
            # "'radiations'",
            "Error:radiations is not in the configuration"
            + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_radiations_value_type_is_none(
        self,
    ) -> None:
        """Verifies if an error is thrown if the value belonging to the
        radiations key in the configuration settings has value: None.

        (The value should be a dict.)
        """

        with self.assertRaises(Exception) as context:
            # radiations dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict=None,
                check_type="radiations",
            )

        self.assertEqual(
            "Error, property is expected to be a Dict, yet"
            + f" it was of type: {type(None)}.",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_radiations_value_type_is_string(
        self,
    ) -> None:
        """Verifies if an error is thrown if the value belonging to the
        radiations key in the configuration settings has a value of type
        string.

        (The value should be a dict.) # TODO: use generic method for
        this test.
        """
        with self.assertRaises(Exception) as context:
            # radiations dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict="string_instead_of_dict",
                check_type="radiations",
            )

        self.assertEqual(
            "Error, property is expected to be a Dict, yet"
            + f" it was of type: {str}.",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_if_radiations_dictionary_keys_are_missing(
        self,
    ) -> None:
        """Verifies an exception is thrown if an empty radiations dict is
        thrown."""
        with self.assertRaises(Exception) as context:
            # radiations dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict={},
                check_type="radiations",
            )

        self.assertEqual(
            "Error, property Dict: radiations was empty.",
            # "Error:radiations is not in the configuration"
            # + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_catch_invalid_radiations_dict_key(self) -> None:
        """."""

        with self.assertRaises(Exception) as context:
            # radiations dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict=self.invalid_radiations_key,
                check_type="radiations",
            )

        self.assertEqual(
            "Error, property.key:non-existing-key is not in the supported "
            + f"property keys:{self.supp_exp_config.radiations.keys()}.",
            str(context.exception),
        )

    @typechecked
    def test_catch_invalid_radiations_dict_value_type_for_key(self) -> None:
        """Tests whether the radiations setting dictionary throws an error if
        it contains an invalid value type for one of its keys.

        In this case, the neuron_death key of the radiations dictionary
        is set to an invalid value type. It is set to string, whereas it
        should be a float or dict.
        """
        with self.assertRaises(Exception) as context:
            # radiations dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict=self.invalid_radiations_value,
                check_type="radiations",
            )

        self.assertEqual(
            "Error, the radiations value is of type:"
            + f"{str}, yet it was expected to be"
            + " float or dict.",
            str(context.exception),
        )
