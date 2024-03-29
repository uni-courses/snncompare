"""Verifies The Supported_experiment_settings object catches invalid adaptation
specifications."""
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


class Test_adaptation_settings(unittest.TestCase):
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

        self.invalid_adaptation_value = {
            "redundancy": "invalid value of type string iso list",
        }

        self.invalid_adaptation_key = {"non-existing-key": 5}

    @typechecked
    def test_error_is_thrown_if_adaptation_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the adaptation key is missing
        from the MDSA algorithm settings dictionary of the supported algorithms
        dictionary of the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(self.with_adaptation_with_radiation)

        # Remove key (and value) of adaptation from configuration settings.
        exp_config.pop("adaptations")

        with self.assertRaises(Exception) as context:
            verify_exp_config(
                supp_exp_config=self.supp_exp_config,
                exp_config=exp_config,
            )

        self.assertEqual(
            # "'adaptation'",
            "Error:adaptations is not in the configuration"
            + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_adaptation_value_type_is_none(
        self,
    ) -> None:
        """Verifies if an error is thrown if the value belonging to the
        adaptation key in the configuration settings has value: None.

        (The value should be a dict.) # TODO use generic method.
        """

        with self.assertRaises(Exception) as context:
            # Adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict=None,
                check_type="adaptations",
            )

        self.assertEqual(
            "Error, property is expected to be a Dict, yet"
            + f" it was of type: {type(None)}.",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_for_invalid_adaptation_value_type_is_string(
        self,
    ) -> None:
        """Verifies if an error is thrown if the value belonging to the
        adaptation key in the configuration settings has a value of type
        string.

        (The value should be a dict.) # TODO use generic method.
        """
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict="string_instead_of_dict",
                check_type="adaptations",
            )

        self.assertEqual(
            "Error, property is expected to be a Dict, yet"
            + f" it was of type: {str}.",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_if_adaptation_dictionary_keys_are_missing(
        self,
    ) -> None:
        """Verifies an exception is thrown if an empty adaptation dict is
        thrown."""
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict={},
                check_type="adaptations",
            )

        self.assertEqual(
            "Error, property Dict: adaptations was empty.",
            # "Error:adaptation is not in the configuration"
            # + f" settings:{exp_config.keys()}",
            str(context.exception),
        )

    @typechecked
    def test_catch_invalid_adaptation_dict_key(self) -> None:
        """."""

        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict=self.invalid_adaptation_key,
                check_type="adaptations",
            )

        self.assertEqual(
            "Error, property.key:non-existing-key is not in the supported "
            + f"property keys:{self.supp_exp_config.adaptations.keys()}.",
            str(context.exception),
        )

    @typechecked
    def test_catch_invalid_adaptation_dict_value_type_for_key(self) -> None:
        """Tests whether the adaptation setting dictionary throws an error if
        it contains an invalid value type for one of its keys.

        In this case, the neuron_death key of the adaptation dictionary
        is set to an invalid value type. It is set to string, whereas it
        should be a float or dict.
        """
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                supp_exp_config=self.supp_exp_config,
                some_dict=self.invalid_adaptation_value,
                check_type="adaptations",
            )

        self.assertEqual(
            'Error, value of adaptations["redundancy"]='
            + f"invalid value of type string iso list, (which has type:{str})"
            + ", is of different type than the expected and supported type: "
            + f"{list}",
            str(context.exception),
        )
