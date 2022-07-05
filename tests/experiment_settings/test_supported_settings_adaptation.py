"""Verifies The Supported_settings object catches invalid adaptation
specifications."""
import copy
import unittest

from src.experiment_settings.verify_supported_settings import (
    verify_adap_and_rad_settings,
    verify_configuration_settings,
)
from tests.experiment_settings.test_generic_configuration import (
    adap_sets,
    rad_sets,
    supp_sets,
    with_adaptation_with_radiation,
)


class Test_adaptation_settings(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_sets = supp_sets
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_iterations = self.supp_sets.iterations

        self.invalid_adaptation_value = {
            "redundancy": "invalid value of type string iso list",
        }

        self.invalid_adaptation_key = {"non-existing-key": 5}

    def test_error_is_thrown_if_adaptation_key_is_missing(self):
        """Verifies an exception is thrown if the adaptation key is missing
        from the MDSA algorithm settings dictionary of the supported algorithms
        dictionary of the configuration settings dictionary."""

        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(self.with_adaptation_with_radiation)

        # Remove key (and value) of adaptation from configuration settings.
        config_settings.pop("adaptation")

        with self.assertRaises(Exception) as context:
            verify_configuration_settings(
                self.supp_sets, config_settings, has_unique_id=False
            )

        self.assertEqual(
            # "'adaptation'",
            "Error:adaptation is not in the configuration"
            + f" settings:{config_settings.keys()}",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_adaptation_value_type_is_none(self):
        """Verifies if an error is thrown if the value belonging to the
        adaptation key in the configuration settings has value: None.

        (The value should be a dict.) # TODO use generic method.
        """

        with self.assertRaises(Exception) as context:
            # Adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(self.supp_sets, None, "adaptation")

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {type(None)}.",
            str(context.exception),
        )

    def test_error_is_thrown_for_invalid_adaptation_value_type_is_string(self):
        """Verifies if an error is thrown if the value belonging to the
        adaptation key in the configuration settings has a value of type
        string.

        (The value should be a dict.) # TODO use generic method.
        """
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                self.supp_sets,
                "string_instead_of_dict",
                "adaptation",
            )

        self.assertEqual(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {str}.",
            str(context.exception),
        )

    def test_error_is_thrown_if_adaptation_dictionary_keys_are_missing(self):
        """Verifies an exception is thrown if an empty adaptation dict is
        thrown."""
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(self.supp_sets, {}, "adaptation")

        self.assertEqual(
            "Error, property dict: adaptation was empty.",
            # "Error:adaptation is not in the configuration"
            # + f" settings:{config_settings.keys()}",
            str(context.exception),
        )

    def test_catch_invalid_adaptation_dict_key(self):
        """."""

        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                self.supp_sets, self.invalid_adaptation_key, "adaptation"
            )

        self.assertEqual(
            "Error, property.key:non-existing-key is not in the supported "
            + f"property keys:{self.supp_sets.adaptation.keys()}.",
            str(context.exception),
        )

    def test_catch_invalid_adaptation_dict_value_type_for_key(self):
        """Tests whether the adaptation setting dictionary throws an error if
        it contains an invalid value type for one of its keys.

        In this case, the neuron_death key of the adaptation dictionary
        is set to an invalid value type. It is set to string, whereas it
        should be a float or dict.
        """
        with self.assertRaises(Exception) as context:
            # adaptation dictionary of type None throws error.
            verify_adap_and_rad_settings(
                self.supp_sets, self.invalid_adaptation_value, "adaptation"
            )

        self.assertEqual(
            'Error, value of adaptation["redundancy"]='
            + f"invalid value of type string iso list, (which has type:{str})"
            + ", is of different type than the expected and supported type: "
            + f"{list}",
            str(context.exception),
        )
