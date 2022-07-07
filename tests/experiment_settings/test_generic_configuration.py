"""Verifies The Supported_settings object catches invalid adaptation
specifications."""
import copy
import unittest

from src.experiment_settings.Adaptation_Rad_settings import (
    Adaptation_settings,
    Radiation_settings,
)
from src.experiment_settings.Supported_settings import Supported_settings
from src.experiment_settings.verify_supported_settings import (
    verify_adap_and_rad_settings,
    verify_configuration_settings,
)

supp_sets = Supported_settings()
adap_sets = Adaptation_settings()
rad_sets = Radiation_settings()
with_adaptation_with_radiation = {
    "algorithms": {
        "MDSA": {
            "m_vals": list(range(0, 1, 1)),
        }
    },
    "iterations": list(range(0, 3, 1)),
    "min_max_graphs": 1,
    "max_max_graphs": 15,
    "min_graph_size": 3,
    "max_graph_size": 20,
    "size_and_max_graphs": [(3, 15), (4, 15)],
    "adaptation": verify_adap_and_rad_settings(
        supp_sets, adap_sets.with_adaptation, "adaptation"
    ),
    "radiation": verify_adap_and_rad_settings(
        supp_sets, rad_sets.with_radiation, "radiation"
    ),
    "overwrite_sim_results": True,
    "overwrite_visualisation": True,
    "simulators": ["nx"],
}


class Test_generic_configuration_settings(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.valid_adaptation = {
            "redundancy": [1.0, 2.0],  # Create 1 and 2 redundant neuron(s) per
            # neuron.
            "population": [
                10.0
            ],  # Create a population of 10 neurons to represent a
            # single neuron.
            "rate_coding": [
                5.0
            ],  # Multiply firing frequency with 5 to limit spike decay
            # impact.
        }
        self.invalid_adaptation_key = "non-existing-key"

        self.invalid_adaptation_value = {
            "redundancy": "invalid value of type string iso list",
        }

    def test_returns_valid_configuration_settings(self):
        """Verifies a valid configuration settings object and object type is
        returned."""
        returned_dict = verify_configuration_settings(
            supp_sets,
            with_adaptation_with_radiation,
            has_unique_id=False,
        )
        self.assertIsInstance(returned_dict, dict)

        self.assertEqual(with_adaptation_with_radiation, returned_dict)

    def test_config_settings_is_none(self):
        """Verifies an error is thrown if configuration settings object is of
        type None."""

        with self.assertRaises(Exception) as context:
            # Configuration Settings of type None throw error.
            verify_configuration_settings(supp_sets, None, has_unique_id=False)

        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f"{type(None)}, yet it was expected to be of"
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_config_settings_type(self):
        """Verifies an error is thrown if configuration settings object is of
        invalid type.

        (String instead of the expected dictionary).
        """

        with self.assertRaises(Exception) as context:
            # iterations dictionary of type None throws error.
            verify_configuration_settings(
                supp_sets, "string_instead_of_dict", has_unique_id=False
            )
        self.assertEqual(
            "Error, the experiment_config is of type:"
            + f'{type("")}, yet it was expected to be of'
            + " type dict.",
            str(context.exception),
        )

    def test_error_is_thrown_on_invalid_configuration_setting_key(self):
        """Verifies an error is thrown on an invalid configuration setting
        key."""
        # Create deepcopy of configuration settings.
        config_settings = copy.deepcopy(with_adaptation_with_radiation)

        # Add invalid key to configuration dictionary.
        config_settings[self.invalid_adaptation_key] = "Filler"

        with self.assertRaises(Exception) as context:
            # iterations dictionary of type None throws error.
            verify_configuration_settings(
                supp_sets, config_settings, has_unique_id=False
            )
        self.assertEqual(
            f"Error:{self.invalid_adaptation_key} is not supported by the"
            + " configuration settings:"
            + f"{supp_sets.config_setting_parameters}",
            str(context.exception),
        )


def verify_error_is_thrown_on_invalid_configuration_setting_value(
    invalid_config_setting_value, config_settings, expected_type, test_object
):
    """Verifies an error is thrown on an invalid configuration setting value.

    This method is called by other test files and is genereric for most
    configuration setting parameters.
    """
    actual_type = type(invalid_config_setting_value)
    if not isinstance(actual_type, type) and not isinstance(
        expected_type, type
    ):
        raise Exception(
            "Error this method requires two types. You gave:"
            + f"{actual_type},{expected_type}"
        )
    with test_object.assertRaises(Exception) as context:
        verify_configuration_settings(
            test_object.supp_sets, config_settings, has_unique_id=False
        )

    test_object.assertEqual(
        f"Error, expected type:{expected_type}, yet it was:"
        + f"{actual_type} for:{invalid_config_setting_value}",
        str(context.exception),
    )
