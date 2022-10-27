"""Verifies The Supported_experiment_settings object catches invalid adaptation
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.experiment_settings.Supported_run_settings import (
    Supported_run_settings,
)
from src.experiment_settings.verify_run_settings import verify_run_config

with_adaptation_with_radiation = {
    "adaptation": {"redundancy": 1.0},
    "algorithm": {
        "MDSA": {
            "m_val": 2,
        }
    },
    "graph_size": 4,
    "graph_nr": 5,
    "iteration": 4,
    "overwrite_sim_results": True,
    "overwrite_visualisation": True,
    "radiation": {
        "delta_synaptic_w": (0.05, 0.4),
    },
    "seed": 5,
    "simulator": "lava",
}

# TODO: verify error is thrown on unsupported value.
# "adaptations": {"redundancy": 1.},
# TODO: verify error is thrown on unsupported value.
# "radiations": {"delta_synaptic_w":(0.05, 0.4),},
# TODO: verify transient is also a supported keyword.
# "radiations": {"delta_synaptic_w":(0.05, 0.4),"transient": 10},
# TODO: verify neuron_death is also supported.
# "radiations": {"neuron_death": 0.25},
# TODO: verify transient is also a supported keyword.
# "radiations": {"neuron_death": 0.25,"transient": 10},


class Test_generic_configuration_settings(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supp_run_settings = Supported_run_settings()
        self.valid_run_setting = with_adaptation_with_radiation
        self.invalid_adaptation_key = "non-existing-key"

    def test_returns_valid_configuration_settings(self):
        """Verifies a valid configuration settings object and object type is
        returned."""
        returned_dict = verify_run_config(
            self.supp_run_settings,
            self.valid_run_setting,
            has_unique_id=False,
            strict=True,
        )
        self.assertIsInstance(returned_dict, dict)

        self.assertEqual(self.valid_run_setting, returned_dict)

    def test_experiment_config_is_none(self):
        """Verifies an error is thrown if configuration settings object is of
        type None."""

        with self.assertRaises(Exception) as context:
            # Configuration Settings of type None throw error.
            verify_run_config(
                self.supp_run_settings, None, has_unique_id=False, strict=True
            )

        self.assertEqual(
            "Error, the run_config is of type:"
            + f"{type(None)}, yet it was expected to be of"
            + " type dict.",
            str(context.exception),
        )

    def test_catch_invalid_experiment_config_type(self):
        """Verifies an error is thrown if configuration settings object is of
        invalid type.

        (String instead of the expected dictionary).
        """

        with self.assertRaises(Exception) as context:
            # iterations dictionary of type None throws error.
            verify_run_config(
                self.supp_run_settings,
                "string_instead_of_dict",
                has_unique_id=False,
                strict=True,
            )
        self.assertEqual(
            "Error, the run_config is of type:"
            + f'{type("")}, yet it was expected to be of'
            + " type dict.",
            str(context.exception),
        )

    def test_error_is_thrown_on_invalid_configuration_setting_key(self):
        """Verifies an error is thrown on an invalid configuration setting
        key."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.valid_run_setting)

        # Add invalid key to configuration dictionary.
        experiment_config[self.invalid_adaptation_key] = "Filler"

        with self.assertRaises(Exception) as context:
            # iterations dictionary of type None throws error.
            verify_run_config(
                self.supp_run_settings,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )
        self.assertEqual(
            f"Error:{self.invalid_adaptation_key} is not supported by the"
            + " configuration settings:"
            + f"{self.supp_run_settings.parameters}",
            str(context.exception),
        )


def verify_error_is_thrown_on_invalid_configuration_setting_value(
    invalid_config_setting_value, experiment_config, expected_type, test_object
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
        verify_run_config(
            test_object.supp_experi_setts,
            experiment_config,
            has_unique_id=False,
            strict=True,
        )

    test_object.assertEqual(
        f"Error, expected type:{expected_type}, yet it was:"
        + f"{actual_type} for:{invalid_config_setting_value}",
        str(context.exception),
    )
