"""Verifies The Supported_experiment_settings object catches invalid adaptation
specifications."""
# pylint: disable=R0801
import copy
import unittest
from typing import Any, Dict

from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.exp_config.run_config.Supported_run_settings import (
    Supported_run_settings,
)
from snncompare.exp_config.run_config.verify_run_settings import (
    verify_run_config,
)

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
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.supp_run_settings = Supported_run_settings()
        self.valid_run_setting = with_adaptation_with_radiation
        self.invalid_adaptation_key = "non-existing-key"

    @typechecked
    def test_returns_valid_configuration_settings(self) -> None:
        """Verifies a valid configuration settings object and object type is
        returned."""
        returned_dict = verify_run_config(
            self.supp_run_settings,
            self.valid_run_setting,
            has_unique_id=False,
            allow_optional=False,
        )
        self.assertIsInstance(returned_dict, Dict)

        self.assertEqual(self.valid_run_setting, returned_dict)

    @typechecked
    def test_error_is_thrown_on_invalid_configuration_setting_key(
        self,
    ) -> None:
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
                allow_optional=False,
            )
        self.assertEqual(
            f"Error:{self.invalid_adaptation_key} is not supported by the"
            + " configuration settings:"
            + f"{self.supp_run_settings.parameters}",
            str(context.exception),
        )


@typechecked
def verify_invalid_config_sett_val_throws_error(  # type:ignore[misc]
    invalid_config_setting_value: Any,
    experiment_config: Exp_config,
    expected_type: type,
    test_object: Any,
) -> None:
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
            test_object.supp_exp_config,
            experiment_config,
            has_unique_id=False,
            allow_optional=False,
        )

    test_object.assertEqual(
        f"Error, expected type:{expected_type}, yet it was:"
        + f"{actual_type} for:{invalid_config_setting_value}",
        str(context.exception),
    )
