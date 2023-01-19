"""Verifies The Supported_experiment_settings object catches invalid adaptation
specifications."""
import copy
import unittest
from typing import Any, Dict, Optional

from snnadaptation.redundancy.Adaptation_Rad_settings import (
    Adaptations_settings,
    Radiation_settings,
)
from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.exp_config.default_setts.create_default_settings import (
    default_exp_config,
)
from snncompare.exp_config.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_config.verify_experiment_settings import verify_exp_config

supp_exp_config = Supported_experiment_settings()
adap_sets = Adaptations_settings()
rad_sets = Radiation_settings()
with_adaptation_with_radiation = default_exp_config()


class Test_generic_configuration_settings(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
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

    @typechecked
    def test_returns_valid_configuration_settings(self) -> None:
        """Verifies a valid configuration settings object and object type is
        returned."""
        verify_exp_config(
            supp_exp_config,
            with_adaptation_with_radiation,
            has_unique_id=False,
            allow_optional=False,
        )
        self.assertIsInstance(with_adaptation_with_radiation, Dict)

        self.assertEqual(
            with_adaptation_with_radiation, with_adaptation_with_radiation
        )

    @typechecked
    def test_exp_config_is_none(self) -> None:
        """Verifies an error is thrown if configuration settings object is of
        type None."""

        with self.assertRaises(Exception) as context:
            # Configuration Settings of type None throw error.
            verify_exp_config(
                supp_exp_config,
                None,
                has_unique_id=False,
                allow_optional=False,
            )

        self.assertEqual(
            "Error, the exp_config is of type:"
            + f"{type(None)}, yet it was expected to be of"
            + " type dict.",
            str(context.exception),
        )

    @typechecked
    def test_catch_invalid_exp_config_type(self) -> None:
        """Verifies an error is thrown if configuration settings object is of
        invalid type.

        (String instead of the expected dictionary).
        """

        with self.assertRaises(Exception) as context:
            # iterations dictionary of type None throws error.
            verify_exp_config(
                supp_exp_config,
                "string_instead_of_dict",
                has_unique_id=False,
                allow_optional=False,
            )
        self.assertEqual(
            "Error, the exp_config is of type:"
            + f'{type("")}, yet it was expected to be of'
            + " type dict.",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_on_invalid_configuration_setting_key(
        self,
    ) -> None:
        """Verifies an error is thrown on an invalid configuration setting
        key."""
        # Create deepcopy of configuration settings.
        exp_config = copy.deepcopy(with_adaptation_with_radiation)

        # Add invalid key to configuration dictionary.
        exp_config[self.invalid_adaptation_key] = "Filler"

        with self.assertRaises(Exception) as context:
            # iterations dictionary of type None throws error.
            verify_exp_config(
                supp_exp_config,
                exp_config,
                has_unique_id=False,
                allow_optional=False,
            )
        self.assertEqual(
            f"Error:{self.invalid_adaptation_key} is not supported by the"
            + " configuration settings:"
            + f"{supp_exp_config.parameters}",
            str(context.exception),
        )


# pylint: disable=R0913
@typechecked
def verify_invalid_config_sett_val_throws_error(  # type:ignore[misc]
    invalid_config_setting_value: Optional[str],
    exp_config: Exp_config,
    expected_type: type,
    test_object: Any,
    non_typechecked_error: Optional[bool] = False,
    alternative_var_name: Optional[str] = None,
) -> None:
    """Verifies an error is thrown on an invalid configuration setting value.

    This method is called by other test files and is genereric for most
    configuration setting parameters.

    TODO: simplify this method.
    Possibly separate it into submethods, or generalize it better
    using input argument improvement.
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
        verify_exp_config(
            test_object.supp_exp_config,
            exp_config,
            has_unique_id=False,
            allow_optional=False,
        )

    # Create expected output message.
    if isinstance(invalid_config_setting_value, type(None)):
        str_of_actual_type = "NoneType"
    if isinstance(invalid_config_setting_value, str):
        str_of_actual_type = "str"
    if isinstance(invalid_config_setting_value, bool):
        str_of_actual_type = "bool"

    if alternative_var_name is None:
        var_name = "integer_setting"
        str_of_expected_type = "int"
    else:
        var_name = alternative_var_name
        str_of_expected_type = "bool"

    if non_typechecked_error:
        # Some functions take the Supported_..() type as argument, which means
        # they, currently, cannot be typechecked with the @decorator. That
        # means the manual type-checking error is raised inside the function.
        test_object.assertEqual(
            f"Error, expected type:{expected_type}, yet it was:"
            + f"{type(invalid_config_setting_value)} for:"
            + f"{invalid_config_setting_value}",
            str(context.exception),
        )
    else:
        # This is the default typechecked error that is raised if an
        # invalid type is passed into the function.
        test_object.assertEqual(
            f'type of argument "{var_name}" must be {str_of_expected_type}; '
            + f"got {str_of_actual_type} instead",
            str(context.exception),
        )
