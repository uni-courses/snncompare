"""Contains the supported experiment settings.

(The values of the settings may vary, yet the values of an experiment
setting should be within the ranges specified in this file, and the
setting types should be identical.)

TODO: determine whether to separate typing object or not, like in
Supported_experiment_settings.py.
"""
from typing import Dict, Union

from typeguard import typechecked


# pylint: disable=R0902
# The settings object contains all the settings as a dictionary, hence no
# hierarchy is used,leading to 10/7 instance attributes.
# pylint: disable=R0801
# pylint: disable=R0903
class Supported_run_settings:
    """Stores the supported experiment setting parameter ranges.

    An experiment can consist of multiple runs. A run is a particular
    combination of experiment setting parameters.
    """

    @typechecked
    def __init__(
        self,
    ) -> None:
        # exp_config dictionary keys:
        self.parameters: Dict = {
            "adaptation": Union[None, Dict],
            "algorithm": Dict,
            "graph_size": int,
            "graph_nr": int,
            "radiation": Union[None, Dict],
            "seed": int,
            "simulator": str,
        }

    @typechecked
    def assert_has_key(
        self, some_dict: Dict, key: str, some_type: type
    ) -> None:
        """Asserts a dictionary has some key with a value of a certain type.

        Throws error if the key does not exist, or if the value is of an
        invalid type.
        """
        if not isinstance(some_dict[key], some_type):
            raise TypeError(
                f"Error, the dictionary:{some_dict} did not"
                + f"contain a key:{key} of type:{type}"
            )
