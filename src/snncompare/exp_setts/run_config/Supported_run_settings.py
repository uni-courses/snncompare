"""Contains the supported experiment settings.

(The values of the settings may vary, yet the values of an experiment
setting should be within the ranges specified in this file, and the
setting types should be identical.)
"""
import copy
import hashlib
import json
from typing import Any, List, Union

from typeguard import typechecked

from .verify_run_settings import verify_run_config

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
        # experiment_config dictionary keys:
        self.parameters: dict[str, Any] = {
            "adaptation": Union[None, dict],
            "algorithm": dict,
            "iteration": int,
            "graph_size": int,
            "graph_nr": int,
            "radiation": Union[None, dict],
            "seed": int,
            "simulator": str,
        }
        self.optional_parameters = {
            "export_images": bool,
            "max_duration": int,
            "overwrite_sim_results": bool,
            "overwrite_visualisation": bool,
            "show_snns": bool,
            "unique_id": str,
        }

    @typechecked
    def append_unique_run_config_id(
        self, run_config: dict, allow_optional: bool
    ) -> dict:
        """Checks if an run configuration dictionary already has a unique
        identifier, and if not it computes and appends it.

        If it does, throws an error.

        :param run_config: dict:
        """
        if "unique_id" in run_config.keys():
            raise Exception(
                f"Error, the run_config:{run_config}\n"
                + "already contains a unique identifier."
            )

        verify_run_config(
            self,
            run_config,
            has_unique_id=False,
            allow_optional=allow_optional,
        )

        minimal_run_config = self.remove_optional_args(
            copy.deepcopy(run_config)
        )
        unique_id = str(
            hashlib.sha256(
                json.dumps(minimal_run_config).encode("utf-8")
            ).hexdigest()
        )
        run_config["unique_id"] = unique_id
        verify_run_config(
            self,
            run_config,
            has_unique_id=True,
            allow_optional=allow_optional,
        )
        return run_config

    def remove_optional_args(self, copied_run_config: dict) -> List[Any]:
        """removes the optional arguments from a run config."""
        optional_keys = []
        for key in copied_run_config.keys():
            if key in self.optional_parameters:
                optional_keys.append(key)
        for key in optional_keys:
            copied_run_config.pop(key)
        verify_run_config(
            self,
            copied_run_config,
            has_unique_id=False,
            allow_optional=False,
        )
        return sorted(copied_run_config)

    @typechecked
    def assert_has_key(
        self, some_dict: dict, key: str, some_type: type
    ) -> None:
        """Asserts a dictionary has some key with a value of a certain type.

        Throws error if the key does not exist, or if the value is of an
        invalid type.
        """
        if not isinstance(some_dict[key], some_type):
            raise Exception(
                "Error, the dictionary:{some_dict} did not"
                + f"contain a key:{key} of type:{type}"
            )
