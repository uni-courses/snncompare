"""Contains the supported experiment settings.

(The values of the settings may vary, yet the values of an experiment
setting should be within the ranges specified in this file, and the
setting types should be identical.)
"""


# pylint: disable=R0902
# The settings object contains all the settings as a dictionary, hence no
# hierarchy is used,leading to 10/7 instance attributes.
# pylint: disable=R0801
# pylint: disable=R0903


from src.snncompare.exp_setts.run_config.verify_run_settings import (
    verify_run_config,
)
from src.snncompare.exp_setts.Supported_experiment_settings import (
    dict_to_frozen_set,
)


class Supported_run_settings:
    """Stores the supported experiment setting parameter ranges.

    An experiment can consist of multiple runs. A run is a particular
    combination of experiment setting parameters.
    """

    def __init__(
        self,
    ) -> None:
        # experiment_config dictionary keys:
        self.parameters = {
            "adaptation": dict,
            "algorithm": dict,
            "iteration": int,
            "graph_size": int,
            "graph_nr": int,
            "radiation": dict,
            "overwrite_sim_results": bool,
            "overwrite_visualisation": bool,
            "seed": int,
            "simulator": str,
        }
        self.optional_parameters = {
            "duration": int,
            "export_images": bool,
            "show_snns": bool,
            "stage": int,  # TODO: remove this parameter.
            "unique_id": int,
        }

    def append_unique_config_id(self, run_config: dict) -> dict:
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

        verify_run_config(self, run_config, has_unique_id=False, strict=True)

        # hash_set = frozenset(run_config.values())
        hash_set = dict_to_frozen_set(run_config)
        unique_id = hash(hash_set)
        run_config["unique_id"] = unique_id
        verify_run_config(self, run_config, has_unique_id=False, strict=False)
        return run_config

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
