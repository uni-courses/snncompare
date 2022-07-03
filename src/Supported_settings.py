"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
from typing import Any, Dict


class Supported_settings:
    """Stores the supported radiation and adaptation settings."""

    def __init__(
        self,
    ) -> None:
        # Specify the (to be) supported adaptation types.
        self.adaptation = {
            "none": [],
            "redundancy": [
                1.0,
            ],  # Create 1 redundant neuron per neuron.
            "population": [
                10.0
            ],  # Create a population of 10 neurons to represent a
            # single neuron.
            "rate_coding": [
                5.0
            ],  # Multiply firing frequency with 5 to limit spike decay
            # impact.
        }

        # List of tuples with x=probabiltity of change, y=average value change
        # in synaptic weights.
        self.delta_synaptic_w = [
            (0.01, 0.5),
            (0.05, 0.4),
            (0.1, 0.3),
            (0.2, 0.2),
            (0.25, 0.1),
        ]

        # List of tuples with x=probabiltity of change, y=average value change
        # in neuronal threshold.
        self.delta_vth = [
            (0.01, 0.5),
            (0.05, 0.4),
            (0.1, 0.3),
            (0.2, 0.2),
            (0.25, 0.1),
        ]

        self.radiation = {
            "none": [],
            "neuron_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            "synaptic_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            "delta_synaptic_w": self.delta_synaptic_w,
            "delta_vth": self.delta_vth,
        }

    def verify_config_setting(self, some_dict, check_type) -> dict:
        """Verifies the settings of adaptation or radiation property are valid.

        :param some_dict: param check_type:
        :param check_type:
        """
        if check_type == "adaptation":
            reference_object: Dict[str, Any] = self.adaptation
        elif check_type == "radiation":
            reference_object = self.radiation
        else:
            raise Exception(f"Check type:{check_type} not supported.")

        # Verify object is a dictionary.
        if isinstance(some_dict, dict):
            if some_dict == {}:
                raise Exception(
                    f"Error, property dict: {check_type} was empty."
                )
            for key in some_dict:

                # Verify the keys are within permissible keys.
                if key not in reference_object:
                    raise Exception(
                        f"Error, property.key:{key} is not in the supported "
                        + f"property keys:{reference_object.keys()}."
                    )
                # Check values belonging to key
                if check_type == "adaptation":
                    self.verify_adaptation_values(some_dict, key)
                elif check_type == "radiation":
                    self.verify_radiation_values(some_dict, key)
            return some_dict
        raise Exception(
            "Error, property is expected to be a dict, yet"
            + f" it was of type: {type(some_dict)}."
        )

    def verify_adaptation_values(self, adaptation: dict, key: str) -> None:
        """

        :param adaptation: dict:
        :param key: str:
        :param adaptation: dict:
        :param key: str:

        """

        if not isinstance(adaptation[key], type(self.adaptation[key])) or (
            not isinstance(adaptation[key], float)
            and not isinstance(adaptation[key], list)
        ):
            raise Exception(
                f'Error, value of adaptation["{key}"]='
                + f"{adaptation[key]}, (which has type:{type(adaptation[key])}"
                + "), is of different type than the expected and supported "
                + f"type: {type(self.adaptation[key])}"
            )
        # TODO: verify the elements in the list are of type float, if the value
        # is a list.

    def verify_radiation_values(self, radiation: dict, key: str) -> None:
        """

        :param radiation: dict:
        :param key: str:
        :param radiation: dict:
        :param key: str:

        """
        if not isinstance(radiation[key], type(self.radiation[key])) or (
            not isinstance(radiation[key], list)
        ):

            raise Exception(
                "Error, the radiation value is of type:"
                + f"{type(radiation[key])}, yet it was expected to be"
                + " float or dict."
            )
        # TODO: verify the elements in the list are of type float, or tuples of
        # (float,float), if the value is a list.

    def append_unique_config_id(self, experiment_config: dict) -> dict:
        """Checks if an experiment configuration dictionary already has a
        unique identifier, and if not it computes and appends it.

        If it does, throws an error.

        :param experiment_config: dict:
        :param experiment_config: dict:
        :param experiment_config: dict:
        :param experiment_config: dict:
        """
        if "unique_id" in experiment_config.keys():
            raise Exception(
                f"Error, the experiment_config:{experiment_config}\n"
                + "already contains a unique identifier."
            )

        self.verify_configuration_settings(
            experiment_config, has_unique_id=False
        )
        hash_set = frozenset(experiment_config.values())
        unique_id = hash(hash_set)
        experiment_config["unique_id"] = unique_id
        self.verify_configuration_settings(
            experiment_config, has_unique_id=True
        )
        return experiment_config

    # pylint: disable=W0613
    def verify_configuration_settings(self, experiment_config, has_unique_id):
        """TODO: Verifies the experiment configuration settings are valid.

        :param experiment_config: param has_unique_id:
        :param has_unique_id:
        """

    # pylint: disable=W0613
    def verify_configuration_settings_types(
        self, experiment_config, has_unique_id
    ):
        """TODO: Verifies the experiment configuration settings are of the
        correct type.

        :param experiment_config: param has_unique_id:
        :param has_unique_id:
        """
