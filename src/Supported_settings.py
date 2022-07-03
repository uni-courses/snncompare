"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
from typing import Any, Dict


# pylint: disable=R0902
# The settings object contains all the settings as a dictionary, hence no
# hierarchy is used, leading to 10/7 instance attributes.
class Supported_settings:
    """Stores examples of the supported experiment settings, such as radiation
    and adaptation settings.

    Also verifies the settings that are created.
    """

    def __init__(
        self,
    ) -> None:
        # The number of iterations for which the Alipour approximation is ran.
        self.m = list(range(0, 1, 1))
        # The number of times the experiment is repeated.
        self.iterations = list(range(0, 3, 1))

        # Specify the maximum number of: (maximum number of graphs per run
        # size).
        self.max_max_graphs = 15
        # The size of the graph and the maximum number of used graphs of that
        # size.
        self.size_and_max_graphs = [
            (3, self.max_max_graphs),
            (4, self.max_max_graphs),
        ]
        # Overwrite the simulation results or not.
        self.overwrite_sim_results = True
        # Overwrite the visualisation of the SNN behaviour or not.
        self.overwrite_visualisation = True
        # The backend/type of simulator that is used.
        self.simulators = ["nx"]

        # Generate the supported adaptation settings.
        self.specify_supported_adaptation_settings()
        # Generate the supported radiation settings.
        self.specify_supported_radiation_settings()

    def specify_supported_adaptation_settings(self):
        """Specifies all the supported types of adaptation settings."""

        # Specify the (to be) supported adaptation types.
        self.adaptation = {
            "None": [],
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

    def specify_supported_radiation_settings(self):
        """Specifies types of supported radiation settings. Some settings
        consist of a list of tuples, with the probability of a change
        occurring, followed by the average magnitude of the change.

        Others only contain a list of floats which represent the
        probability of radiation induced change occurring.
        """
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

        # Create a supported radiation setting example.
        self.radiation = {
            # No radiation
            "None": [],
            # Radiation effects are transient, they last for 1 or 10 simulation
            # steps. If transient is 0., the changes are permanent.
            "transient": [0.0, 1.0, 10.0],
            # List of probabilities of a neuron dying due to radiation.
            "neuron_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            # List of probabilities of a synapse dying due to radiation.
            "synaptic_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            # List of: (probability of synaptic weight change, and the average
            # factor with which it changes due to radiation).
            "delta_synaptic_w": self.delta_synaptic_w,
            # List of: (probability of neuron threshold change, and the average
            # factor with which it changes due to radiation).
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
        :param adaptation: dict:
        :param key: str:
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
        if isinstance(adaptation[key], list):
            for setting in adaptation[key]:
                self.verify_object_type(setting, float, None)

    def verify_object_type(self, obj, expected_type, tuple_types=None):
        """

        :param obj: param expected_type:
        :param tuple_types: Default value = None)
        :param expected_type:

        """

        # Verify the object type is as expected.
        if not isinstance(obj, expected_type):
            raise Exception(
                f"Error, expected type:{expected_type}, yet it was:{type(obj)}"
                + f" for:{obj}"
            )

        # If object is of type float, verify the tuple element types.
        if isinstance(obj, tuple):

            # Verify user passed the expected tuple element types.
            if tuple_types is None:
                raise Exception(
                    "Expected two types in a list to check tuple contents."
                )

            # Verify the tuple element types.
            if not (
                isinstance(obj, tuple) and list(map(type, obj)) == tuple_types
            ):
                raise Exception(
                    f"Error, obj={obj}, its type is:{list(map(type, obj))},"
                    + f" expected type:{tuple_types}"
                )

    def verify_radiation_values(self, radiation: dict, key: str) -> None:
        """

        :param radiation: dict:
        :param key: str:
        :param radiation: dict:
        :param key: str:
        :param radiation: dict:
        :param key: str:
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

        # Verify radiation setting types.
        if isinstance(radiation[key], list):
            for setting in radiation[key]:

                # Verify radiation setting can be of type float.
                if isinstance(setting, float):
                    # TODO: superfluous check.
                    self.verify_object_type(setting, float, None)
                # Verify radiation setting can be of type tuple.
                elif isinstance(setting, tuple):
                    # Verify the radiation setting tuple is of type float,
                    # float.
                    self.verify_object_type(setting, tuple, [float, float])
                else:
                    # Throw error if the radiation setting is something other
                    # than a float or tuple of floats.
                    raise Exception(
                        f"Unexpected setting type:{type(setting)} for:"
                        + f" {setting}."
                    )

    def append_unique_config_id(self, experiment_config: dict) -> dict:
        """Checks if an experiment configuration dictionary already has a
        unique identifier, and if not it computes and appends it.

        If it does, throws an error.

        :param experiment_config: dict:
        :param experiment_config: dict:
        :param experiment_config: dict:
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
        self.verify_m_setting(experiment_config["m"])
        if has_unique_id:
            print("TODO: test unique id type.")

    def verify_m_setting(self, m_setting):
        """Verifies the type of m setting is valid, and that its values are
        within the supported range.

        :param m_setting:
        """
        if not isinstance(m_setting, list):
            # TODO: verify subtypes.
            raise Exception(
                "Error, m was expected to be a list of integers."
                + f" Instead, it was:{type(m_setting)}"
            )
        if len(m_setting) < 1:
            raise Exception(
                "Error, m was expected contain at least 1 integer."
                + f" Instead, it has length:{len(m_setting)}"
            )
        for m in m_setting:
            if m not in self.m:
                raise Exception(
                    "Error, m was expected to be in range:{self.m}."
                    + f" Instead, it contains:{len(m)}"
                )
