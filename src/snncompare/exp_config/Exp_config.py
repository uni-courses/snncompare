""""Stores the run config Dict type."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict

from snnadaptation.redundancy.verify_redundancy_settings import (
    verify_redundancy_settings_for_exp_config,
)
from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from snnalgorithms.verify_algos import verify_algos_in_exp_config
from snnradiation.Rad_damage import Rad_damage
from typeguard import typechecked

from snncompare.exp_config.rad_dict2obj import (
    get_radiations_from_exp_config_dict,
)


# pylint: disable=R0902
# pylint: disable=R0903
class Exp_config:
    """Stores the exp_configriment settings object."""

    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        adaptations: None | dict[str, int] | dict,
        algorithms: dict[str, list[dict[str, int]]],
        max_graph_size: int,
        max_max_graphs: int,
        min_graph_size: int,
        min_max_graphs: int,
        neuron_models: list,
        radiations: dict,
        seeds: list[int],
        simulators: list,
        size_and_max_graphs: list,
        synaptic_models: list,
    ):
        """Stores run configuration settings for the exp_configriment."""

        # Required properties
        self.adaptations: None | dict[str, int] = adaptations
        self.algorithms: dict[str, list[dict[str, int]]] = algorithms
        self.max_graph_size: int = max_graph_size
        self.max_max_graphs: int = max_max_graphs
        self.min_graph_size: int = min_graph_size
        self.min_max_graphs: int = min_max_graphs
        self.neuron_models: list = neuron_models
        self.radiations: list[
            Rad_damage
        ] = get_radiations_from_exp_config_dict(radiations=radiations)
        self.seeds: list[int] = seeds
        self.simulators: list = simulators
        self.size_and_max_graphs: list = size_and_max_graphs
        self.synaptic_models: list = synaptic_models

        self.unique_id: str = self.get_unique_exp_config_id()

        # Verify run config object.
        supp_exp_config = Supported_experiment_settings()
        verify_exp_config(
            supp_exp_config=supp_exp_config,
            exp_config=self,
        )

    @typechecked
    def get_unique_exp_config_id(
        self,
    ) -> str:
        """Returns a unique hash for the exp_config object."""
        some_exp_config: Exp_config = copy.deepcopy(self)
        rad_hashes: list[str] = []
        for rad_obj in self.radiations:
            rad_hashes.append(rad_obj.get_rad_settings_hash())
        del some_exp_config.radiations
        some_exp_config.radiations = sorted(rad_hashes)
        unique_id = str(
            hashlib.sha256(
                json.dumps(some_exp_config.__dict__).encode("utf-8")
            ).hexdigest()
        )
        return unique_id


# pylint: disable=R0902
class Supported_experiment_settings:
    """Contains the settings that are supported for the exp_config.

    (The values of the settings may vary, yet the values of an
    experiment setting should be within the ranges specified in this
    file, and the setting types should be identical.)
    """

    @typechecked
    def __init__(
        self,
    ) -> None:
        self.seed = 5

        # Create dictionary with algorithm name as key, and algorithm settings
        # object as value.
        mdsa_min = MDSA([]).min_m_vals
        mdsa_max = MDSA([]).max_m_vals
        self.algorithms = get_algo_configs(
            algo_spec=MDSA(list(range(mdsa_min, mdsa_max, 1))).__dict__
        )

        # Specify the maximum number of: (maximum number of graphs per run
        # size).
        self.min_max_graphs = 1
        self.max_max_graphs = 20

        # Specify the maximum graph size.
        self.min_graph_size = 3
        self.max_graph_size = 20
        # verify_min_max(self.min_graph_size, self.max_graph_size)

        # The size of the graph and the maximum number of used graphs of that
        # size.
        self.size_and_max_graphs = [
            (self.min_graph_size, self.max_max_graphs),
            (5, 4),  # Means: get 4 graphs of size 5 for experiment.
            (self.max_graph_size, self.max_max_graphs),
        ]

        # Overwrite the simulation results or not.
        self.recreate_s4 = True
        # Overwrite the visualisation of the SNN behaviour or not.
        self.recreate_s3 = True

        self.seeds = list(range(0, 1000))

        # The backend/type of simulator that is used.
        self.simulators = ["nx", "lava", "simsnn"]

        # Generate the supported adaptation settings.
        self.specify_supported_adaptation_settings()

        # Generate the supported radiations settings.
        self.specify_supported_radiations_settings()

        # Specify the supported image export file extensions.
        self.export_types = ["gif", "pdf", "png", "svg"]

    @typechecked
    def specify_supported_radiations_settings(self) -> None:
        """Specifies types of supported radiations settings. Some settings
        consist of a list of tuples, with the probability of a change
        occurring, followed by the average magnitude of the change.

        Others only contain a list of floats which represent the
        probability of radiations induced change occurring.
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

        # Create a supported radiations setting example.
        self.radiations = {
            # No radiations
            "None": [],
            # radiations effects are transient, they last for 1 or 10
            # simulation steps. If transient is 0., the changes are permanent.
            "transient": [0.0, 1.0, 10.0],
            # List of probabilities of a neuron dying due to radiations.
            "neuron_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            # List of probabilities of a synapse dying due to radiations.
            "synaptic_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            # List of: (probability of synaptic weight change, and the average
            # factor with which it changes due to radiations).
            "delta_synaptic_w": self.delta_synaptic_w,
            # List of: (probability of neuron threshold change, and the average
            # factor with which it changes due to radiations).
            "delta_vth": self.delta_vth,
        }

    @typechecked
    def specify_supported_adaptation_settings(self) -> None:
        """Specifies all the supported types of adaptation settings."""
        # Specify the (to be) supported adaptation types.
        self.adaptations = {
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


# pylint: disable=W0613
@typechecked
def verify_exp_config(
    *,
    supp_exp_config: Supported_experiment_settings,
    exp_config: Exp_config,
) -> None:
    """Verifies the selected experiment configuration settings are valid.

    :param exp_config: param has_unique_id:
    :param has_unique_id: param supp_exp_config:
    :param supp_exp_config:
    """

    # Verify the algorithms
    verify_algos_in_exp_config(exp_config=exp_config)

    # Verify settings of type: list and tuple.
    verify_list_setting(
        supp_exp_config=supp_exp_config,
        setting=exp_config.seeds,
        element_type=int,
        setting_name="seeds",
    )

    verify_list_setting(
        supp_exp_config=supp_exp_config,
        setting=exp_config.simulators,
        element_type=str,
        setting_name="simulators",
    )
    verify_size_and_max_graphs_settings(
        supp_exp_config=supp_exp_config,
        size_and_max_graphs_setting=exp_config.size_and_max_graphs,
    )

    # Verify settings of type integer.
    verify_integer_settings(
        integer_setting=exp_config.min_max_graphs,
        min_val=supp_exp_config.min_max_graphs,
        max_val=supp_exp_config.max_max_graphs,
    )
    verify_integer_settings(
        integer_setting=exp_config.max_max_graphs,
        min_val=supp_exp_config.min_max_graphs,
        max_val=supp_exp_config.max_max_graphs,
    )
    verify_integer_settings(
        integer_setting=exp_config.min_graph_size,
        min_val=supp_exp_config.min_graph_size,
        max_val=supp_exp_config.max_graph_size,
    )
    verify_integer_settings(
        integer_setting=exp_config.max_graph_size,
        min_val=supp_exp_config.min_graph_size,
        max_val=supp_exp_config.max_graph_size,
    )

    # Verify a lower bound/min is not larger than a upper bound/max value.
    verify_min_max(
        min_val=exp_config.min_graph_size,
        max_val=exp_config.max_graph_size,
    )
    verify_min_max(
        min_val=exp_config.min_max_graphs,
        max_val=exp_config.max_max_graphs,
    )


@typechecked
def verify_list_element_types_and_list_len(  # type:ignore[misc]
    *,
    list_setting: Any,
    element_type: type,
) -> None:
    """Verifies the types and minimum length of configuration settings that are
    stored with a value of type list.

    :param list_setting: param element_type:
    :param element_type:
    """
    verify_object_type(
        obj=list_setting, expected_type=list, element_type=element_type
    )
    if len(list_setting) < 1:
        raise ValueError(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{len(list_setting)}"
        )


def verify_list_setting(  # type:ignore[misc]
    *,
    supp_exp_config: Supported_experiment_settings,
    setting: Any,
    element_type: type,
    setting_name: str,
) -> None:
    """Verifies the configuration settings that have values of type list, that
    the list has at least 1 element in it, and that its values are within the
    supported range.

    :param setting: param supp_exp_config:
    :param element_type: param setting_name:
    :param supp_exp_config:
    :param setting_name:
    """

    # Check if the configuration setting is a list with length at least 1.
    verify_list_element_types_and_list_len(
        list_setting=setting, element_type=element_type
    )

    # Verify the configuration setting list elements are all within the
    # supported range.
    expected_range = get_expected_range(
        setting_name=setting_name, supp_exp_config=supp_exp_config
    )
    for element in setting:
        if element not in expected_range:
            raise ValueError(
                f"Error, {setting_name} was expected to be in range:"
                + f"{expected_range}. Instead, it"
                + f" contains:{element}."
            )


def get_expected_range(
    *, setting_name: str, supp_exp_config: Supported_experiment_settings
) -> list[int] | list[str]:
    """Returns the ranges as specified in the Supported_experiment_settings
    object for the asked setting.

    :param setting_name: param supp_exp_config:
    :param supp_exp_config:
    """
    if setting_name == "m_val":
        return list(range(MDSA([1]).min_m_vals, MDSA([1]).max_m_vals, 1))
    if setting_name == "simulators":
        return supp_exp_config.simulators
    if setting_name == "seeds":
        return supp_exp_config.seeds

    # TODO: test this is raised.
    raise NotImplementedError(
        f"Error, unsupported parameter requested:{setting_name}"
    )


def verify_size_and_max_graphs_settings(
    *,
    supp_exp_config: Supported_experiment_settings,
    size_and_max_graphs_setting: list[tuple[int, int]] | None,
) -> None:
    """Verifies the configuration setting size_and_max_graphs_setting values
    are a list of tuples with at least 1 tuple, and that its values are within
    the supported range.

    :param supp_exp_config:
    :param size_and_max_graphs_setting:
    :param supp_exp_config:
    """
    verify_list_element_types_and_list_len(
        list_setting=size_and_max_graphs_setting, element_type=tuple
    )

    # Verify the tuples contain valid values for size and max_graphs.
    if size_and_max_graphs_setting is not None:
        for size_and_max_graphs in size_and_max_graphs_setting:
            size = size_and_max_graphs[0]
            max_graphs = size_and_max_graphs[1]

            verify_integer_settings(
                integer_setting=size,
                min_val=supp_exp_config.min_graph_size,
                max_val=supp_exp_config.max_graph_size,
            )

            verify_integer_settings(
                integer_setting=max_graphs,
                min_val=supp_exp_config.min_max_graphs,
                max_val=supp_exp_config.max_max_graphs,
            )


@typechecked
def verify_integer_settings(
    *,
    integer_setting: int,
    min_val: int,
    max_val: int,
) -> None:
    """Verifies an integer setting is of type integer and that it is within the
    supported minimum and maximum value range.

    :param integer_setting:
    :param min_val:
    :param max_val:
    """
    if (min_val is not None) and (max_val is not None):
        if integer_setting < min_val:
            raise ValueError(
                f"Error, setting expected to be at least {min_val}. "
                + f"Instead, it is:{integer_setting}"
            )
        if integer_setting > max_val:
            raise ValueError(
                "Error, setting expected to be at most"
                + f" {max_val}. Instead, it is:"
                + f"{integer_setting}"
            )
    else:
        raise SyntaxError("Error, meaningless verification.")


@typechecked
def verify_min_max(*, min_val: int, max_val: int) -> None:
    """Verifies a lower bound/minimum value is indeed smaller than an
    upperbound/maximum value.

    Also verifies the values are either of type integer or float.
    """
    if min_val > max_val:
        raise ValueError(
            f"Lower bound:{min_val} is larger than upper bound:"
            + f"{max_val}."
        )


# TODO: determine why this can not be typechecked.
def verify_object_type(
    *,
    obj: float | list | tuple,
    expected_type: type,
    element_type: type | None = None,
) -> None:
    """Verifies an incoming object has the expected type, and if the object is
    a tuple or list, it also verifies the types of the elements in the tuple or
    list.

    :param obj: param expected_type:
    :param element_type: Default value = None
    :param expected_type:
    """

    # Verify the object type is as expected.
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"Error, expected type:{expected_type}, yet it was:{type(obj)}"
            + f" for:{obj}"
        )

    # If object is of type list or tuple, verify the element types.
    if isinstance(obj, (list, tuple)):
        # Verify user passed the expected element types.
        if element_type is None:
            raise TypeError("Expected a type to check list element types.")

        # Verify the element types.
        if not all(isinstance(n, element_type) for n in obj):
            # if list(map(type, obj)) != element_type:
            raise TypeError(
                f"Error, obj={obj}, its type is:{list(map(type, obj))},"
                + f" expected type:{element_type}"
            )


def verify_adap_and_rad_settings(
    *,
    supp_exp_config: Supported_experiment_settings,
    some_dict: dict | str | None,
    check_type: str,
) -> dict:
    """Verifies the settings of adaptations or radiations property are valid.
    Returns a dictionary with the adaptations setting if the settngs are valid.

    :param some_dict: param check_type:
    :param check_type: param supp_exp_config:
    :param supp_exp_config:
    """

    # Load the example settings from the Supported_experiment_settings object.
    if check_type == "adaptations":
        reference_object: dict[  # type:ignore[misc]
            str, Any
        ] = supp_exp_config.adaptations
    elif check_type == "radiations":
        reference_object = supp_exp_config.radiations
    else:
        raise TypeError(f"Check type:{check_type} not supported.")

    # Verify object is a dictionary.
    if isinstance(some_dict, Dict):
        if some_dict == {}:
            raise TypeError(f"Error, property Dict: {check_type} was empty.")
        for key in some_dict:
            # Verify the keys are within the supported dictionary keys.
            if key not in reference_object:
                raise TypeError(
                    f"Error, property.key:{key} is not in the supported "
                    + f"property keys:{reference_object.keys()}."
                )
            # Check if values belonging to key are within supported range.
            if check_type == "adaptations":
                verify_redundancy_settings_for_exp_config(
                    adaptation=some_dict[key]
                )
            elif check_type == "radiations":
                verify_radiations_values(
                    supp_exp_config=supp_exp_config,
                    radiations=some_dict,
                    key=key,
                )
        return some_dict
    raise TypeError(
        "Error, property is expected to be a Dict, yet"
        + f" it was of type: {type(some_dict)}."
    )


def verify_radiations_values(
    *,
    supp_exp_config: Supported_experiment_settings,
    radiations: dict,
    key: str,
) -> None:
    """The configuration settings contain key named: radiations. The value of
    belonging to this key is a dictionary, which also has several keys.

    This method checks whether these radiations dictionary keys, are within
    the supported range of adaptations setting keys. These adaptations
    dictionary keys should each have values of the type list. These list
    elements should have the type float, tuple(float, float) or be empty lists.
    The empty list represents: no radiations is used, signified by the key
    name: "None".

    This method verifies the keys in the adaptations dictionary are within the
    supported range. It also checks if the values of the adaptations dictionary
    keys are a list, and whether all elements in those lists are of type float
    or tuple. If the types are tuple, it also checks whether the values within
    those tuples are of type float.

    :param radiations: Dict:
    :param key: str:
    :param supp_exp_config:
    """
    if not isinstance(
        radiations[key], type(supp_exp_config.radiations[key])
    ) or (not isinstance(radiations[key], list)):
        raise TypeError(
            "Error, the radiations value is of type:"
            + f"{type(radiations[key])}, yet it was expected to be"
            + " float or dict."
        )

    # Verify radiations setting types.
    if isinstance(radiations[key], list):
        for setting in radiations[key]:
            # Verify radiations setting can be of type float.
            if isinstance(setting, float):
                # TODO: superfluous check.
                verify_object_type(
                    obj=setting, expected_type=float, element_type=None
                )
            # Verify radiations setting can be of type tuple.
            elif isinstance(setting, tuple):
                # Verify the radiations setting tuple is of type float,
                # float.
                # TODO: change type((1.0, 2.0)) to the type it is.
                verify_object_type(
                    obj=setting,
                    expected_type=tuple,
                    element_type=type((1.0, 2.0)),
                )

            else:
                # Throw error if the radiations setting is something other
                # than a float or tuple of floats.
                raise TypeError(
                    f"Unexpected setting type:{type(setting)} for:"
                    + f" {setting}."
                )
