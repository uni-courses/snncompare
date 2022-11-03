"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.snncompare.exp_setts.algos.MDSA import MDSA
from src.snncompare.exp_setts.algos.verify_algos import (
    verify_algos_in_experiment_config,
)

if TYPE_CHECKING:
    from src.snncompare.exp_setts.Supported_experiment_settings import (
        Supported_experiment_settings,
    )


# pylint: disable=W0613
def verify_experiment_config(
    supp_exp_setts: Supported_experiment_settings,
    experiment_config: None | str | dict,
    has_unique_id: bool,
    strict: bool,
) -> None:
    """Verifies the selected experiment configuration settings are valid.

    :param experiment_config: param has_unique_id:
    :param has_unique_id: param supp_exp_setts:
    :param supp_exp_setts:
    """
    if not isinstance(has_unique_id, bool):
        raise Exception(f"has_unique_id={has_unique_id}, should be a boolean")
    if not isinstance(experiment_config, dict):
        raise Exception(
            "Error, the experiment_config is of type:"
            + f"{type(experiment_config)}, yet it was expected to be of"
            + " type dict."
        )

    verify_experiment_config_dict_is_complete(
        supp_exp_setts, experiment_config
    )
    # Verify no unknown configuration settings are presented.
    verify_experiment_config_dict_contains_only_valid_entries(
        supp_exp_setts, experiment_config, strict
    )

    # Verify the algorithms
    verify_algos_in_experiment_config(experiment_config)

    # Verify settings of type: list and tuple.
    verify_list_setting(
        supp_exp_setts, experiment_config["iterations"], int, "iterations"
    )

    verify_list_setting(
        supp_exp_setts, experiment_config["simulators"], str, "simulators"
    )
    verify_size_and_max_graphs_settings(
        supp_exp_setts, experiment_config["size_and_max_graphs"]
    )

    # Verify settings of type integer.
    verify_integer_settings(experiment_config["seed"])

    verify_integer_settings(
        experiment_config["min_max_graphs"],
        supp_exp_setts.min_max_graphs,
        supp_exp_setts.max_max_graphs,
    )
    verify_integer_settings(
        experiment_config["max_max_graphs"],
        supp_exp_setts.min_max_graphs,
        supp_exp_setts.max_max_graphs,
    )
    verify_integer_settings(
        experiment_config["min_graph_size"],
        supp_exp_setts.min_graph_size,
        supp_exp_setts.max_graph_size,
    )
    verify_integer_settings(
        experiment_config["max_graph_size"],
        supp_exp_setts.min_graph_size,
        supp_exp_setts.max_graph_size,
    )

    # Verify a lower bound/min is not larger than a upper bound/max value.
    verify_min_max(
        experiment_config["min_graph_size"],
        experiment_config["max_graph_size"],
    )
    verify_min_max(
        experiment_config["min_max_graphs"],
        experiment_config["max_max_graphs"],
    )

    # Verify settings of type bool.
    verify_bool_setting(experiment_config["overwrite_sim_results"])
    verify_bool_setting(experiment_config["overwrite_visualisation"])


def verify_experiment_config_dict_is_complete(
    supp_exp_setts: Supported_experiment_settings,
    experiment_config: dict[str, Any],
) -> None:
    """Verifies the configuration settings dictionary is complete."""
    for expected_key in supp_exp_setts.parameters:
        if expected_key not in experiment_config.keys():
            raise Exception(
                f"Error:{expected_key} is not in the configuration"
                + f" settings:{experiment_config.keys()}"
            )


def verify_experiment_config_dict_contains_only_valid_entries(
    supp_exp_setts: Supported_experiment_settings,
    experiment_config: dict,
    strict: bool,
) -> None:
    """Verifies the configuration settings dictionary does not contain any
    invalid keys."""
    for actual_key in experiment_config.keys():
        if actual_key not in supp_exp_setts.parameters:
            if strict:
                raise Exception(
                    f"Error:{actual_key} is not supported by the configuration"
                    + f" settings:{supp_exp_setts.parameters}"
                )
            if actual_key not in supp_exp_setts.optional_parameters:
                raise Exception(
                    f"Error:{actual_key} is not supported by the configuration"
                    + f" settings:{supp_exp_setts.parameters}, nor by the"
                    + " optional settings:"
                    + f"{supp_exp_setts.optional_parameters}"
                )


def verify_list_element_types_and_list_len(
    list_setting: Any, element_type: type
) -> None:
    """Verifies the types and minimum length of configuration settings that are
    stored with a value of type list.

    :param list_setting: param element_type:
    :param element_type:
    """
    verify_object_type(list_setting, list, element_type=element_type)
    if len(list_setting) < 1:
        raise Exception(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{len(list_setting)}"
        )


def verify_list_setting(
    supp_exp_setts: Supported_experiment_settings,
    setting: Any,
    element_type: type,
    setting_name: str,
) -> None:
    """Verifies the configuration settings that have values of type list, that
    the list has at least 1 element in it, and that its values are within the
    supported range.

    :param setting: param supp_exp_setts:
    :param element_type: param setting_name:
    :param supp_exp_setts:
    :param setting_name:
    """

    # Check if the configuration setting is a list with length at least 1.
    verify_list_element_types_and_list_len(setting, element_type)

    # Verify the configuration setting list elements are all within the
    # supported range.
    expected_range = get_expected_range(setting_name, supp_exp_setts)
    for element in setting:
        if element not in expected_range:
            raise Exception(
                f"Error, {setting_name} was expected to be in range:"
                + f"{expected_range}. Instead, it"
                + f" contains:{element}."
            )


def get_expected_range(
    setting_name: str, supp_exp_setts: Supported_experiment_settings
) -> list[int] | list[str]:
    """Returns the ranges as specified in the Supported_experiment_settings
    object for the asked setting.

    :param setting_name: param supp_exp_setts:
    :param supp_exp_setts:
    """
    if setting_name == "iterations":
        return supp_exp_setts.iterations
    if setting_name == "m_val":
        print(f"supp_exp_setts.algorithms={supp_exp_setts.algorithms}")
        # return supp_exp_setts.algorithms["MDSA"]
        return list(range(MDSA([1]).min_m_vals, MDSA([1]).max_m_vals, 1))
    if setting_name == "simulators":
        return supp_exp_setts.simulators

    # TODO: test this is raised.
    raise Exception("Error, unsupported parameter requested.")


def verify_size_and_max_graphs_settings(
    supp_exp_setts: Supported_experiment_settings,
    size_and_max_graphs_setting: list[tuple[int, int]] | None,
) -> None:
    """Verifies the configuration setting size_and_max_graphs_setting values
    are a list of tuples with at least 1 tuple, and that its values are within
    the supported range.

    :param supp_exp_setts:
    :param size_and_max_graphs_setting:
    :param supp_exp_setts:
    """
    verify_list_element_types_and_list_len(size_and_max_graphs_setting, tuple)

    # Verify the tuples contain valid values for size and max_graphs.
    if size_and_max_graphs_setting is not None:
        for size_and_max_graphs in size_and_max_graphs_setting:
            size = size_and_max_graphs[0]
            max_graphs = size_and_max_graphs[1]

            verify_integer_settings(
                size,
                supp_exp_setts.min_graph_size,
                supp_exp_setts.max_graph_size,
            )

            verify_integer_settings(
                max_graphs,
                supp_exp_setts.min_max_graphs,
                supp_exp_setts.max_max_graphs,
            )


def verify_integer_settings(
    integer_setting: int | None,
    min_val: int | None = None,
    max_val: int | None = None,
) -> None:
    """Verifies an integer setting is of type integer and that it is within the
    supported minimum and maximum value range..

    :param integer_setting:
    :param min_val:
    :param max_val:
    """
    if not isinstance(integer_setting, int):
        raise Exception(
            f"Error, expected type:{int}, yet it was:"
            + f"{type(integer_setting)} for:{integer_setting}"
        )
    if (min_val is not None) and (max_val is not None):

        if integer_setting < min_val:
            raise Exception(
                f"Error, setting expected to be at least {min_val}. "
                + f"Instead, it is:{integer_setting}"
            )
        if integer_setting > max_val:
            raise Exception(
                "Error, setting expected to be at most"
                + f" {max_val}. Instead, it is:"
                + f"{integer_setting}"
            )


def verify_min_max(min_val: int, max_val: int) -> None:
    """Verifies a lower bound/minimum value is indeed smaller than an
    upperbound/maximum value.

    Also verifies the values are either of type integer or float.
    """
    for val in [min_val, max_val]:
        if not isinstance(val, (float, int)):
            raise Exception("Expected {val} to be of type int, or float.")
    if min_val > max_val:
        raise Exception(
            f"Lower bound:{min_val} is larger than upper bound:"
            + f"{max_val}."
        )


def verify_bool_setting(bool_setting: None | bool | str) -> None:
    """Verifies the bool_setting value is of type: boolean.

    :param bool_setting:
    """
    if not isinstance(bool_setting, bool):
        raise Exception(
            f"Error, expected type:{bool}, yet it was:"
            + f"{type(bool_setting)} for:{bool_setting}"
        )


def verify_object_type(
    obj: float | list | tuple,
    expected_type: type,
    element_type: type = None,
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
        raise Exception(
            f"Error, expected type:{expected_type}, yet it was:{type(obj)}"
            + f" for:{obj}"
        )

    # If object is of type list or tuple, verify the element types.
    if isinstance(obj, (list, tuple)):

        # Verify user passed the expected element types.
        if element_type is None:
            raise Exception("Expected a type to check list element types.")

        # Verify the element types.
        if not all(isinstance(n, element_type) for n in obj):

            # if list(map(type, obj)) != element_type:
            raise Exception(
                f"Error, obj={obj}, its type is:{list(map(type, obj))},"
                + f" expected type:{element_type}"
            )


def verify_adap_and_rad_settings(
    supp_exp_setts: Supported_experiment_settings,
    some_dict: dict | str | None,
    check_type: str,
) -> dict:
    """Verifies the settings of adaptations or radiations property are valid.
    Returns a dictionary with the adaptations setting if the settngs are valid.

    :param some_dict: param check_type:
    :param check_type: param supp_exp_setts:
    :param supp_exp_setts:
    """

    # Load the example settings from the Supported_experiment_settings object.
    if check_type == "adaptations":
        reference_object: dict[str, Any] = supp_exp_setts.adaptations
    elif check_type == "radiations":
        reference_object = supp_exp_setts.radiations
    else:
        raise Exception(f"Check type:{check_type} not supported.")

    # Verify object is a dictionary.
    if isinstance(some_dict, dict):
        if some_dict == {}:
            raise Exception(f"Error, property dict: {check_type} was empty.")
        for key in some_dict:

            # Verify the keys are within the supported dictionary keys.
            if key not in reference_object:
                raise Exception(
                    f"Error, property.key:{key} is not in the supported "
                    + f"property keys:{reference_object.keys()}."
                )
            # Check if values belonging to key are within supported range.
            if check_type == "adaptations":
                verify_adaptation_values(supp_exp_setts, some_dict, key)
            elif check_type == "radiations":
                verify_radiations_values(supp_exp_setts, some_dict, key)
        return some_dict
    raise Exception(
        "Error, property is expected to be a dict, yet"
        + f" it was of type: {type(some_dict)}."
    )


def verify_algorithm_settings(
    supp_exp_setts: Supported_experiment_settings,
    some_dict: dict,
    check_type: str,
) -> dict:
    """TODO: Verifies the settings of the algorithm are valid."""


def verify_adaptation_values(
    supp_exp_setts: Supported_experiment_settings, adaptations: dict, key: str
) -> None:
    """The configuration settings contain key named: adaptations. The value of
    belonging to this key is a dictionary, which also has several keys.

    This method checks whether these adaptations dictionary keys, are within
    the supported range of adaptations setting keys. These adaptations
    dictionary keys should each have values of the type list. These list
    elements should have the type float, or be empty lists. The empty list
    represents: no adaptations is used, signified by the key name: "None".

    This method verifies the keys in the adaptations dictionary are within the
    supported range. It also checks if the values of the adaptations dictionary
    keys are a list, and whether all elements in those lists are of type float.

    :param adaptations: dict:
    :param key: str:
    :param supp_exp_setts:
    """

    # Verifies the configuration settings adaptations value is of the same type
    # as the supported adaptations configuration setting (which is a list)).
    if not isinstance(
        adaptations[key], type(supp_exp_setts.adaptations[key])
    ) and (not isinstance(adaptations[key], list)):
        raise Exception(
            f'Error, value of adaptations["{key}"]='
            + f"{adaptations[key]}, (which has type:{type(adaptations[key])}"
            + "), is of different type than the expected and supported "
            + f"type: {type(supp_exp_setts.adaptations[key])}"
        )

    # Verifies the values in the list of adaptations settings are of type
    # float.
    if isinstance(adaptations[key], list):
        for setting in adaptations[key]:
            verify_object_type(setting, float, None)


def verify_radiations_values(
    supp_exp_setts: Supported_experiment_settings, radiations: dict, key: str
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

    :param radiations: dict:
    :param key: str:
    :param supp_exp_setts:
    """
    if not isinstance(
        radiations[key], type(supp_exp_setts.radiations[key])
    ) or (not isinstance(radiations[key], list)):

        raise Exception(
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
                verify_object_type(setting, float, None)
            # Verify radiations setting can be of type tuple.
            elif isinstance(setting, tuple):
                # Verify the radiations setting tuple is of type float,
                # float.
                # TODO: change type((1.0, 2.0)) to the type it is.
                verify_object_type(setting, tuple, type((1.0, 2.0)))

            else:
                # Throw error if the radiations setting is something other
                # than a float or tuple of floats.
                raise Exception(
                    f"Unexpected setting type:{type(setting)} for:"
                    + f" {setting}."
                )


def verify_has_unique_id(experiment_config: dict) -> None:
    """Verifies the config setting has a unique id."""
    if not isinstance(experiment_config, dict):
        raise Exception(
            "The configuration settings is not a dictionary,"
            + f"instead it is: of type:{type(experiment_config)}."
        )
    if "unique_id" not in experiment_config.keys():
        raise Exception(
            "The configuration settings do not contain a unique id even though"
            + f" that was expected. experiment_config is:{experiment_config}."
        )
