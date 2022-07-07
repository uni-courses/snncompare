"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
# pylint: disable=R0801
from typing import Any, Dict


# pylint: disable=W0613
def verify_run_config(supp_sets, run_config, has_unique_id):
    """Verifies the selected experiment configuration settings are valid.

    :param run_config: param has_unique_id:
    :param has_unique_id: param supp_sets:
    :param supp_sets:
    """
    if not isinstance(has_unique_id, bool):
        raise Exception(f"has_unique_id={has_unique_id}, should be a boolean")
    if not isinstance(run_config, dict):
        raise Exception(
            "Error, the run_config is of type:"
            + f"{type(run_config)}, yet it was expected to be of"
            + " type dict."
        )

    verify_run_config_dict_is_complete(supp_sets, run_config)
    # Verify no unknown configuration settings are presented.
    verify_run_config_dict_contains_only_valid_entries(supp_sets, run_config)

    # Verify settings of type: list and tuple.
    verify_list_setting(
        supp_sets,
        run_config["algorithms"]["MDSA"]["m_vals"],
        int,
        "m_vals",
    )
    verify_list_setting(supp_sets, run_config["iterations"], int, "iterations")

    verify_list_setting(supp_sets, run_config["simulators"], str, "simulators")
    verify_size_and_max_graphs_settings(
        supp_sets, run_config["size_and_max_graphs"]
    )

    # Verify settings of type integer.
    verify_integer_settings(run_config["seed"])

    verify_integer_settings(
        run_config["min_max_graphs"],
        supp_sets.min_max_graphs,
        supp_sets.max_max_graphs,
    )
    verify_integer_settings(
        run_config["max_max_graphs"],
        supp_sets.min_max_graphs,
        supp_sets.max_max_graphs,
    )
    verify_integer_settings(
        run_config["min_graph_size"],
        supp_sets.min_graph_size,
        supp_sets.max_graph_size,
    )
    verify_integer_settings(
        run_config["max_graph_size"],
        supp_sets.min_graph_size,
        supp_sets.max_graph_size,
    )

    # Verify a lower bound/min is not larger than a upper bound/max value.
    verify_min_max(
        run_config["min_graph_size"],
        run_config["max_graph_size"],
    )
    verify_min_max(
        run_config["min_max_graphs"],
        run_config["max_max_graphs"],
    )

    # Verify settings of type bool.
    verify_bool_setting(run_config["overwrite_sim_results"])
    verify_bool_setting(run_config["overwrite_visualisation"])

    if has_unique_id:
        print("TODO: test unique id type.")
    return run_config


def verify_run_config_dict_is_complete(supp_sets, run_config):
    """Verifies the configuration settings dictionary is complete."""
    print(f"run_config.keys()={run_config.keys()}")
    for expected_key in supp_sets.config_setting_parameters:
        print(f"expected_key={expected_key}")
        if expected_key not in run_config.keys():
            raise Exception(
                f"Error:{expected_key} is not in the configuration"
                + f" settings:{run_config.keys()}"
            )


def verify_run_config_dict_contains_only_valid_entries(supp_sets, run_config):
    """Verifies the configuration settings dictionary does not contain any
    invalid keys."""
    for actual_key in run_config.keys():
        if actual_key not in supp_sets.config_setting_parameters:
            raise Exception(
                f"Error:{actual_key} is not supported by the configuration"
                + f" settings:{supp_sets.config_setting_parameters}"
            )


def verify_list_element_types_and_list_len(list_setting, element_type):
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


def verify_list_setting(supp_sets, setting, element_type, setting_name):
    """Verifies the configuration settings that have values of type list, that
    the list has at least 1 element in it, and that its values are within the
    supported range.

    :param setting: param supp_sets:
    :param element_type: param setting_name:
    :param supp_sets:
    :param setting_name:
    """

    # Check if the configuration setting is a list with length at least 1.
    verify_list_element_types_and_list_len(setting, element_type)

    # Verify the configuration setting list elements are all within the
    # supported range.
    expected_range = get_expected_range(setting_name, supp_sets)
    for element in setting:
        if element not in expected_range:
            raise Exception(
                f"Error, {setting_name} was expected to be in range:"
                + f"{expected_range}. Instead, it"
                + f" contains:{element}."
            )


def get_expected_range(setting_name, supp_sets):
    """Returns the ranges as specified in the Supported_experiment_settings
    object for the asked setting.

    :param setting_name: param supp_sets:
    :param supp_sets:
    """
    if setting_name == "iterations":
        return supp_sets.iterations
    if setting_name == "m_vals":
        return supp_sets.algorithms["MDSA"].m_vals
    if setting_name == "simulators":
        return supp_sets.simulators

    # TODO: test this is raised.
    raise Exception("Error, unsupported parameter requested.")


def verify_size_and_max_graphs_settings(
    supp_sets, size_and_max_graphs_setting
):
    """Verifies the configuration setting size_and_max_graphs_setting values
    are a list of tuples with at least 1 tuple, and that its values are within
    the supported range.

    :param supp_sets:
    :param size_and_max_graphs_setting:
    :param supp_sets:
    """
    verify_list_element_types_and_list_len(size_and_max_graphs_setting, tuple)

    # Verify the tuples contain valid values for size and max_graphs.
    for size_and_max_graphs in size_and_max_graphs_setting:
        size = size_and_max_graphs[0]
        max_graphs = size_and_max_graphs[1]

        verify_integer_settings(
            size,
            supp_sets.min_graph_size,
            supp_sets.max_graph_size,
        )

        verify_integer_settings(
            max_graphs,
            supp_sets.min_max_graphs,
            supp_sets.max_max_graphs,
        )


def verify_integer_settings(integer_setting, min_val=None, max_val=None):
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


def verify_min_max(min_val, max_val):
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


def verify_bool_setting(bool_setting):
    """Verifies the bool_setting value is of type: boolean.

    :param bool_setting:
    """
    if not isinstance(bool_setting, bool):
        raise Exception(
            f"Error, expected type:{bool}, yet it was:"
            + f"{type(bool_setting)} for:{bool_setting}"
        )


def verify_object_type(obj, expected_type, element_type=None):
    """Verifies an incoming object has the expected type, and if the object is
    a tuple or list, it also verifies the types of the elements in the tuple or
    list.

    :param obj: param expected_type:
    :param element_type: Default value = None)
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
        print(f"element_type={element_type}")
        if not all(isinstance(n, element_type) for n in obj):

            # if list(map(type, obj)) != element_type:
            raise Exception(
                f"Error, obj={obj}, its type is:{list(map(type, obj))},"
                + f" expected type:{element_type}"
            )


def verify_adap_and_rad_settings(supp_sets, some_dict, check_type) -> dict:
    """Verifies the settings of adaptation or radiation property are valid.
    Returns a dictionary with the adaptation setting if the settngs are valid.

    :param some_dict: param check_type:
    :param check_type: param supp_sets:
    :param supp_sets:
    """

    # Load the example settings from the Supported_experiment_settings object.
    if check_type == "adaptation":
        reference_object: Dict[str, Any] = supp_sets.adaptation
    elif check_type == "radiation":
        reference_object = supp_sets.radiation
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
            if check_type == "adaptation":
                verify_adaptation_values(supp_sets, some_dict, key)
            elif check_type == "radiation":
                verify_radiation_values(supp_sets, some_dict, key)
        return some_dict
    raise Exception(
        "Error, property is expected to be a dict, yet"
        + f" it was of type: {type(some_dict)}."
    )


def verify_algorithm_settings(supp_sets, some_dict, check_type) -> dict:
    """TODO: Verifies the settings of the algorithm are valid."""


def verify_adaptation_values(supp_sets, adaptation: dict, key: str) -> None:
    """The configuration settings contain key named: adaptation. The value of
    belonging to this key is a dictionary, which also has several keys.

    This method checks whether these adaptation dictionary keys, are within
    the supported range of adaptation setting keys. These adaptation dictionary
    keys should each have values of the type list. These list elements should
    have the type float, or be empty lists. The empty list represents: no
    adaptation is used, signified by the key name: "None".

    This method verifies the keys in the adaptation dictionary are within the
    supported range. It also checks if the values of the adaptation dictionary
    keys are a list, and whether all elements in those lists are of type float.

    :param adaptation: dict:
    :param key: str:
    :param supp_sets:
    """

    # Verifies the configuration settings adaptation value is of the same type
    # as the supported adaptation configuration setting (which is a list)).
    if not isinstance(adaptation[key], type(supp_sets.adaptation[key])) and (
        not isinstance(adaptation[key], list)
    ):
        raise Exception(
            f'Error, value of adaptation["{key}"]='
            + f"{adaptation[key]}, (which has type:{type(adaptation[key])}"
            + "), is of different type than the expected and supported "
            + f"type: {type(supp_sets.adaptation[key])}"
        )

    # Verifies the values in the list of adaptation settings are of type float.
    if isinstance(adaptation[key], list):
        for setting in adaptation[key]:
            verify_object_type(setting, float, None)


def verify_radiation_values(supp_sets, radiation: dict, key: str) -> None:
    """The configuration settings contain key named: radiation. The value of
    belonging to this key is a dictionary, which also has several keys.

    This method checks whether these radiation dictionary keys, are within
    the supported range of adaptation setting keys. These adaptation dictionary
    keys should each have values of the type list. These list elements should
    have the type float, tuple(float, float) or be empty lists. The empty list
    represents: no radiation is used, signified by the key name: "None".

    This method verifies the keys in the adaptation dictionary are within the
    supported range. It also checks if the values of the adaptation dictionary
    keys are a list, and whether all elements in those lists are of type float
    or tuple. If the types are tuple, it also checks whether the values within
    those tuples are of type float.

    :param radiation: dict:
    :param key: str:
    :param supp_sets:
    """
    if not isinstance(radiation[key], type(supp_sets.radiation[key])) or (
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
                verify_object_type(setting, float, None)
            # Verify radiation setting can be of type tuple.
            elif isinstance(setting, tuple):
                # Verify the radiation setting tuple is of type float,
                # float.
                verify_object_type(setting, tuple, (float, float))
            else:
                # Throw error if the radiation setting is something other
                # than a float or tuple of floats.
                raise Exception(
                    f"Unexpected setting type:{type(setting)} for:"
                    + f" {setting}."
                )


def verify_has_unique_id(run_config):
    """Verifies the config setting has a unique id."""
    if not isinstance(run_config, dict):
        raise Exception(
            "The configuration settings is not a dictionary,"
            + f"instead it is: of type:{type(run_config)}."
        )
    if "unique_id" not in run_config.keys():
        raise Exception(
            "The configuration settings do not contain a unique id even though"
            + f" that was expected. run_config is:{run_config}."
        )
