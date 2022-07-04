"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
from typing import Any, Dict


# pylint: disable=W0613
def verify_configuration_settings(supp_sets, experiment_config, has_unique_id):
    """TODO: Verifies the experiment configuration settings are valid.

    :param experiment_config: param has_unique_id:
    :param has_unique_id:
    :param supp_sets:

    """
    if not isinstance(has_unique_id, bool):
        raise Exception(f"has_unique_id={has_unique_id}, should be a boolean")
    if not isinstance(experiment_config, dict):
        raise Exception(
            "Error, the experiment_config is of type:"
            + f"{type(experiment_config)}, yet it was expected to be of"
            + " type dict."
        )

    # Verify settings of type: list and tuple.
    verify_list_setting(supp_sets, experiment_config["m"], int, "m")
    verify_list_setting(
        supp_sets, experiment_config["iterations"], int, "iterations"
    )
    verify_size_and_max_graphs_settings(
        supp_sets, experiment_config["size_and_max_graphs"]
    )

    # Verify settings of type integer.
    # TODO: verify min is smaller than max for supported settings.
    # TODO: verify min is smaller than max for experiment_config.
    verify_integer_settings(
        supp_sets,
        experiment_config["min_max_graphs"],
        supp_sets.min_max_graphs,
        supp_sets.max_max_graphs,
    )
    verify_integer_settings(
        supp_sets,
        experiment_config["max_max_graphs"],
        supp_sets.min_max_graphs,
        supp_sets.max_max_graphs,
    )
    verify_integer_settings(
        supp_sets,
        experiment_config["min_graph_size"],
        supp_sets.min_graph_size,
        supp_sets.max_graph_size,
    )
    verify_integer_settings(
        supp_sets,
        experiment_config["max_graph_size"],
        supp_sets.min_graph_size,
        supp_sets.max_graph_size,
    )

    # Verify settings of type bool.
    verify_bool_setting(experiment_config["overwrite_sim_results"])
    verify_bool_setting(experiment_config["overwrite_visualisation"])

    if has_unique_id:
        print("TODO: test unique id type.")
    return experiment_config


def verify_list_element_types_and_list_len(list_setting, element_type):
    """Verifies the types and length of configuration settings that are stored
    with a value of type list.

    :param list_setting:
    :param element_type:
    """
    verify_object_type(list_setting, list, element_type=element_type)
    if len(list_setting) < 1:
        raise Exception(
            "Error, list was expected contain at least 1 integer."
            + f" Instead, it has length:{len(list_setting)}"
        )


def verify_list_setting(
    supp_sets, iterations_setting, element_type, setting_name
):
    """Verifies the type of m setting is valid, and that its values are within
    the supported range.

    :param iterations_setting:
    :param supp_sets:
    :param element_type:
    :param setting_name:
    """

    verify_list_element_types_and_list_len(iterations_setting, element_type)
    for iteration in iterations_setting:
        if iteration not in supp_sets.iterations:
            raise Exception(
                f"Error, {setting_name} was expected to be in range:"
                + f"{get_expected_range(setting_name,supp_sets)}. Instead, it"
                + f" contains:{iteration}."
            )


def get_expected_range(setting_name, supp_sets):
    """

    :param setting_name:
    :param supp_sets:

    """
    if setting_name == "iterations":
        return supp_sets.iterations
    if setting_name == "m":
        return supp_sets.m
    raise Exception("Error, unsupported parameter requested.")


def verify_size_and_max_graphs_settings(
    supp_sets, size_and_max_graphs_setting
):
    """Verifies the type of m setting is valid, and that its values are within
    the supported range.

    :param iterations_setting:
    :param supp_sets:
    :param size_and_max_graphs_setting:
    """
    print(f"size_and_max_graphs_setting={size_and_max_graphs_setting}")
    verify_list_element_types_and_list_len(size_and_max_graphs_setting, tuple)

    for size_and_max_graphs in size_and_max_graphs_setting:
        size = size_and_max_graphs[0]
        max_graphs = size_and_max_graphs[1]

        verify_integer_settings(
            supp_sets,
            size,
            supp_sets.min_graph_size,
            supp_sets.max_graph_size,
        )

        verify_integer_settings(
            supp_sets,
            max_graphs,
            supp_sets.min_max_graphs,
            supp_sets.max_max_graphs,
        )


def verify_integer_settings(
    supp_sets, max_max_graphs_setting, min_val, max_val
):
    """Verifies the maximum value that the max_graphs per size can have is a
    positive integer.

    # TODO: verify min is smaller than max for supported settings.

    :param max_max_graphs_setting:
    :param supp_sets:
    :param min_val:
    :param max_val:
    """
    if not isinstance(max_max_graphs_setting, int):
        raise Exception(
            f"Error, expected type:{int}, yet it was:"
            + f"{type(max_max_graphs_setting)}"
        )
    if max_max_graphs_setting < min_val:
        raise Exception(
            f"Error, setting expected to be at least {min_val}. "
            + f"Instead, it is:{max_max_graphs_setting}"
        )
    if max_max_graphs_setting > max_val:
        raise Exception(
            "Error, setting expected to be at most"
            + f" {max_val}. Instead, it is:"
            + f"{max_max_graphs_setting}"
        )


def verify_bool_setting(bool_setting):
    """Verifies the bool_setting value is a boolean.

    :param bool_setting:
    """
    if not isinstance(bool_setting, bool):
        raise Exception(
            f"Error, expected type:{bool}, yet it was:"
            + f"{type(bool_setting)}"
        )


def verify_object_type(obj, expected_type, element_type=None):
    """Verifies an object type, and if the object is a tuple, it also verifies
    the types within the tuple or list.

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

    :param some_dict: param check_type:
    :param check_type:
    :param supp_sets:
    """
    if check_type == "adaptation":
        reference_object: Dict[str, Any] = supp_sets.adaptation
    elif check_type == "radiation":
        reference_object = supp_sets.radiation
    else:
        raise Exception(f"Check type:{check_type} not supported.")

    # Verify object is a dictionary.
    print(f"some_dict={some_dict}")
    if isinstance(some_dict, dict):
        if some_dict == {}:
            raise Exception(f"Error, property dict: {check_type} was empty.")
        for key in some_dict:

            # Verify the keys are within permissible keys.
            if key not in reference_object:
                raise Exception(
                    f"Error, property.key:{key} is not in the supported "
                    + f"property keys:{reference_object.keys()}."
                )
            # Check values belonging to key
            if check_type == "adaptation":
                verify_adaptation_values(supp_sets, some_dict, key)
            elif check_type == "radiation":
                verify_radiation_values(supp_sets, some_dict, key)
        return some_dict
    raise Exception(
        "Error, property is expected to be a dict, yet"
        + f" it was of type: {type(some_dict)}."
    )


def verify_adaptation_values(supp_sets, adaptation: dict, key: str) -> None:
    """

    :param adaptation: dict:
    :param key: str:
    :param supp_sets:

    """

    if not isinstance(adaptation[key], type(supp_sets.adaptation[key])) or (
        not isinstance(adaptation[key], float)
        and not isinstance(adaptation[key], list)
    ):
        raise Exception(
            f'Error, value of adaptation["{key}"]='
            + f"{adaptation[key]}, (which has type:{type(adaptation[key])}"
            + "), is of different type than the expected and supported "
            + f"type: {type(supp_sets.adaptation[key])}"
        )
    # TODO: verify the elements in the list are of type float, if the value
    # is a list.
    if isinstance(adaptation[key], list):
        for setting in adaptation[key]:
            verify_object_type(setting, float, None)


def verify_radiation_values(supp_sets, radiation: dict, key: str) -> None:
    """

    :param radiation: dict:
    :param key: str:

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
