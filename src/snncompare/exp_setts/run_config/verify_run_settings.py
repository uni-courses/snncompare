"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
# pylint: disable=R0801

# controllers.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.snncompare.exp_setts.verify_experiment_settings import (
    verify_integer_settings,
)

if TYPE_CHECKING:
    from src.snncompare.exp_setts.run_config.Supported_run_settings import (
        Supported_run_settings,
    )


# pylint: disable=W0613
def verify_run_config(
    supp_run_setts: Supported_run_settings,
    run_config: str | dict | None,
    has_unique_id: bool,
    strict: bool,
) -> dict:
    """Verifies the selected experiment configuration settings are valid.

    :param run_config: param has_unique_id:
    :param has_unique_id: param supp_exp_setts:
    :param supp_exp_setts:
    """
    if not isinstance(has_unique_id, bool):
        raise Exception(f"has_unique_id={has_unique_id}, should be a boolean")
    if not isinstance(run_config, dict):
        raise Exception(
            "Error, the run_config is of type:"
            + f"{type(run_config)}, yet it was expected to be of"
            + " type dict."
        )

    verify_run_config_dict_is_complete(supp_run_setts, run_config)

    # Verify no unknown configuration settings are presented.
    verify_run_config_dict_contains_only_valid_entries(
        supp_run_setts, run_config, strict
    )

    verify_parameter_types(supp_run_setts, run_config)

    # TODO: verify a single algorithm is evaluated in a single run.
    verify_integer_settings(run_config["algorithm"]["MDSA"]["m_val"])
    # TODO: verify radiation setting for single run.

    # TODO: test unique id type
    return run_config


def verify_parameter_types(
    supp_run_setts: Supported_run_settings, run_config: dict[str, Any]
) -> None:
    """Checks for each parameter in the supported_run_settings object whether
    it is of a valid type."""
    for supported_key in supp_run_setts.parameters.keys():
        if not isinstance(
            run_config[supported_key], supp_run_setts.parameters[supported_key]
        ):
            raise Exception(
                f"Error, {supported_key} is of type: "
                + f"{type(run_config[supported_key])} whereas it is expected"
                " to be of type :"
                + f"{supp_run_setts.parameters[supported_key]}"
            )


def verify_run_config_dict_is_complete(
    supp_run_setts: Supported_run_settings, run_config: dict[str, Any]
) -> None:
    """Verifies the configuration settings dictionary is complete."""
    for expected_key in supp_run_setts.parameters.keys():
        if expected_key not in run_config.keys():
            raise Exception(
                f"Error:{expected_key} is not in the configuration"
                + f" settings:{run_config.keys()}"
            )


def verify_run_config_dict_contains_only_valid_entries(
    supp_run_setts: Supported_run_settings, run_config: dict, strict: bool
) -> None:
    """Verifies the configuration settings dictionary does not contain any
    invalid keys."""
    for actual_key in run_config.keys():
        if actual_key not in supp_run_setts.parameters:
            if strict:
                # TODO: allow for optional arguments:
                # stage, show, export, duration
                raise Exception(
                    f"Error:{actual_key} is not supported by the configuration"
                    + f" settings:{supp_run_setts.parameters}"
                )
            if actual_key not in supp_run_setts.optional_parameters:
                raise Exception(
                    f"Error:{actual_key} is not supported by the configuration"
                    + f" settings:{supp_run_setts.parameters}, nor by the"
                    + " optional settings:"
                    + f"{supp_run_setts.optional_parameters}"
                )


def verify_has_unique_id(run_config: dict) -> None:
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
