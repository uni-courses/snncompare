"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
# controllers.py
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from typeguard import typechecked

from snncompare.exp_config.Exp_config import verify_integer_settings
from snncompare.exp_config.run_config.Run_config import Run_config

# pylint: disable=R0801


if TYPE_CHECKING:
    from snncompare.exp_config.run_config.Supported_run_settings import (
        Supported_run_settings,
    )


# pylint: disable=W0613
def verify_run_config(
    supp_run_setts: Supported_run_settings,
    run_config: Run_config,
    has_unique_id: bool,
    allow_optional: bool,
) -> Run_config:
    """Verifies the selected experiment configuration settings are valid.

    :param run_config: param has_unique_id:
    :param has_unique_id: param supp_exp_config:
    :param supp_exp_config:
    """
    if not isinstance(has_unique_id, bool):
        raise Exception(f"has_unique_id={has_unique_id}, should be a boolean")
    if not isinstance(run_config, Run_config):
        raise Exception(
            "Error, the run_config is of type:"
            + f"{type(run_config)}, yet it was expected to be of"
            + " type dict."
        )

    verify_run_config_dict_is_complete(supp_run_setts, run_config)

    # Verify no unknown configuration settings are presented.
    verify_run_config_dict_contains_only_valid_entries(
        supp_run_setts, run_config, has_unique_id, allow_optional
    )

    # TODO: verify a single algorithm is evaluated in a single run.
    verify_integer_settings(run_config.algorithm["MDSA"]["m_val"])
    # TODO: verify radiation setting for single run.

    # TODO: test unique id type
    return run_config


def verify_run_config_dict_is_complete(
    supp_run_setts: Supported_run_settings,
    run_config: Run_config,
) -> None:
    """Verifies the configuration settings dictionary is complete."""
    for expected_key in supp_run_setts.parameters.keys():
        attr_val = getattr(run_config, expected_key)
        if attr_val is None and expected_key not in [
            "adaptation",
            "radiation",
        ]:
            raise Exception(
                f"Error:{expected_key} is not in the configuration"
                + f" settings:{run_config.__dict__.keys()}"
            )


def verify_run_config_dict_contains_only_valid_entries(
    supp_run_setts: Supported_run_settings,
    run_config: Run_config,
    has_unique_id: bool,
    allow_optional: bool,
) -> None:
    """Verifies the configuration settings dictionary does not contain any
    invalid keys."""
    for actual_key in run_config.__dict__.keys():
        if actual_key not in supp_run_setts.parameters:
            if not allow_optional:
                if not (has_unique_id and actual_key == "unique_id"):
                    if getattr(run_config, actual_key) is not None:
                        raise Exception(
                            f"Error:{actual_key}, with value:"
                            + f"{getattr(run_config,actual_key)} is not "
                            + "supported by the configuration settings:"
                            + f"{supp_run_setts.parameters}"
                        )
            if actual_key not in supp_run_setts.optional_parameters:
                raise Exception(
                    f"Error:{actual_key} is not supported by the configuration"
                    + f" settings:{supp_run_setts.parameters}, nor by the"
                    + " optional settings:"
                    + f"{supp_run_setts.optional_parameters}"
                )


@typechecked
def verify_has_unique_id(
    some_dict: dict,
) -> None:
    """Verifies the config setting has a unique id.

    TODO: eliminate duplicate func naming.
    """
    if not isinstance(some_dict, Dict):
        raise Exception(
            "The configuration settings is not a dictionary,"
            + f"instead it is: of type:{type(some_dict)}."
        )
    if "unique_id" not in some_dict.keys():
        raise Exception(
            "The configuration settings do not contain a unique id even though"
            + f" that was expected. some_dict is:{some_dict}."
        )
