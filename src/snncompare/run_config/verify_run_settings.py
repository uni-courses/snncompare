"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""
# controllers.py
from __future__ import annotations

from typing import TYPE_CHECKING

from snncompare.run_config.Run_config import Run_config

# pylint: disable=R0801


if TYPE_CHECKING:
    from snncompare.run_config.Supported_run_settings import (
        Supported_run_settings,
    )


# pylint: disable=W0613
def verify_run_config(
    *,
    supp_run_setts: Supported_run_settings,
    run_config: Run_config,
) -> Run_config:
    """Verifies the selected experiment configuration settings are valid.

    :param run_config: param has_unique_id:
    :param has_unique_id: param supp_exp_config:
    :param supp_exp_config:
    """
    if not isinstance(run_config, Run_config):
        raise TypeError(
            "Error, the run_config is of type:"
            + f"{type(run_config)}, yet it was expected to be of"
            + " type dict."
        )

    verify_run_config_dict_is_complete(
        supp_run_setts=supp_run_setts, run_config=run_config
    )

    # TODO: verify a single algorithm is evaluated in a single run.

    # TODO: verify radiation setting for single run.

    # TODO: test unique id type
    return run_config


def verify_run_config_dict_is_complete(
    *,
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
            raise KeyError(
                f"Error:{expected_key} is not in the configuration"
                + f" settings:{run_config.__dict__.keys()}"
            )
