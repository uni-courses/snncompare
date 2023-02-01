"""Used to create custom experiment and run configuration files."""

from typing import TYPE_CHECKING, Dict

from snncompare.json_configurations.run_configs.algo_test import (
    get_exp_config_mdsa_size5_m4,
    load_exp_config_from_file,
    long_exp_config_for_mdsa_testing,
    store_exp_config_to_file,
)

if TYPE_CHECKING:
    from snncompare.exp_config.Exp_config import Exp_config


def create_basic_test_config(*, custom_config_path: str) -> None:
    """Creates and exports an experiment setup that can be used to quickly test
    the MDSA algorithm."""
    mdsa_creation_only_size_3_4: Exp_config = (
        long_exp_config_for_mdsa_testing()
    )
    store_exp_config_to_file(
        custom_config_path=custom_config_path,
        exp_config=mdsa_creation_only_size_3_4,
        filename="mdsa_creation_only_size_3_4",
    )
    load_exp_config_from_file(
        custom_config_path=custom_config_path,
        filename="mdsa_creation_only_size_3_4",
    )

    mdsa_size5_m4: Dict = get_exp_config_mdsa_size5_m4()
    store_exp_config_to_file(
        custom_config_path=custom_config_path,
        exp_config=mdsa_size5_m4,
        filename="mdsa_size5_m4",
    )
    load_exp_config_from_file(
        custom_config_path=custom_config_path, filename="mdsa_size5_m4"
    )
