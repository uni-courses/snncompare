"""Used to create custom experiment and run configuration files."""

from typing import Dict

from snncompare.exp_config.custom_setts.run_configs.algo_test import (
    get_exp_config_mdsa_size5_m4,
    load_experiment_config_from_file,
    long_exp_config_for_mdsa_testing,
    store_experiment_config_to_file,
)


def create_basic_test_config(custom_config_path: str) -> None:
    """Creates and exports an experiment setup that can be used to quickly test
    the MDSA algorithm."""
    mdsa_creation_only_size_3_4: Dict = long_exp_config_for_mdsa_testing()
    store_experiment_config_to_file(
        custom_config_path,
        mdsa_creation_only_size_3_4,
        "mdsa_creation_only_size_3_4",
    )
    load_experiment_config_from_file(
        custom_config_path, "mdsa_creation_only_size_3_4"
    )

    mdsa_size5_m4: Dict = get_exp_config_mdsa_size5_m4()
    store_experiment_config_to_file(
        custom_config_path, mdsa_size5_m4, "mdsa_size5_m4"
    )
    load_experiment_config_from_file(custom_config_path, "mdsa_size5_m4")
