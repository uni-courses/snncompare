"""Used to create custom experiment and run configuration files."""

from snncompare.exp_setts.custom_setts.run_configs.algo_test import (
    experiment_config_for_mdsa_testing,
    load_experiment_config_from_file,
    store_experiment_config_to_file,
)


def create_basic_test_config() -> None:
    """Creates and exports an experiment setup that can be used to quickly test
    the MDSA algorithm."""
    mdsa_creation_only_size_3_4: dict = experiment_config_for_mdsa_testing()
    store_experiment_config_to_file(
        mdsa_creation_only_size_3_4, "mdsa_creation_only_size_3_4"
    )
    load_experiment_config_from_file("mdsa_creation_only_size_3_4")
