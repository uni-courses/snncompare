"""Completes the tasks specified in the arg_parser."""

import argparse

from typeguard import typechecked

from ..exp_setts.custom_setts.run_configs.algo_test import (
    experiment_config_for_mdsa_testing,
    load_experiment_config_from_file,
    store_experiment_config_to_file,
)


@typechecked
def process_args(args: argparse.Namespace) -> None:
    """Processes the arguments and ensures the accompanying tasks are
    executed."""
    print(f"args={args}")
    mdsa_creation_only_size_3_4: dict = experiment_config_for_mdsa_testing()

    store_experiment_config_to_file(
        mdsa_creation_only_size_3_4, "mdsa_creation_only_size_3_4"
    )
    load_experiment_config_from_file("mdsa_creation_only_size_3_4")
    print("Done")
