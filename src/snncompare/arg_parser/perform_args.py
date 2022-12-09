"""Completes the tasks specified in the arg_parser."""

import argparse

from typeguard import typechecked

from snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from snncompare.Experiment_runner import Experiment_runner

from ..exp_setts.custom_setts.run_configs.algo_test import (
    load_experiment_config_from_file,
)


@typechecked
def process_args(args: argparse.Namespace, custom_config_path: str) -> None:
    """Processes the arguments and ensures the accompanying tasks are executed.

    TODO: --graph-filepath
    TODO: --run-config
    TODO: list existing exp_configs
    TODO: list existing exp_configs
    """

    # mdsa_creation_only_size_3_4
    # mdsa_size3_5_m_0_5
    # mdsa_size3_m1
    # mdsa_size3_m0
    # mdsa_size5_m4
    # mdsa_size4_m0
    exp_setts = load_experiment_config_from_file(
        custom_config_path, args.experiment_settings_name
    )

    if args.visualise_snn:
        exp_setts["show_snns"] = True
    else:
        exp_setts["show_snns"] = False

    if args.export_images:
        exp_setts["export_images"] = True
    else:
        exp_setts["export_images"] = False

    # if not args.overwrite_visualisation:
    #    exp_setts["export_images"] = True
    #    exp_setts["overwrite_images"] = True

    verify_experiment_config(
        Supported_experiment_settings(),
        exp_setts,
        has_unique_id=False,
        allow_optional=True,
    )

    # python -m src.snncompare -e mdsa_creation_only_size_3_4 -v
    Experiment_runner(exp_setts)
    # TODO: verify expected output results have been generated successfully.
    print("Done")
