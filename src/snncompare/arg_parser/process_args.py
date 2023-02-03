"""Completes the tasks specified in the arg_parser."""
import argparse
import os
import shutil
from typing import List

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists
from snncompare.optional_config.Output_config import (
    Extra_storing_config,
    Output_config,
    Zoom,
)

from ..json_configurations.run_configs.algo_test import (
    load_exp_config_from_file,
)


@typechecked
def process_args(*, args: argparse.Namespace, custom_config_path: str) -> None:
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
    exp_config: Exp_config = load_exp_config_from_file(
        custom_config_path=custom_config_path,
        filename=args.experiment_settings_name,
    )

    output_config: Output_config = manage_export_parsing(args=args)

    # python -m src.snncompare -e mdsa_creation_only_size_3_4 -v
    Experiment_runner(
        exp_config=exp_config,
        output_config=output_config,
        specific_run_config=None,
        perform_run=any(
            x in output_config.output_json_stages for x in [1, 2, 3, 4]
        ),
    )
    # TODO: verify expected output results have been generated successfully.
    print("Done")


# pylint: disable=R0912
@typechecked
def manage_export_parsing(*, args: argparse.Namespace) -> Output_config:
    """Performs the argument parsing related to data export settings."""
    create_root_dir_if_not_exists(root_dir_name="latex/Images/graphs")
    optional_config_args_dict = {}
    extra_storing_config_dict = {}

    if args.delete_images and os.path.exists("latex"):
        shutil.rmtree("latex")

    if args.delete_results and os.path.exists("results"):
        shutil.rmtree("results")

    if args.export_images is not None:
        optional_config_args_dict["export_types"] = args.export_images.split(
            ","
        )
    else:
        optional_config_args_dict["export_types"] = []
    optional_config_args_dict["zoom"] = parse_zoom_arg(args=args)
    optional_config_args_dict["recreate_stages"] = parse_recreate_stages(
        args=args
    )
    optional_config_args_dict["output_json_stages"] = parse_output_json_stages(
        args=args
    )
    extra_storing_config_dict["count_spikes"] = args.count_fires
    extra_storing_config_dict["count_neurons"] = args.count_neurons
    extra_storing_config_dict["count_synapses"] = args.count_synapses
    extra_storing_config_dict["store_died_neurons"] = args.store_died_neurons
    optional_config_args_dict["extra_storing_config"] = Extra_storing_config(
        **extra_storing_config_dict
    )

    output_config: Output_config = Output_config(**optional_config_args_dict)

    return output_config


@typechecked
def parse_zoom_arg(
    *,
    args: argparse.Namespace,
) -> Zoom:
    """Processeses the zoom argument and returns Zoom object."""
    if args.zoom is None:
        zoom = Zoom(
            create_zoomed_image=False,
            left_right=None,
            bottom_top=None,
        )
    else:
        coords = args.zoom.split(",")
        zoom = Zoom(
            create_zoomed_image=True,
            left_right=(coords[0], coords[1]),
            bottom_top=(coords[2], coords[3]),
        )
    return zoom


@typechecked
def parse_recreate_stages(
    *,
    args: argparse.Namespace,
) -> List[int]:
    """Performs the argument parsing related to experiment settings."""
    recreate_stages = []
    if args.recreate_stage_1:
        recreate_stages.append(1)
    if args.recreate_stage_2:
        recreate_stages.append(2)
    if args.recreate_stage_3:
        recreate_stages.append(3)
    if args.recreate_stage_4:
        recreate_stages.append(4)
    if args.recreate_stage_5:
        recreate_stages.append(5)
    return recreate_stages


@typechecked
def parse_output_json_stages(
    *,
    args: argparse.Namespace,
) -> List[int]:
    """Performs the argument parsing related to experiment settings."""
    output_json_stages = []
    if args.output_json_stage_1:
        output_json_stages.append(1)
    if args.output_json_stage_2:
        output_json_stages.append(2)
    # No json outputted at stage 3, only images.
    if args.output_json_stage_4:
        output_json_stages.append(4)
    if args.output_json_stage_5:
        output_json_stages.append(5)
    return output_json_stages
