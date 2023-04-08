"""Completes the tasks specified in the arg_parser."""
import argparse
import os
import shutil
from typing import List, Union

from typeguard import typechecked

from snncompare.arg_parser.helper import convert_csv_list_arg_to_list
from snncompare.exp_config.Exp_config import Exp_config
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_plots.plot_graphs import create_root_dir_if_not_exists
from snncompare.helper import get_snn_graph_names
from snncompare.optional_config.Output_config import (
    Extra_storing_config,
    Output_config,
    Zoom,
)
from snncompare.run_config.Run_config import Run_config

from ..json_configurations.algo_test import (
    load_exp_config_from_file,
    load_run_config_from_file,
)


@typechecked
def process_args(*, args: argparse.Namespace, custom_config_path: str) -> None:
    """Processes the arguments and ensures the accompanying tasks are executed.

    TODO: --graph-filepath
    TODO: --run-config
    TODO: list existing exp_configs
    TODO: list existing exp_configs
    """
    # if args.experiment_settings_name is not None:
    exp_config: Exp_config = load_exp_config_from_file(
        custom_config_path=custom_config_path,
        filename=args.experiment_settings_name,
    )

    if args.run_config_path is not None:
        specific_run_config: Union[
            None, Run_config
        ] = load_run_config_from_file(
            custom_config_path=custom_config_path,
            filename=f"{args.run_config_path}",
        )
    else:
        specific_run_config = None

    output_config: Output_config = manage_export_parsing(args=args)

    # python -m src.snncompare -e mdsa_creation_only_size_3_4 -v
    Experiment_runner(
        exp_config=exp_config,
        output_config=output_config,
        perform_run=any(
            x in output_config.output_json_stages for x in [1, 2, 3, 4]
        ),
        reverse=args.reverse,
        specific_run_config=specific_run_config,
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

    # To show figures, they need to be (created), (and hence) exported.
    if args.show_images:
        if args.export_images is None:
            args.export_images = ["svg"]
        else:
            if "svg" not in args.export_images:
                args.export_images.append("svg")

    optional_config_args_dict["export_types"] = convert_csv_list_arg_to_list(
        arg_name="export_images", arg_val=args.export_images
    )

    # Ensure user only specifies graph types to show if show_images isn't None.
    if args.show_graph_type and not args.show_images:
        raise SyntaxError(
            "Specified a graph type without asking the graphs to be displayed."
            + f"{args.show_graph_type}"
        )
    # Ensure user only specifies graph types to show if show_images isn't None.
    if args.show_images and (
        args.show_graph_type is None or len(args.show_graph_type) == 1
    ):
        raise SyntaxError(
            "show-images is True, yet no graph type (to show) is specified."
        )

    optional_config_args_dict["graph_types"] = convert_csv_list_arg_to_list(
        arg_name="show_graph_type", arg_val=args.show_graph_type
    )

    if any(
        show_graph_name not in get_snn_graph_names()
        for show_graph_name in optional_config_args_dict["graph_types"]
    ):
        raise ValueError(
            "show_graph_type values:"
            + f'{optional_config_args_dict["show_graph_type"]}'
            + " not in supported"
            + f"graph_names:{get_snn_graph_names()}"
        )
    optional_config_args_dict["dash_port"] = args.dash_port
    if args.dash_port and args.dash_port < 8000:
        raise ValueError(
            "Error, port nr should be >8000. Not necessarily over 9000."
        )

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
    extra_storing_config_dict["skip_stage_2_output"] = args.skip_stage_2_output
    extra_storing_config_dict["show_images"] = args.show_images
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
