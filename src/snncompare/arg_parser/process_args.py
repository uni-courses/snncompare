"""Completes the tasks specified in the arg_parser."""
import argparse
import os
import shutil
import sys

from snnbackends.plot_graphs import create_root_dir_if_not_exists
from typeguard import typechecked

from snncompare.exp_config.Exp_config import (
    Exp_config,
    Supported_experiment_settings,
)
from snncompare.Experiment_runner import Experiment_runner
from snncompare.export_results.analysis.create_performance_plots import (
    create_performance_plots,
)

from ..exp_config.custom_setts.run_configs.algo_test import (
    load_exp_config_from_file,
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
    exp_config: Exp_config = load_exp_config_from_file(
        custom_config_path, args.experiment_settings_name
    )

    manage_export_parsing(args, exp_config)
    manage_exp_config_parsing(args, exp_config)

    # if not args.overwrite_images_only:
    #    exp_config.export_images = True
    #    exp_config.overwrite_images = True

    # verify_exp_config(
    #    Supported_experiment_settings(),
    #    exp_config,
    #    has_unique_id=False,
    #    allow_optional=True,
    # )

    # python -m src.snncompare -e mdsa_creation_only_size_3_4 -v
    Experiment_runner(exp_config)
    # TODO: verify expected output results have been generated successfully.
    print("Done")


# pylint: disable=R0912
@typechecked
def manage_export_parsing(
    args: argparse.Namespace, exp_config: Exp_config
) -> None:
    """Performs the argument parsing related to data export settings."""
    create_root_dir_if_not_exists("latex/Images/graphs")
    supp_setts = Supported_experiment_settings()

    if args.delete_images and os.path.exists("latex"):
        shutil.rmtree("latex")

    if args.delete_results and os.path.exists("results"):
        shutil.rmtree("results")

    # Don't export if it is not wanted.
    if args.export_images is None:
        exp_config.export_images = False
        if args.overwrite_visualisation:
            raise ValueError(
                "Overwrite images is not allowed without export_images."
            )
    # Allow user to specify image export types (and verify them).
    else:
        if args.overwrite_visualisation:
            exp_config.overwrite_images_only = args.overwrite_visualisation
        else:
            exp_config.overwrite_images_only = False
        extensions = args.export_images.split(",")
        for extension in extensions:
            if extension in supp_setts.export_types:
                print(f"extensions={extensions}")
            elif extension == "gif":
                exp_config.gif = True
                extensions.remove("gif")
            elif extension == "zoom":
                exp_config.zoom = True
                extensions.remove("zoom")
            else:
                raise Exception(
                    f"Error, image output extension:{extension} is"
                    " not supported."
                )
        exp_config.export_images = True
        exp_config.export_types = extensions

    if args.create_boxplots:
        create_performance_plots(exp_config)
        print("Created boxplots.")
        sys.exit()


@typechecked
def manage_exp_config_parsing(
    args: argparse.Namespace, exp_config: Exp_config
) -> None:
    """Performs the argument parsing related to experiment settings."""
    # Process the graph_size argument.
    if args.graph_size is not None:
        if not isinstance(args.graph_size, int):
            raise TypeError("args.graphs_size should be int.")
        # Assume only one iteration is used if graph size is specified.
        exp_config.size_and_max_graphs = [(args.graph_size, 1)]

    # Process the m_val argument.
    if args.m_val is not None:
        if not isinstance(args.m_val, int):
            raise TypeError("args.m_val should be int.")
        # Assume only one iteration is used if graph size is specified.
        exp_config.algorithms["MDSA"] = [{"m_val": args.m_val}]

    # Process the m_val argument.
    if args.redundancy is not None:
        if not isinstance(args.redundancy, int):
            raise TypeError("args.redundancy should be int.")
        # Assume only one iteration is used if graph size is specified.
        exp_config.adaptations = {"redundancy": [args.redundancy]}

    if args.recreate_stage_1:
        exp_config.recreate_s1 = True
    if args.recreate_stage_2:
        exp_config.recreate_s2 = True
    if args.recreate_stage_4:
        exp_config.recreate_s4 = True
