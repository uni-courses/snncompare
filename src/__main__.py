"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

# Import code belonging to this project.
import os
import shutil

from src.experiment_settings.Experiment_runner import (
    Experiment_runner,
    example_experi_config,
)
from src.export_results.plot_graphs import create_root_dir_if_not_exists

# Remove results directory if it exists.
if os.path.exists("results"):
    shutil.rmtree("results")
if os.path.exists("latex"):
    shutil.rmtree("latex")
    create_root_dir_if_not_exists("latex/Images/graphs")

# Run experiment example run.
# TODO: do this as a consequence of parsing the CLI arguments.
experi_config = example_experi_config()
show_snns = False
export_snns = False
Experiment_runner(experi_config, export_snns=export_snns, show_snns=show_snns)
# TODO: verify expected output results have been generated successfully.

# Parse command line interface arguments to determine what this script does.
# args = parse_cli_args()
