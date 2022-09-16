"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

# Import code belonging to this project.
from src.experiment_settings.Experiment_runner import (
    Experiment_runner,
    example_experi_config,
)

# Run experiment example run.
# TODO: do this as a consequence of parsing the CLI arguments.
experi_config = example_experi_config()
show_snns = False
export_snns = True
Experiment_runner(experi_config, show_snns=show_snns, export_snns=export_snns)

# Parse command line interface arguments to determine what this script does.
# args = parse_cli_args()
