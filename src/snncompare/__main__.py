"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

# Import code belonging to this project.
import os
import shutil

from snnalgorithms.get_alg_configs import get_algo_configs, verify_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from snnbackends.plot_graphs import create_root_dir_if_not_exists

from .arg_parser.arg_parser import parse_cli_args
from .exp_setts.default_setts.create_default_settings import (
    create_default_graph_json,
    default_experiment_config,
)
from .Experiment_runner import Experiment_runner

# Remove results directory if it exists.
if os.path.exists("results"):
    shutil.rmtree("results")
if os.path.exists("latex"):
    shutil.rmtree("latex")
    create_root_dir_if_not_exists("latex/Images/graphs")

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()

mdsa = MDSA(list(range(0, 4, 1)))
mdsa_configs = get_algo_configs(mdsa.__dict__)
verify_algo_configs("MDSA", mdsa_configs)
create_default_graph_json()

# Run experiment example run.
# TODO: do this as a consequence of parsing the CLI arguments.
experiment_config = default_experiment_config()
show_snns = False
export_images = False
Experiment_runner(
    experiment_config, export_images=export_images, show_snns=show_snns
)
# TODO: verify expected output results have been generated successfully.
