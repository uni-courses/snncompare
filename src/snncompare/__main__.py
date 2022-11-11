"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

# Import code belonging to this project.
import os
import shutil
import sys

from snnbackends.plot_graphs import create_root_dir_if_not_exists

from snncompare.exp_setts.custom_setts.create_custom_setts import (
    create_basic_test_config,
)

from .arg_parser.arg_parser import parse_cli_args
from .arg_parser.perform_args import process_args
from .exp_setts.default_setts.create_default_settings import (
    create_default_graph_json,
)

# Remove results directory if it exists.
if os.path.exists("results"):
    shutil.rmtree("results")
if os.path.exists("latex"):
    shutil.rmtree("latex")
    create_root_dir_if_not_exists("latex/Images/graphs")

create_basic_test_config()
create_default_graph_json()

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()
process_args(args)
sys.exit()

# mdsa = MDSA(list(range(0, 4, 1)))
# mdsa_configs = get_algo_configs(mdsa.__dict__)
# verify_algo_configs("MDSA", mdsa_configs)
