"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

# Import code belonging to this project.


from snncompare.arg_parser.arg_verification import verify_args
from snncompare.exp_config.custom_setts.create_custom_setts import (
    create_basic_test_config,
)

from .arg_parser.arg_parser import parse_cli_args
from .arg_parser.process_args import process_args
from .exp_config.default_setts.create_default_settings import (
    create_default_graph_json,
)

custom_config_path = "src/snncompare/exp_config/custom_setts/exp_config/"

create_basic_test_config(custom_config_path)
create_default_graph_json()

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()
verify_args(args, custom_config_path)
process_args(args, custom_config_path)
