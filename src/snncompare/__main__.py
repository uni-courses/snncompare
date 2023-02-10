"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""

from snncompare.arg_parser.arg_verification import verify_args
from snncompare.json_configurations.create_custom_setts import (
    create_basic_test_config,
)

from .arg_parser.arg_parser import parse_cli_args
from .arg_parser.process_args import process_args
from .json_configurations.create_default_settings import (
    create_default_graph_json,
)

custom_config_path = "src/snncompare/json_configurations/exp_config/"

create_basic_test_config(custom_config_path=custom_config_path)
create_default_graph_json()

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()
verify_args(args=args, custom_config_path=custom_config_path)
process_args(args=args, custom_config_path=custom_config_path)
