"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""


from snncompare.arg_parser.arg_verification import verify_args

from .arg_parser.arg_parser import parse_cli_args
from .arg_parser.process_args import process_args

custom_config_path = "src/snncompare/json_configurations/"

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()
verify_args(args=args, custom_config_path=custom_config_path)
process_args(args=args, custom_config_path=custom_config_path)
