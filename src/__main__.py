# Entry point for this project, runs the project code based on the cli command
# that invokes this script.


# Import external libraries.
# import networkx as nx

# Import code from this project.
from .get_graph import get_standard_graph_4_nodes
from .arg_parser import parse_cli_args

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()

# Get a standard graph for illustratory purposes.
G = get_standard_graph_4_nodes()
