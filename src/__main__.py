# Entry point for this project, runs the project code based on the cli command
# that invokes this script.


# Import external libraries.
# import networkx as nx

# Import code from this project.
from .run_on_networkx import simulate_snn_on_networkx
from .get_graph import get_networkx_graph_of_2_neurons
from .arg_parser import parse_cli_args

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()

# Get a standard graph for illustratory purposes.
G = get_networkx_graph_of_2_neurons()

if args.run_on_networkx:
    simulate_snn_on_networkx(G, 30)
