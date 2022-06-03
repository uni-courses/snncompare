"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""


# Import external libraries.
# import networkx as nx

from src.run_on_lava import (
    add_lava_neurons_to_networkx_graph,
    simulate_snn_on_lava,
)

from .arg_parser import parse_cli_args
from .get_graph import get_networkx_graph_of_2_neurons

# Import code from this project.
from .run_on_networkx import (
    add_nx_neurons_to_networkx_graph,
    run_snn_on_networkx,
)

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()

# Get a standard graph for illustratory purposes.
G = get_networkx_graph_of_2_neurons()

if args.run_on_networkx:
    add_nx_neurons_to_networkx_graph(G)
    run_snn_on_networkx(G, 2)
elif args.run_on_lava:
    # Convert the networkx specification to lava SNN.
    add_lava_neurons_to_networkx_graph(G)
    simulate_snn_on_lava(G, 2)

    for node in G.nodes:
        print(f'node={node},u={G.nodes[node]["lava_LIF"].u.get()}')
    # Terminate Loihi simulation.
    G.nodes[0]["lava_LIF"].stop()
