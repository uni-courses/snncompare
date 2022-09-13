"""Entry point for this project, runs the project code based on the cli command
that invokes this script."""


# Import external libraries.
# import networkx as nx

from src.experiment_settings.Experiment_runner import (
    Experiment_runner,
    example_experi_config,
)
from src.graph_generation.get_graph import get_networkx_graph_of_2_neurons
from src.simulation.run_on_lava import (
    add_lava_neurons_to_networkx_graph,
    simulate_snn_on_lava,
)
from src.simulation.run_on_networkx import run_snn_on_networkx

from .arg_parser import parse_cli_args

# Import code from this project.

experi_config = example_experi_config()
show_snns = False
export_snns = True
Experiment_runner(experi_config, show_snns=show_snns, export_snns=export_snns)

# Parse command line interface arguments to determine what this script does.
args = parse_cli_args()

# Get a standard graph for illustratory purposes.
G = get_networkx_graph_of_2_neurons()

if args.run_on_networkx:
    # TODO: verify why this is necessary.
    # append_neurons_to_networkx_graph(G)
    run_snn_on_networkx(G, 2)
elif args.run_on_lava:
    # Convert the networkx specification to lava SNN.
    add_lava_neurons_to_networkx_graph(G, t=0)
    starter_node_name = 0
    simulate_snn_on_lava(G, starter_node_name, 2)

    for node in G.nodes:
        print(f'node={node},u={G.nodes[node]["lava_LIF"].u.get()}')
    # Terminate Loihi simulation.
    G.nodes[starter_node_name]["lava_LIF"].stop()
