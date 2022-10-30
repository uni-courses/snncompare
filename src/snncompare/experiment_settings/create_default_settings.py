"""Used to generate the default experiment configuration json, run
configuration settings, default input graph and default input graphs."""

import jsons
import networkx as nx

from src.snncompare.arg_parser.arg_verification import verify_input_graph_path
from src.snncompare.export_results.export_json_results import (
    write_dict_to_json,
)
from src.snncompare.export_results.export_nx_graph_to_json import (
    digraph_to_json,
)
from src.snncompare.graph_generation.Used_graphs import Used_graphs


def create_default_graph_json() -> None:
    """Generates a default input graph and exports it to a json file."""
    used_graphs = Used_graphs()
    default_nx_graph: nx.DiGraph = used_graphs.five[0]

    # Convert nx.DiGraph to dict.
    default_json_graph = digraph_to_json(default_nx_graph)

    graphs_json_filepath = (
        "src/snncompare/experiment_settings/default_graph_MDSA.json"
    )
    write_dict_to_json(graphs_json_filepath, jsons.dump(default_json_graph))

    # Verify file exists and that it has a valid content.
    verify_input_graph_path(graphs_json_filepath)

    # Verify file content.
