"""Used to generate the default experiment configuration json, run
configuration settings, default input graph and default input graphs."""

import jsons
import networkx as nx
from snnalgorithms.Used_graphs import Used_graphs
from typeguard import typechecked

from ..arg_parser.arg_verification import verify_input_graph_path
from ..export_results.export_json_results import (
    verify_loaded_json_content_is_nx_graph,
    write_to_json,
)
from ..export_results.export_nx_graph_to_json import digraph_to_json


@typechecked
def create_default_graph_json() -> None:
    """Generates a default input graph and exports it to a json file."""
    used_graphs = Used_graphs()
    default_nx_graph: nx.Graph = used_graphs.five[0]

    # Convert nx.DiGraph to dict.
    default_json_graph = digraph_to_json(G=default_nx_graph)

    graphs_json_filepath = (
        "src/snncompare/json_configurations/default_graph_MDSA.json"
    )
    write_to_json(
        output_filepath=graphs_json_filepath,
        some_dict=jsons.dump(default_json_graph),
    )
    verify_loaded_json_content_is_nx_graph(
        output_filepath=graphs_json_filepath,
        some_dict=jsons.dump(default_json_graph),
    )

    # Verify file exists and that it has a valid content.
    verify_input_graph_path(graph_path=graphs_json_filepath)

    # Verify file content.
