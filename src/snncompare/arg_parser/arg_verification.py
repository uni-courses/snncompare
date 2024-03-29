"""Verifies the given CLI arguments are valid in combination with each
other."""
import argparse
import json

import networkx as nx
from typeguard import typechecked

from ..helper import file_exists


@typechecked
def verify_args(*, args: argparse.Namespace, custom_config_path: str) -> None:
    """Performs the checks to verify the parser."""
    if isinstance(args.graph_filepath, str):
        verify_input_graph_path(graph_path=args.graph_filepath)

    # Verify output extension is passed correctly.

    verify_experiment_settings(
        custom_config_path=custom_config_path,
        exp_config_name=args.experiment_settings_name,
    )


@typechecked
def verify_input_graph_path(*, graph_path: str) -> None:
    """Verifies the filepath for the input graph exists and contains a valid
    networkx graph."""
    # Assert graph file exists.
    if not file_exists(filepath=graph_path):
        raise FileNotFoundError(f"Input Graph path was invalid:{graph_path}")

    # Read output JSON file into dict.
    with open(graph_path, encoding="utf-8") as json_file:
        json_graph = json.load(json_file)
        json_file.close()

    # Convert json_graph back to nx.DiGraph
    nx_graph = nx.node_link_graph(json_graph)

    # Assert graph file contains a valid networkx graph.
    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph)):
        raise TypeError(
            "Error, the loaded graph is not of type nx.DiGraph. "
            + f"Instead, it is of type:{type(nx_graph)}"
        )


@typechecked
def verify_experiment_settings(
    *, custom_config_path: str, exp_config_name: str
) -> None:
    """Verifies the filepath for the input graph exists and contains a valid
    networkx graph."""
    # Assert graph file exists.
    exp_config_path = f"{custom_config_path}/exp_config/{exp_config_name}.json"
    if not file_exists(filepath=exp_config_path):
        raise FileNotFoundError(
            "Input experiment settings path was invalid: "
            + f"{exp_config_path}"
        )

    # Read output JSON file into dict.
    with open(exp_config_path, encoding="utf-8") as json_file:
        json.load(json_file)
        json_file.close()
