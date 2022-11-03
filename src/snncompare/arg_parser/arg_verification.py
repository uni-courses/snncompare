"""Verifies the given CLI arguments are valid in combination with each
other."""

import json
from typing import Any

import networkx as nx

from src.snncompare.helper import file_exists


def verify_args(args: Any) -> None:
    """Performs the checks to verify the parser."""
    verify_input_graph_path(args.graph_filepath)
    verify_experiment_settings(args.experiment_settings_path)


def verify_input_graph_path(graph_path: str) -> None:
    """Verifies the filepath for the input graph exists and contains a valid
    networkx graph."""
    # Assert graph file exists.
    if not file_exists(graph_path):
        raise FileNotFoundError("Input Graph path was invalid:{graph_path}")

    # Read output JSON file into dict.
    with open(graph_path, encoding="utf-8") as json_file:
        json_graph = json.load(json_file)

    # Convert json_graph back to nx.DiGraph
    nx_graph = nx.node_link_graph(json_graph)

    # Assert graph file contains a valid networkx graph.
    if not isinstance(nx_graph, (nx.Graph, nx.DiGraph)):
        raise Exception(
            "Error, the loaded graph is not of type nx.DiGraph. "
            + f"Instead, it is of type:{type(nx_graph)}"
        )


def verify_experiment_settings(exp_setts_path: str) -> None:
    """Verifies the filepath for the input graph exists and contains a valid
    networkx graph."""
    # Assert graph file exists.
    if not file_exists(exp_setts_path):
        raise FileNotFoundError(
            "Input experiment settings path was invalid: "
            + f"{exp_setts_path}"
        )

    # Read output JSON file into dict.
    with open(exp_setts_path, encoding="utf-8") as json_file:
        json.load(json_file)
