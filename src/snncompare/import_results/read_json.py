"""Code to load and parse the simulation results dict consisting of the
experiment config, run config and json graphs, from a json dict.

Appears to also be used to partially convert json graphs back into nx
graphs.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Union

import networkx as nx
from networkx.readwrite import json_graph
from snnbackends.networkx.LIF_neuron import Synapse, manually_create_lif_neuron
from snnbackends.verify_nx_graphs import verify_results_nx_graphs
from typeguard import typechecked

from snncompare.exp_config.run_config.Run_config import Run_config


@typechecked
def load_results_from_json(
    *,
    json_filepath: str,
    run_config: Run_config,
) -> Dict:
    """Loads the results from a json file, and then converts the graph dicts
    back into a nx.DiGraph object."""
    # Load the json dictionary of results.
    results_loaded_graphs: Dict = load_json_file_into_dict(
        json_filepath=json_filepath
    )

    # Verify the dict contains a key for the graph dict.
    if "graphs_dict" not in results_loaded_graphs:
        raise Exception(
            "Error, the graphs dict key was not in the stage_1_Dict:"
            + f"{results_loaded_graphs}"
        )

    # Verify the graphs dict is of type dict.
    if results_loaded_graphs["graphs_dict"] == {}:
        raise Exception("Error, the graphs dict was an empty dict.")

    for graph_name in results_loaded_graphs["graphs_dict"].keys():
        results_loaded_graphs["graphs_dict"][
            graph_name
        ] = json_graph.node_link_graph(
            results_loaded_graphs["graphs_dict"][graph_name]
        )
        set_graph_attributes(
            graph=results_loaded_graphs["graphs_dict"][graph_name]
        )

    # TODO: Verify node and edge attributes are of valid object type.
    verify_results_nx_graphs(
        results_nx_graphs=results_loaded_graphs, run_config=run_config
    )
    return results_loaded_graphs


@typechecked
def set_graph_attributes(*, graph: Union[nx.Graph, nx.DiGraph]) -> None:
    """Converts the edge and node attributes Synapse and nx_Lif back into their
    respective objects."""
    for edge in graph.edges:
        if "synapse" in graph.edges[edge].keys():
            graph.edges[edge]["synapse"] = Synapse(
                **graph.edges[edge]["synapse"]
            )
    for node in graph.nodes:
        if "nx_lif" in graph.nodes[node].keys():
            # print(graph.nodes[node]["nx_lif"])
            graph.nodes[node]["nx_lif"] = list(
                map(
                    manually_create_lif_neuron,
                    graph.nodes[node]["nx_lif"],
                )
            )


@typechecked
def load_json_file_into_dict(
    *,
    json_filepath: str,
) -> Dict[str, Optional[Dict]]:
    """TODO: make this into a private function that cannot be called by
    any other object than some results loader.
    Loads a json file into dict from a filepath."""
    if not Path(json_filepath).is_file():
        raise Exception("Error, filepath does not exist:{filepath}")
    # TODO: verify extension.
    # TODO: verify json formatting is valid.
    with open(json_filepath, encoding="utf-8") as json_file:
        the_dict = json.load(json_file)
        json_file.close()
    return the_dict
