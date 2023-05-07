"""Helps with exporting input graphs."""
import json
import os
import pickle  # nosec
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import networkx as nx
from networkx.readwrite import json_graph
from typeguard import typechecked

# if TYPE_CHECKING:
from snncompare.import_results.helper import (
    create_relative_path,
    get_isomorphic_graph_hash,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def load_input_graph_based_on_nr(graph_size: int, graph_nr: int) -> nx.Graph:
    """Loads an input graph based on the graph_size and graph_nr.

    Sorts the input graph filenames(=input graph hashes), loads the nth
    input  graph, converts it into an nx object and returns it.
    """

    output_dir: str = f"results/stage1/input_graphs/{graph_size}/"
    input_graph_hashes: List[str] = [
        name
        for name in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, name))
    ]
    output_filepath: str = f"{output_dir}{input_graph_hashes[graph_nr]}"

    # Load graph from file and verify it results in the same graph.
    with open(output_filepath, encoding="utf-8") as json_file:
        some_json_graph = json.load(json_file)
        json_file.close()
    loaded_input_graph = nx.node_link_graph(some_json_graph)
    return loaded_input_graph


@typechecked
def has_outputted_input_graph_for_graph_size_and_nr(
    *, graph_size: int, graph_nr: int
) -> bool:
    """Returns True if this input graph already exists."""
    output_dir: str = f"results/stage1/input_graphs/{graph_size}/"
    nr_of_input_graphs: int = len(
        [
            name
            for name in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, name))
        ]
    )
    return nr_of_input_graphs > graph_nr


@typechecked
def get_input_graph_output_dir(*, input_graph: nx.Graph) -> str:
    """Returns the dir in which the input graph as it will be outputted."""
    output_dir: str = f"results/stage1/input_graphs/{len(input_graph)}/"
    return output_dir


@typechecked
def get_input_graph_output_filepath(*, input_graph: nx.Graph) -> str:
    """Returns the path towards the input graph as it will be outputted."""
    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=input_graph)
    output_dir: str = get_input_graph_output_dir(input_graph=input_graph)
    output_filepath: str = f"{output_dir}{isomorphic_hash}.json"
    return output_filepath


@typechecked
def output_input_graph_if_not_exist(
    *,
    input_graph: nx.Graph,
) -> None:
    """Outputs input graph it is not yet outputted."""
    output_dir: str = get_input_graph_output_dir(input_graph=input_graph)
    output_filepath: str = get_input_graph_output_filepath(
        input_graph=input_graph
    )
    if not Path(output_filepath).is_file():
        create_relative_path(some_path=output_dir)

        # Write undirected graph to json file.
        write_undirected_graph_to_json(
            output_filepath=output_filepath, the_graph=input_graph
        )


@typechecked
def write_undirected_graph_to_json(
    *, output_filepath: str, the_graph: nx.Graph
) -> None:
    """Writes an undirected graph to json and verifies it can be loaded back
    into the graph."""
    with open(output_filepath, "w", encoding="utf-8") as fp:
        # json.dump(the_graph.__dict__, fp, indent=4, sort_keys=True)
        some_json_graph: Dict = json_graph.node_link_data(the_graph)
        json.dump(some_json_graph, fp, indent=4, sort_keys=True)
        fp.close()

    # Verify the file exists.
    if not Path(output_filepath).is_file():
        raise FileExistsError(
            f"Error, filepath:{output_filepath} was not created."
        )

    # Load graph from file and verify it results in the same graph.
    with open(output_filepath, encoding="utf-8") as json_file:
        some_json_graph = json.load(json_file)
        json_file.close()
    loaded_graph = nx.node_link_graph(some_json_graph)

    # loaded_graph: nx.Graph = nx.Graph(**the_dict)
    if not nx.utils.misc.graphs_equal(the_graph, loaded_graph):
        print("Outputting graph:")
        pprint(the_graph.__dict__)
        print("Loaded graph:")
        pprint(loaded_graph.__dict__)
        raise ValueError(
            "Error, the outputted graph is not equal to the loaded graph."
        )


@typechecked
def store_pickle(*, run_configs: List[Run_config], filepath: str) -> None:
    """Stores run_config list into pickle file."""
    with open(filepath, "wb") as handle:
        pickle.dump(run_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)


@typechecked
def load_pickle(*, filepath: str) -> List[Run_config]:
    """Stores run_config list into pickle file."""
    with open(filepath, "rb") as handle:
        run_configs: List[Run_config] = pickle.load(handle)  # nosec
    return run_configs
