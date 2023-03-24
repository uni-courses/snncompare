"""Exports the following structure to an output file for simsnn:

Output in separate folders:
    /exp_config/unique_id.json
    /run_config/run_config_name.json

/stage_1/run_config_name.json with content of stage1 algo dict.
    input_graph: nodes and edges
    snn_algo_graph: nodes, lif values and edges.
    adapted_snn_algo_graph: nodes, lif values and edges.
    radiation type, died neurons list without adaptation.
    radiation type, Died neurons list with adaptation.
"""
import json
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

import jsons
import networkx as nx
from networkx.readwrite import json_graph
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_results.export_json_results import (
    verify_loaded_json_content_is_nx_graph,
    write_to_json,
)
from snncompare.export_results.helper import (
    exp_config_to_filename,
    run_config_to_filename,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def output_stage_1_configs_and_input_graphs(
    *,
    exp_config: Exp_config,
    run_config: Run_config,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> None:
    """Exports results dict to a json file."""
    output_simsnn_stage1_exp_config(exp_config=exp_config)
    output_simsnn_stage1_run_config(run_config=run_config)
    output_input_graph(input_graph=graphs_dict["input_graph"])
    output_mdsa_rand_nrs(
        input_graph=graphs_dict["input_graph"], run_config=run_config
    )


@typechecked
def output_simsnn_stage1_exp_config(
    *,
    exp_config: Exp_config,
) -> None:
    """Exports results dict to a json file."""
    relative_dir: str = "results/exp_configs/"
    create_relative_path(some_path=relative_dir)

    # Convert exp_config to exp_config name.

    exp_config_filename = exp_config_to_filename(
        exp_config_dict=exp_config.__dict__
    )

    # Write exp_config to file.
    write_to_json(
        output_filepath=f"{relative_dir}{exp_config_filename}.json",
        some_dict=jsons.dump(exp_config.__dict__),
    )
    verify_loaded_json_content_is_nx_graph(
        output_filepath=f"{relative_dir}{exp_config_filename}.json",
        some_dict=jsons.dump(exp_config.__dict__),
    )


@typechecked
def output_simsnn_stage1_run_config(*, run_config: Run_config) -> None:
    """Exports Run_config to a json file."""
    relative_dir: str = "results/run_configs/"
    create_relative_path(some_path=relative_dir)

    # Convert exp_config to exp_config name.
    run_config_filename = run_config_to_filename(
        run_config_dict=run_config.__dict__
    )

    # Write exp_config to file.
    write_to_json(
        output_filepath=f"{relative_dir}{run_config_filename}.json",
        some_dict=jsons.dump(run_config.__dict__),
    )
    verify_loaded_json_content_is_nx_graph(
        output_filepath=f"{relative_dir}{run_config_filename}.json",
        some_dict=jsons.dump(run_config.__dict__),
    )


@typechecked
def create_relative_path(*, some_path: str) -> None:
    """Exports Run_config to a json file."""
    absolute_path: str = f"{os.getcwd()}/{some_path}"
    # Create subdirectory in results dir.
    if not os.path.exists(absolute_path):
        # Path(absolute_path).mkdir(parents=True, exist_ok=True)
        os.makedirs(absolute_path)

    if not os.path.exists(absolute_path):
        raise NotADirectoryError(f"{absolute_path} does not exist.")


@typechecked
def output_input_graph(
    *,
    input_graph: nx.Graph,
) -> None:
    """Outputs input graph it is not yet outputted."""
    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=input_graph)

    output_dir: str = f"results/input_graphs/{len(input_graph)}/"
    output_filepath: str = f"{output_dir}{isomorphic_hash}.json"
    if not Path(output_filepath).is_file():
        create_relative_path(some_path=output_dir)

        # Write undirected graph to json file.
        write_undirected_graph_to_json(
            output_filepath=output_filepath, the_graph=input_graph
        )


@typechecked
def json_undirected_graph_into_nx_graph(*, input_graph: Dict) -> nx.graph:
    """Converts undirected graph into nx graph.

    TODO: delete.
    """
    edges: List[Tuple[int, int]] = []
    for edge in input_graph["links"]:
        edges.append((edge["source"], edge["target"]))

    output_graph: nx.Graph = nx.Graph()
    output_graph.add_edges_from(edges)
    return output_graph


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
def get_isomorphic_graph_hash(*, some_graph: nx.Graph) -> str:
    """Returns the hash of the isomorphic graph. Meaning all graphs that have
    the same shape, return the same hash string.

    An isomorphic graph is one that looks the same as another/has the
    same shape as another, (if you are blind to the node numbers).
    """
    isomorphic_hash: str = (
        nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(some_graph)
    )
    return isomorphic_hash


@typechecked
def output_mdsa_rand_nrs(
    *, input_graph: nx.Graph, run_config: Run_config
) -> None:
    """Stores the random numbers chosen for the original MDSA snn algorithm."""
    snn_rand_edge_weights: List[int] = []
    for node_index in input_graph:
        snn_rand_edge_weights.append(
            input_graph.graph["alg_props"]["rand_edge_weights"][node_index]
        )

    output_dir: str = f"results/rand_nrs/{run_config.graph_size}/"
    target_file_exists, output_filepath = prepare_target_file_output(
        output_dir=output_dir, some_graph=input_graph
    )
    if not target_file_exists:
        write_to_json(
            output_filepath=output_filepath,
            some_dict=snn_rand_edge_weights,
        )
    else:
        # Read target_file, check if these random nrs are already in,
        # and append them if not.
        pass


@typechecked
def prepare_target_file_output(
    *, output_dir: str, some_graph: Union[nx.Graph, nx.DiGraph]
) -> Tuple[bool, str]:
    """Creates the relative filepath if it does not exist.

    Returns True if the target file already exists, False otherwise.
    """

    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=some_graph)
    output_filepath: str = f"{output_dir}{isomorphic_hash}.json"

    if not Path(output_filepath).is_file():
        create_relative_path(some_path=output_dir)
    return Path(output_filepath).is_file(), output_filepath
