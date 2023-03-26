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
import hashlib
import json
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

import jsons
import networkx as nx
from networkx.readwrite import json_graph
from simsnn.core.simulators import Simulator
from snnradiation.Radiation_damage import get_random_neurons
from typeguard import typechecked

# if TYPE_CHECKING:
from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_results.export_json_results import (
    verify_loaded_json_content_is_nx_graph,
    write_to_json,
)
from snncompare.export_results.helper import (
    exp_config_to_filename,
    run_config_to_filename,
)
from snncompare.import_results.helper import (
    create_relative_path,
    get_isomorphic_graph_hash,
    get_radiation_description,
    simsnn_files_exists_and_get_path,
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
    output_simsnn_stage1_exp_config(exp_config=exp_config, stage_index=1)
    output_simsnn_stage1_run_config(run_config=run_config, stage_index=1)
    output_input_graph(
        input_graph=graphs_dict["input_graph"],
        stage_index=1,
    )
    output_mdsa_rand_nrs(
        input_graph=graphs_dict["input_graph"],
        run_config=run_config,
        stage_index=1,
    )
    output_radiation(
        graphs_dict=graphs_dict, run_config=run_config, stage_index=1
    )


@typechecked
def output_simsnn_stage1_exp_config(
    *,
    exp_config: "Exp_config",
    stage_index: int,
) -> None:
    """Exports results dict to a json file."""
    relative_dir: str = f"results/stage{stage_index}/exp_configs/"
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
def output_simsnn_stage1_run_config(
    *, run_config: Run_config, stage_index: int
) -> None:
    """Exports Run_config to a json file."""
    relative_dir: str = f"results/stage{stage_index}/run_configs/"
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
def output_input_graph(
    *,
    input_graph: nx.Graph,
    stage_index: int,
) -> None:
    """Outputs input graph it is not yet outputted."""
    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=input_graph)

    output_dir: str = (
        f"results/stage{stage_index}/input_graphs/{len(input_graph)}/"
    )
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
def output_mdsa_rand_nrs(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
    stage_index: int,
) -> None:
    """Stores the random numbers chosen for the original MDSA snn algorithm."""

    rand_nrs, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)

    rand_nrs_exists, rand_nrs_filepath = simsnn_files_exists_and_get_path(
        output_category="rand_nrs",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=False,
        stage_index=stage_index,
        rand_nrs_hash=rand_nrs_hash,
    )

    output_unique_list(
        output_filepath=rand_nrs_filepath,
        some_list=rand_nrs,
        target_file_exists=rand_nrs_exists,
    )


@typechecked
def get_rand_nrs_and_hash(
    *,
    input_graph: nx.Graph,
) -> Tuple[List[int], str]:
    """Returns the rand nrs and accompanying hash."""
    rand_nrs: List[int] = input_graph.graph["alg_props"]["rand_edge_weights"]
    rand_nrs_hash: str = str(
        hashlib.sha256(json.dumps(rand_nrs).encode("utf-8")).hexdigest()
    )
    return rand_nrs, rand_nrs_hash


@typechecked
def output_radiation(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
    stage_index: int,
) -> None:
    """Stores the random numbers chosen for the original MDSA snn algorithm."""

    radiation_name, radiation_parameter = get_radiation_description(
        run_config=run_config
    )
    # pylint:disable=R0801
    if radiation_name == "neuron_death":
        for with_adaptation in [False, True]:
            if with_adaptation:
                snn_graph = graphs_dict["adapted_snn_graph"]
            else:
                snn_graph = graphs_dict["snn_algo_graph"]

            (
                affected_neurons,
                rad_affected_neurons_hash,
            ) = get_radiation_names_and_hash(
                snn_graph=snn_graph,
                radiation_parameter=radiation_parameter,
                run_config=run_config,
            )
            (
                radiation_file_exists,
                radiation_filepath,
            ) = simsnn_files_exists_and_get_path(
                output_category=f"{radiation_name}_{radiation_parameter}",
                input_graph=graphs_dict["input_graph"],
                run_config=run_config,
                with_adaptation=with_adaptation,
                stage_index=stage_index,
                rad_affected_neurons_hash=rad_affected_neurons_hash,
            )

            output_unique_list(
                output_filepath=radiation_filepath,
                some_list=affected_neurons,
                target_file_exists=radiation_file_exists,
            )
    else:
        raise NotImplementedError(
            f"Error:{radiation_name} is not yet implemented."
        )


@typechecked
def get_radiation_names_and_hash(
    snn_graph: Simulator, radiation_parameter: float, run_config: Run_config
) -> Tuple[List[str], str]:
    """Returns the neuron names that are affected by the radiation, and the
    accompanying hash."""
    simsnn_neuron_names: List[str] = list(
        map(lambda neuron: neuron.name, list(snn_graph.network.nodes))
    )
    affected_neurons: List[str] = get_random_neurons(
        neuron_names=simsnn_neuron_names,
        probability=radiation_parameter,
        seed=run_config.seed,
    )
    rad_affected_neurons_hash: str = str(
        hashlib.sha256(
            json.dumps(affected_neurons).encode("utf-8")
        ).hexdigest()
    )
    return affected_neurons, rad_affected_neurons_hash


@typechecked
def output_unique_list(
    *,
    output_filepath: str,
    some_list: List[Union[int, str]],
    target_file_exists: bool,
) -> None:
    """Stores the random numbers chosen for the original MDSA snn algorithm."""

    if not target_file_exists:
        write_to_json(
            output_filepath=output_filepath,
            some_dict=some_list,
        )
    else:
        raise NotImplementedError("Error, target already exists. Write check.")
