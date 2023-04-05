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
from typing import Dict, List, Optional, Tuple, Union

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
from snncompare.helper import get_snn_graph_from_graphs_dict
from snncompare.import_results.helper import (
    create_relative_path,
    file_contains_line,
    get_isomorphic_graph_hash,
    get_radiation_description,
    seed_rad_neurons_hash_file_exists,
    seed_rand_nrs_hash_file_exists,
    simsnn_files_exists_and_get_path,
)
from snncompare.run_config.Run_config import Run_config


# pylint: disable=R0902
class Radiation_data:
    """Stores the data used in outputting radiation."""

    # pylint: disable=R0903
    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        affected_neurons: List[str],
        rad_affected_neurons_hash: str,
        radiation_file_exists: bool,
        radiation_filepath: str,
        radiation_name: str,
        radiation_parameter: float,
        seed_hash_file_exists: bool,
        seed_in_seed_hash_file: bool,
        seed_hash_filepath: str,
    ) -> None:
        self.affected_neurons: List[str] = affected_neurons
        self.rad_affected_neurons_hash: str = rad_affected_neurons_hash
        self.radiation_file_exists: bool = radiation_file_exists
        self.radiation_filepath: str = radiation_filepath
        self.radiation_name: str = radiation_name
        self.radiation_parameter: float = radiation_parameter
        self.seed_hash_file_exists: bool = seed_hash_file_exists
        self.seed_in_seed_hash_file: bool = seed_in_seed_hash_file
        self.seed_hash_filepath: str = seed_hash_filepath


class Rand_nrs_data:
    """Stores the data used in outputting random numbers belonging to MDSA
    instance."""

    # pylint: disable=R0903
    # pylint: disable=R0913
    @typechecked
    def __init__(
        self,
        rand_nrs_file_exists: bool,
        rand_nrs_filepath: str,
        rand_nrs: List[int],
        rand_nrs_hash: str,
        seed_hash_file_exists: bool,
        seed_in_seed_hash_file: bool,
        seed_hash_filepath: str,
    ) -> None:
        self.rand_nrs_file_exists: bool = rand_nrs_file_exists
        self.rand_nrs_filepath: str = rand_nrs_filepath
        self.rand_nrs: List[int] = rand_nrs
        self.rand_nrs_hash: str = rand_nrs_hash
        self.seed_hash_file_exists: bool = seed_hash_file_exists
        self.seed_in_seed_hash_file: bool = seed_in_seed_hash_file
        self.seed_hash_filepath: str = seed_hash_filepath


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
    )
    output_mdsa_rand_nrs(
        input_graph=graphs_dict["input_graph"],
        run_config=run_config,
        stage_index=1,
    )

    # Output radiation affected neurons.
    for with_adaptation in [False, True]:
        snn_graph: Union[
            nx.DiGraph, Simulator
        ] = get_snn_graph_from_graphs_dict(
            with_adaptation=with_adaptation,
            with_radiation=False,  # No radiation graph is needed to compute
            # which neurons are affected by radiation.
            graphs_dict=graphs_dict,
        )

        output_radiation_data(
            graphs_dict=graphs_dict,
            run_config=run_config,
            snn_graph=snn_graph,
            with_adaptation=with_adaptation,
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
def output_input_graph(
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
def get_rand_nrs_data(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
    stage_index: int,
) -> Rand_nrs_data:
    """Returns True if the random numbers belonging to some MDSA instance are
    outputted.

    False otherwise.
    """
    rand_nrs, rand_nrs_hash = get_rand_nrs_and_hash(input_graph=input_graph)

    (
        seed_hash_file_exists,
        seed_hash_filepath,
    ) = seed_rand_nrs_hash_file_exists(
        output_category="rand_nrs",
        run_config=run_config,
    )

    if seed_hash_file_exists:
        seed_in_seed_hash_file: bool = file_contains_line(
            filepath=seed_hash_filepath, expected_line=rand_nrs_hash
        )
    else:
        seed_in_seed_hash_file = False

    # pylint:disable=R0801
    rand_nrs_file_exists, rand_nrs_filepath = simsnn_files_exists_and_get_path(
        output_category="rand_nrs",
        input_graph=input_graph,
        run_config=run_config,
        with_adaptation=False,
        stage_index=stage_index,
        rand_nrs_hash=rand_nrs_hash,
    )

    return Rand_nrs_data(
        rand_nrs_file_exists=rand_nrs_file_exists,
        rand_nrs_filepath=rand_nrs_filepath,
        rand_nrs=rand_nrs,
        rand_nrs_hash=rand_nrs_hash,
        seed_hash_file_exists=seed_hash_file_exists,
        seed_hash_filepath=seed_hash_filepath,
        seed_in_seed_hash_file=seed_in_seed_hash_file,
    )


@typechecked
def output_mdsa_rand_nrs(
    *,
    input_graph: nx.Graph,
    run_config: Run_config,
    stage_index: int,
) -> None:
    """Stores the random numbers chosen for the original MDSA snn algorithm."""

    rand_nrs_data: Rand_nrs_data = get_rand_nrs_data(
        input_graph=input_graph,
        run_config=run_config,
        stage_index=stage_index,
    )

    if not rand_nrs_data.seed_hash_file_exists or not file_contains_line(
        filepath=rand_nrs_data.seed_hash_filepath,
        expected_line=rand_nrs_data.rand_nrs_hash,
    ):
        with open(
            rand_nrs_data.seed_hash_filepath, "a", encoding="utf-8"
        ) as txt_file:
            txt_file.write(f"{rand_nrs_data.rand_nrs_hash}\n")
            txt_file.close()

    if not rand_nrs_data.rand_nrs_file_exists:
        output_unique_list(
            output_filepath=rand_nrs_data.rand_nrs_filepath,
            some_list=rand_nrs_data.rand_nrs,
            target_file_exists=rand_nrs_data.rand_nrs_file_exists,
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


# pylint: disable=R0914
@typechecked
def get_rad_name_filepath_and_exists(
    *,
    input_graph: nx.Graph,
    snn_graph: Union[nx.DiGraph, Simulator],
    run_config: Run_config,
    stage_index: int,
    with_adaptation: bool,
    rand_nrs_hash: Optional[str] = None,
) -> Radiation_data:
    """Stores the random numbers chosen for the original MDSA snn algorithm.

    TODO: verify for each call to this function whether rand_nrs_hash should
    be included.
    """

    # Get the type of radiation used in this run_config.
    radiation_name, radiation_parameter = get_radiation_description(
        run_config=run_config
    )

    # pylint:disable=R0801
    if radiation_name == "neuron_death":
        # Get the list of affected neurons and the accompanying hash.
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
            input_graph=input_graph,
            run_config=run_config,
            with_adaptation=with_adaptation,
            stage_index=stage_index,
            rad_affected_neurons_hash=rad_affected_neurons_hash,
            rand_nrs_hash=rand_nrs_hash,
        )

        (
            seed_hash_file_exists,
            seed_hash_filepath,
        ) = seed_rad_neurons_hash_file_exists(
            output_category=f"{radiation_name}_{radiation_parameter}",
            run_config=run_config,
            with_adaptation=with_adaptation,
        )

        if seed_hash_file_exists:
            seed_in_seed_hash_file: bool = file_contains_line(
                filepath=seed_hash_filepath,
                expected_line=rad_affected_neurons_hash,
            )
        else:
            seed_in_seed_hash_file = False

        radiation_data: Radiation_data = Radiation_data(
            affected_neurons=affected_neurons,
            rad_affected_neurons_hash=rad_affected_neurons_hash,
            radiation_file_exists=radiation_file_exists,
            radiation_filepath=radiation_filepath,
            radiation_name=radiation_name,
            radiation_parameter=radiation_parameter,
            seed_hash_filepath=seed_hash_filepath,
            seed_hash_file_exists=seed_hash_file_exists,
            seed_in_seed_hash_file=seed_in_seed_hash_file,
        )
        return radiation_data
    raise NotImplementedError(
        f"Error:{radiation_name} is not yet implemented."
    )


@typechecked
def get_radiation_names_and_hash(
    snn_graph: Simulator, radiation_parameter: float, run_config: Run_config
) -> Tuple[List[str], str]:
    """Returns the neuron names that are affected by the radiation, and the
    accompanying hash."""
    simsnn_neuron_names: List[str] = sorted(
        list(map(lambda neuron: neuron.name, list(snn_graph.network.nodes)))
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


@typechecked
def output_radiation_data(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
    snn_graph: Union[nx.DiGraph, Simulator],
    with_adaptation: bool,
) -> None:
    """Exports json file with radiation data."""
    # Output radiation data
    radiation_data: Radiation_data = get_rad_name_filepath_and_exists(
        input_graph=graphs_dict["input_graph"],
        snn_graph=snn_graph,
        run_config=run_config,
        stage_index=1,
        with_adaptation=with_adaptation,
    )

    if not radiation_data.radiation_file_exists:
        output_unique_list(
            output_filepath=radiation_data.radiation_filepath,
            some_list=radiation_data.affected_neurons,
            target_file_exists=radiation_data.radiation_file_exists,
        )
    if not radiation_data.seed_hash_file_exists or not file_contains_line(
        filepath=radiation_data.seed_hash_filepath,
        expected_line=radiation_data.rad_affected_neurons_hash,
    ):
        with open(
            radiation_data.seed_hash_filepath, "a", encoding="utf-8"
        ) as txt_file:
            txt_file.write(f"{radiation_data.rad_affected_neurons_hash}\n")
            txt_file.close()
