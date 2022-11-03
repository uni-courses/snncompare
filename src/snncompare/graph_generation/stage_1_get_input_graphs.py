"""Gets the input graphs that may be adapted in later stages.

Takes run config of an experiment config as input, and returns a
networkx Graph.
"""
import copy
from typing import Any, Dict, List

import networkx as nx

from src.snncompare.graph_generation.adaptation.redundancy import (
    implement_adaptation_mechanism,
)
from src.snncompare.graph_generation.radiation.Radiation_damage import (
    Radiation_damage,
    verify_radiation_is_applied,
)
from src.snncompare.graph_generation.snn_algo.mdsa_snn_algo import (
    Alipour_properties,
    specify_mdsa_network_properties,
)
from src.snncompare.graph_generation.Used_graphs import Used_graphs
from src.snncompare.helper import add_stage_completion_to_graph


def get_used_graphs(run_config: dict) -> dict:
    """First gets the input graph.

    Then generates a graph with adaptation if it is required. Then
    generates a graph with radiation if it is required. Then returns
    this list of graphs.
    """
    graphs = {}
    graphs["input_graph"] = get_input_graph(run_config)

    # TODO: write test to verify the algorithm yields valid results on default
    # MDSA SNN.
    graphs["snn_algo_graph"] = get_snn_algo_graph(
        graphs["input_graph"], run_config
    )

    # TODO: write test to verify the algorithm yields valid results on default
    # MDSA SNN.
    if has_adaptation(run_config):
        graphs["adapted_snn_graph"] = get_adapted_graph(
            graphs["snn_algo_graph"], run_config
        )

    if has_radiation(run_config):
        graphs["rad_snn_algo_graph"] = get_radiation_graph(
            graphs["snn_algo_graph"], run_config, run_config["seed"]
        )
        graphs["rad_adapted_snn_graph"] = get_radiation_graph(
            graphs["adapted_snn_graph"], run_config, run_config["seed"]
        )

    # Indicate the graphs have completed stage 1.
    for graph in graphs.values():
        add_stage_completion_to_graph(graph, 1)
    return graphs


def get_input_graph(run_config: dict) -> nx.DiGraph:
    """TODO: support retrieving graph sizes larger than size 5.
    TODO: ensure those graphs have valid properties, e.g. triangle-free and
    non-planar."""

    # Get the graph of the right size.
    # TODO: Pass random seed.
    input_graph: nx.DiGraph = get_the_input_graph(run_config)

    # TODO: Verify the graphs are valid (triangle free and planar for MDSA).
    return input_graph


def get_the_input_graph(run_config: dict) -> nx.DiGraph:
    """Returns a specific input graph from the list of input graphs."""
    return get_input_graphs(run_config)[run_config["graph_nr"]]


def get_input_graphs(run_config: dict) -> List[nx.DiGraph]:
    """Removes graphs that are not used, because of a maximum nr of graphs that
    is to be evaluated."""
    used_graphs = Used_graphs()
    input_graphs: List[nx.DiGraph] = used_graphs.get_graphs(
        run_config["graph_size"]
    )
    if len(input_graphs) > run_config["graph_nr"]:
        for input_graph in input_graphs:
            # TODO: set alg_props:
            if "alg_props" not in input_graph.graph.keys():
                input_graph.graph["alg_props"] = Alipour_properties(
                    input_graph, run_config["seed"]
                ).__dict__

            if not isinstance(input_graph, nx.Graph):
                raise Exception(
                    "Error, the input graph is not a networkx graph:"
                    + f"{type(input_graph)}"
                )

        return input_graphs
    raise Exception(
        f"For input_graph of size:{run_config['graph_size']}, I found:"
        + f"{len(input_graphs)} graphs, yet expected graph_nr:"
        + f"{run_config['graph_nr']}. Please lower the max_graphs setting in:"
        + "size_and_max_graphs in the experiment configuration."
    )


def get_snn_algo_graph(
    input_graph: nx.DiGraph, run_config: dict
) -> nx.DiGraph:
    """Takes the input graph and converts it to an snn that solves some
    algorithm when it is being ran.

    This SNN is encoded as a networkx graph.
    """

    # TODO: include check to verify only one algorithm is selected.
    # TODO: verify only one setting value is selected per algorithm setting.
    for algo_name, algo_settings in run_config["algorithm"].items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                snn_algo_graph = specify_mdsa_network_properties(
                    input_graph, algo_settings["m_val"], run_config["seed"]
                )
            else:
                raise Exception("Error, m_val setting is not of type int.")
        else:
            raise Exception(
                "Error, algo_name:{algo_name} is not (yet) supported."
            )
    return snn_algo_graph


def get_adapted_graph(
    snn_algo_graph: nx.DiGraph, run_config: dict
) -> nx.DiGraph:
    """Converts an input graph of stage 1 and applies a form of brain-inspired
    adaptation to it."""

    for adaptation_name, adaptation_setting in run_config[
        "adaptation"
    ].items():

        if adaptation_name is None:
            raise Exception(
                "Error, if no adaptation is selected, this method should not"
                + " be reached."
            )
        if adaptation_name == "redundancy":
            if not isinstance(adaptation_setting, float):
                raise Exception(
                    f"Error, adaptation_setting={adaptation_setting},"
                    + "which is not an int."
                )
            adaptation_graph: nx.DiGraph = get_redundant_graph(
                snn_algo_graph, adaptation_setting
            )
            return adaptation_graph
        raise Exception(
            f"Error, adaptation_name:{adaptation_name} is not" + " supported."
        )


def has_adaptation(run_config: Dict[str, Any]) -> bool:
    """Checks if the adaptation contains a None setting.

    TODO: ensure the adaptation only consists of 1 setting per run.
    TODO: throw an error if the adaptation settings contain multiple
    settings, like "redundancy" and "None" simultaneously.
    """
    for adaptation_name in run_config["adaptation"].keys():
        if adaptation_name is not None:
            return True
    return False


def has_radiation(run_config: Dict[str, Any]) -> bool:
    """Checks if the radiation contains a None setting.

    TODO: ensure the radiation only consists of 1 setting per run.
    TODO: throw an error if the radiation settings contain multiple
    settings, like "redundancy" and "None" simultaneously.
    """
    for radiation_name in run_config["radiation"].keys():
        if radiation_name is not None:
            return True
    return False


def get_redundant_graph(
    snn_algo_graph: nx.DiGraph, red_lev: float
) -> nx.DiGraph:
    """Returns a networkx graph that has a form of adaptation added."""
    if red_lev == 0:
        raise Exception(
            "Redundancy level 0 not supported if adaptation is" + " required."
        )
    if red_lev == 1:
        adaptation_graph = copy.deepcopy(snn_algo_graph)
        # TODO: apply redundancy
        implement_adaptation_mechanism(
            adaptation_graph,
        )
        return adaptation_graph

    raise Exception(
        "Error, redundancy level above 1 is currently not" + " supported."
    )


def get_radiation_graph(
    snn_graph: nx.DiGraph, run_config: dict, seed: int
) -> nx.DiGraph:
    """Makes a deep copy of the incoming graph and applies radiation to it.

    Then returns the graph with the radiation, as well as a list of
    neurons that are dead.
    """

    # TODO: determine on which graphs to apply the adaptation.

    # TODO: Verify incoming graph has valid SNN properties.

    # TODO: Check different radiation simulation times.

    # Apply radiation simulation.

    for radiation_name, radiation_setting in run_config["radiation"].items():

        if radiation_name is None:
            raise Exception(
                "Error, if no radiation is selected, this method should not"
                + " be reached."
            )
        if radiation_name == "neuron_death":
            if not isinstance(radiation_setting, float):
                raise Exception(
                    f"Error, radiation_setting={radiation_setting},"
                    + "which is not an int."
                )

            rad_dam = Radiation_damage(probability=radiation_setting)
            radiation_graph: nx.DiGraph = copy.deepcopy(snn_graph)
            dead_neuron_names = rad_dam.inject_simulated_radiation(
                radiation_graph, rad_dam.neuron_death_probability, seed
            )
            # TODO: verify radiation is injected with V1000
            verify_radiation_is_applied(
                radiation_graph, dead_neuron_names, radiation_name
            )

            return radiation_graph
        raise Exception(
            f"Error, radiation_name:{radiation_name} is not supported."
        )
