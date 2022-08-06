"""Simulates the SNN graphs and returns a deep copy of the graph per
timestep."""

import networkx as nx

from src.export_results.load_pickles_get_results import (
    old_graph_to_new_graph_properties,
)
from src.simulation.run_on_networkx import (
    add_nx_neurons_to_networkx_graph,
    run_snn_on_networkx,
)
from src.simulation.verify_graph_is_snn import (
    assert_no_duplicate_edges_exist,
    assert_synaptic_edgeweight_type_is_correct,
    verify_networkx_snn_spec,
)


def sim_graphs(
    stage_1_graphs: dict,
    run_config: dict,
) -> dict:
    """Simulates the snn graphs and makes a deep copy for each timestep.

    :param stage_1_graphs: dict:
    :param run_config: dict:
    """
    stage_2_graphs = {"input_graph": stage_1_graphs["input_graph"]}
    for graph_name, snn_graph in stage_1_graphs.items():
        if graph_name != "input_graph":

            # TODO: add lava neurons if run config demands lava.
            some_conversion(snn_graph)

            # TODO: compute actual inhibition and mval
            stage_2_graphs[graph_name] = run_snn_on_networkx(
                snn_graph, get_sim_duration(snn_graph, run_config)
            )
    return stage_2_graphs


def get_sim_duration(
    snn_graph: nx.DiGraph,
    run_config: dict,
) -> int:
    """Compute the simulation duration for a given algorithm and graph."""
    for algo_name, algo_settings in run_config["algorithm"].items():
        if algo_name == "MDSA":

            # TODO: determine why +10 is required.
            # TODO: Move into stage_1 get input graphs.
            sim_time = (
                snn_graph.graph["alg_props"]["inhibition"]
                * (algo_settings["m_val"] + 1)
                + 10
            )
            return sim_time
        raise Exception("Error, algo_name:{algo_name} is not (yet) supported.")
    raise Exception("Error, the simulation time was not found.")


def some_conversion(G: nx.DiGraph):
    """Converts the SNN graph specfification to a networkx SNN that can be ran.

    :param G: nx.DiGraph:
    """
    # Assert each edge has a weight.
    for edge in G.edges:
        assert_synaptic_edgeweight_type_is_correct(G, edge)

    # Assert no duplicate edges exist.
    assert_no_duplicate_edges_exist(G)

    # Store node properties in nx LIF neuron object under nx_LIF key of node.
    # get_graph_behaviour(G, sim_time)
    old_graph_to_new_graph_properties(G)

    # Assert all neuron properties are specified.
    verify_networkx_snn_spec(G)

    # Generate networkx network.
    add_nx_neurons_to_networkx_graph(G)

    # TODO: add lava neurons if run config demands lava.
    # Generate lava network.
    # add_lava_neurons_to_networkx_graph(G)

    # Verify the simulations produce identical static
    # neuron properties.
    print("")
    # compare_static_snn_properties(self, G)
