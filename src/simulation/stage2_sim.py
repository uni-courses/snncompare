"""Simulates the SNN graphs and returns a deep copy of the graph per
timestep."""
from typing import List

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
    snn_graphs: dict,
    run_config: dict,
) -> List[nx.DiGraph]:
    """Simulates the snn graphs and makes a deep copy for each timestep.

    :param snn_graphs: dict:
    :param run_config: dict:
    """
    print(run_config)
    for graph_name, snn_graph in snn_graphs.items():
        if graph_name != "input_graph":

            # TODO: add lava neurons if run config demands lava.
            some_conversion(snn_graph)

            # TODO: compute actual inhibition and mval
            inhibition = 4
            m_val = 1
            sim_time = inhibition * (m_val + 1) + 10
            graphs = run_snn_on_networkx(snn_graph, sim_time)
    return graphs


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
