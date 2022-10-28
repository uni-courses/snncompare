"""Performs tests that verify lava simulation produces the same results as the
networkx simulation."""

import math
import unittest

import networkx as nx
import numpy as np

from src.snn_algo_compare.exp_setts.Scope_of_tests import Long_scope_of_tests
from src.snn_algo_compare.graph_generation.get_graph import (
    gnp_random_connected_graph,
)
from src.snn_algo_compare.simulation.LIF_neuron import (
    print_neuron_properties_per_graph,
)
from src.snn_algo_compare.simulation.run_on_lava import (
    add_lava_neurons_to_networkx_graph,
)
from src.snn_algo_compare.simulation.run_on_networkx import (
    add_nx_neurons_to_networkx_graph,
)
from src.snn_algo_compare.simulation.verify_graph_is_networkx_snn import (
    assert_no_duplicate_edges_exist,
    assert_synaptic_edgeweight_type_is_correct,
)
from src.snn_algo_compare.simulation.verify_graph_is_snn import (
    verify_networkx_snn_spec,
)
from tests.simulation.test_cyclic_graph_propagation import (
    compare_static_snn_properties,
)


class Test_propagation_with_recurrent_edges(unittest.TestCase):
    """Performs tests that verify lava simulation produces the same results as
    the networkx simulation."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.test_scope = Scope_of_tests()
        self.test_scope = Long_scope_of_tests(export=True, show=False)

    # pylint: disable=R0801
    def test_random_networks_are_propagated_the_same_on_networkx_and_lava(
        self,
    ):
        """Tests whether the random_snn_networks that are generated yield the
        same behaviour on lava as on networkx."""
        # pylint: disable=R1702
        # This test tests all combinations of properties, hence 6 nested for
        # loops are considered acceptable in this case.
        # Get graph without edge to self.
        for size in range(self.test_scope.min_size, self.test_scope.max_size):
            for density in np.arange(
                self.test_scope.min_edge_density,
                self.test_scope.max_edge_density,
                self.test_scope.edge_density_stepsize,
            ):

                for recurrent_density in np.arange(
                    self.test_scope.min_recurrent_edge_density,
                    self.test_scope.max_recurrent_edge_density,
                    self.test_scope.recurrent_edge_density_stepsize,
                ):
                    # Ensure the simulation works for all starter neurons.
                    # pylint: disable=W0612
                    for starter_neuron in range(size):
                        # Only generate graphs that have at least 1 edge.
                        if math.floor(size * density) > 1:
                            G = gnp_random_connected_graph(
                                density,
                                recurrent_density,
                                size,
                                self.test_scope,
                            )
                            # pylint: disable=R0801
                            # For clarity of what is tested is is considered
                            # better to include the tests here.
                            # Assert graph is connected.
                            # self.assertTrue(nx.is_connected(G))
                            self.assertFalse(
                                not nx.is_strongly_connected(G)
                                and not nx.is_weakly_connected(G)
                            )

                            # Assert size of graph.
                            self.assertEqual(size, len(G))

                            # Assert number of edges without recurrent edges.
                            # print(f"size={size},density={density}")
                            # self.assertGreaterEqual(G.number_of_edges(),math.floor(size*density))
                            # self.assertLessEqual(G.number_of_edges(),math.ceil(size*density))

                            # Assert each edge has a weight.
                            for edge in G.edges:
                                assert_synaptic_edgeweight_type_is_correct(
                                    G, edge
                                )

                            # Assert no duplicate edges exist.
                            assert_no_duplicate_edges_exist(G)

                            # Assert all neuron properties are specified.
                            verify_networkx_snn_spec(G, t=0, backend="nx")

                            # Generate networkx network.
                            add_nx_neurons_to_networkx_graph(G, t=0)

                            # Generate lava network.
                            add_lava_neurons_to_networkx_graph(G, t=0)
                            verify_networkx_snn_spec(G, t=0, backend="lava")

                            # Verify the simulations produce identical static
                            # neuron properties.
                            print("")
                            compare_static_snn_properties(self, G)

                            print_neuron_properties_per_graph(G, True, t=0)

                            # TODO: determine why you can't make a deep copy
                            # of this graph. Probably because it runs Lava
                            # processes that cannot be copied.
                            # run_simulation_for_t_steps(
                            #    self, G, starter_neuron, sim_duration=20
                            # )

    def compare_dynamic_snn_properties(self, G, t):
        """Performs comparison of static neuron properties at each timestep.

        :param G: The original graph on which the MDSA algorithm is ran.
        """
        for node in G.nodes:
            lava_neuron = G.nodes[node]["lava_LIF"]
            nx_neuron = G.nodes[node]["nx_LIF"][t]

            # Assert u is equal.
            self.assertEqual(lava_neuron.u.get(), nx_neuron.u.get())

            # Assert v is equal.
            self.assertEqual(lava_neuron.v.get(), nx_neuron.v.get())
