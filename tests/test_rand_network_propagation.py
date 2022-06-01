# -*- coding: utf-8 -*-
"""Performs tests that verify lava simulation produces the same results as the
networkx simulation."""

import math
import unittest

import networkx as nx
import numpy as np

from src.get_graph import gnp_random_connected_graph
from src.run_on_lava import (
    add_lava_neurons_to_networkx_graph,
    simulate_snn_on_lava,
)
from src.Scope_of_tests import Scope_of_tests
from src.verify_graph_is_snn import (
    assert_no_duplicate_edges_exist,
    assert_synaptic_edgeweight_type_is_correct,
    verify_networkx_snn_spec,
)


class Test_networkx_and_lava_snn_simulation_produce_identical_results(
    unittest.TestCase
):
    """Performs tests that verify lava simulation produces the same results as
    the networkx simulation."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_scope = Scope_of_tests()

    def test_random_networks_are_propagated_the_same_on_networkx_and_lava(
        self,
    ):
        """Tests whether the random_snn_networks that are generated yield the
        same behaviour on lava as on networkx."""
        # Get graph without edge to self.
        for size in range(self.test_scope.min_size, self.test_scope.max_size):
            for density in np.arange(
                self.test_scope.min_edge_density,
                self.test_scope.max_edge_density,
                self.test_scope.edge_density_stepsize,
            ):

                # Only generate graphs that have at least 1 edge.
                if math.floor(size * density) > 1:
                    G = gnp_random_connected_graph(
                        density,
                        size,
                        self.test_scope,
                    )

                    # Assert graph is connected.
                    # self.assertTrue(nx.is_connected(G))
                    self.assertFalse(
                        not nx.is_strongly_connected(G)
                        and not nx.is_weakly_connected(G)
                    )

                    # Assert size of graph.
                    self.assertEqual(size, len(G))

                    # Assert number of edges without recurrent edges.
                    print(f"size={size},density={density}")
                    # self.assertGreaterEqual(G.number_of_edges(),math.floor(size*density))
                    # self.assertLessEqual(G.number_of_edges(),math.ceil(size*density))

                    # Assert each edge has a weight.
                    for edge in G.edges:
                        assert_synaptic_edgeweight_type_is_correct(G, edge)

                    # Assert no duplicate edges exist.
                    assert_no_duplicate_edges_exist(G)

                    # Assert all neuron properties are specified.
                    verify_networkx_snn_spec(G)

                    # Generate networkx network.

                    # Generate lava network.
                    add_lava_neurons_to_networkx_graph(G)

                    # Run the simulation on networkx.
                    # simulate_snn_on_networkx(G, 10)

                    # Run the simulation on lava.
                    simulate_snn_on_lava(G, 1)

                    # Verify the simulations produce identical neuron
                    # properties.

                    self.compare_static_snn_properties(G)

                    # Terminate Loihi simulation.
                    G.nodes[0]["lava_LIF"].stop()

    def compare_static_snn_properties(self, G):
        """Performs comparison of static neuron properties at each timestep.

        TODO: call for every timestep.
        """
        for node in G.nodes:
            print(f"In comparison node={node}")
            lava_neuron = G.nodes[0]["lava_LIF"]
            nx_neuron = G.nodes[0]["nx_LIF"]
            print(f"lava_neuron.bias.get()={lava_neuron.bias.get()}")
            print(f"lava_neuron.bias.get()={nx_neuron.bias.get()}")
            # Assert bias is equal.
            self.assertEqual(lava_neuron.bias.get(), nx_neuron.bias.get())
            print(f"lava_neuron.du.get()={nx_neuron.du.get()}")
            print(f"lava_neuron.du.get()={lava_neuron.du.get()}")

            # dicts
            # print(f"lava_neuron.__dict__={lava_neuron.__dict__}")
            # print(f"lava_neuron.__dict__={nx_neuron.__dict__}")

            # Assert du is equal.
            self.assertEqual(lava_neuron.du.get(), nx_neuron.du.get())

            print(f"lava_neuron.dv.get()={lava_neuron.dv.get()}")
            print(f"lava_neuron.dv.get()={nx_neuron.dv.get()}")
            # Assert dv is equal.
            self.assertEqual(lava_neuron.dv.get(), nx_neuron.dv.get())

            # print(f"lava_neuron.name.get()={lava_neuron.name.get()}")
            # print(f"lava_neuron.name.get()={nx_neuron.name.get()}")
            # Assert name is equal.
            # self.assertEqual(lava_neuron.name, nx_neuron.name)

            print(f"lava_neuron.u.get()={lava_neuron.u.get()}")
            print(f"lava_neuron.u.get()={nx_neuron.u.get()}")
            # Assert u is equal.
            self.assertEqual(lava_neuron.u.get(), nx_neuron.u.get())

            print(f"lava_neuron.v.get()={lava_neuron.v.get()}")
            print(f"lava_neuron.v.get()={nx_neuron.v.get()}")
            # Assert v is equal.
            self.assertEqual(lava_neuron.v.get(), nx_neuron.v.get())

            print(f"lava_neuron.vth.get()={lava_neuron.vth.get()}")
            print(f"lava_neuron.vth.get()={nx_neuron.vth.get()}")
            # Assert vth is equal.
            self.assertEqual(lava_neuron.vth.get(), nx_neuron.vth.get())

            # Assert v_reset is equal. (Not yet implemented in Lava.)
            # self.assertEqual(
            #    lava_neuron.v_reset.get(), nx_neuron.v_reset.get()
            # )
