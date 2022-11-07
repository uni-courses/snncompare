"""Performs tests that verify lava simulation produces the same results as the
networkx simulation."""
import math
import unittest

import networkx as nx
import numpy as np
from typeguard import typechecked

from snncompare.graph_generation.get_graph import gnp_random_connected_graph
from snncompare.simulation.verify_graph_is_networkx_snn import (
    assert_no_duplicate_edges_exist,
    assert_synaptic_edgeweight_type_is_correct,
)
from snncompare.simulation.verify_graph_is_snn import verify_networkx_snn_spec
from tests.exp_setts.unsorted.test_scope import Long_scope_of_tests


class Test_networkx_and_lava_snn_simulation_produce_identical_results(
    unittest.TestCase
):
    """Performs tests that verify lava simulation produces the same results as
    the networkx simulation."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.test_scope = Long_scope_of_tests(export=True, show=False)

    @typechecked
    def test_generates_valid_snn_networks_without_recursive_edges(
        self,
    ) -> None:
        """Tests whether the random_snn_networks that are generated are valid
        snn networks.
        TODO: separate into testing for lava and for nx backend.

        The graph must be connected, every edge must have a weight, no
        duplicate edges in the same direction may exist. All neurons
        must have all neuron properties specified.
        """

        # Get graph without edge to self.
        for size in range(self.test_scope.min_size, self.test_scope.max_size):
            for density in np.arange(
                self.test_scope.min_edge_density,
                self.test_scope.max_edge_density,
                self.test_scope.edge_density_stepsize,
            ):

                recurrent_edge_density = 0
                # Only generate graphs that have at least 1 edge.
                if math.floor(size * density) > 1:
                    G = gnp_random_connected_graph(
                        density,
                        recurrent_edge_density,
                        size,
                        self.test_scope,
                    )

                    # Assert graph is connected.
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
                    verify_networkx_snn_spec(G, t=0, backend="nx")

    @typechecked
    def test_generates_valid_snn_networks_with_recursive_edges(self) -> None:
        """Tests whether the random_snn_networks that are generated are valid
        snn networks.

        Adds random recurrent edges. The graph must be connected, every
        edge must have a weight, no duplicate edges in the same
        direction may exist. All neurons must have all neuron properties
        specified.
        """

        # Get graph without edge to self.
        for size in range(self.test_scope.min_size, self.test_scope.max_size):
            for density in np.arange(
                self.test_scope.min_edge_density,
                self.test_scope.max_edge_density,
                self.test_scope.edge_density_stepsize,
            ):

                recurrent_edge_density = 0
                # Only generate graphs that have at least 1 edge.
                if math.floor(size * density) > 1:
                    G = gnp_random_connected_graph(
                        density,
                        recurrent_edge_density,
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
                    verify_networkx_snn_spec(G, t=0, backend="nx")


# Assert number of edges without recurrent edges.

# Add random recursive edges.

# Assert number of edges with recurrent edges.
