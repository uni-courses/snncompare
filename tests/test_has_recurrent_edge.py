# -*- coding: utf-8 -*-
"""Performs tests that verify the behaviour of a recurrent edge in the
network."""

import unittest

from src.get_graph import get_networkx_graph_of_2_neurons
from src.run_on_lava import (
    add_lava_neurons_to_networkx_graph,
    simulate_network_on_lava,
)
from src.run_on_networkx import simulate_snn_on_networkx


class Test_get_graph(unittest.TestCase):
    """Tests whether a graph has a recurrent edge.

    A recurrent edge is an edge that is connected to itself.
    """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_returns_2_nodes(self):
        """Tests whether the get_networkx_graph_of_2_neurons function returns a
        graph with two nodes."""

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()
        self.assertEqual(len(G), 2)

        # Assert the graph has only 1 edge.
        self.assertEqual(len(G.edges), 1)

        # Assert the edge goes from node 0 to node 1.
        self.assertEqual(set(G.edges), {(0, 1)})

        # Add the recurrent edge.
        G.add_edge(0, 0)

        # Assert the graph has 2 edges.
        self.assertEqual(len(G.edges), 2)

        # Assert the edges go from node 0 to node 1, and from 0 to 0.
        self.assertEqual(set(G.edges), {(0, 0), (0, 1)})

    def test_verify_recurrent_edge_without_weight_throws_error(self):
        """Creates an SNN consisting of 2 neurons, and verifies their behaviour
        over time.

        Then does the same again yet with a recurrent connection in the
        first neuron.
        """

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()

        # Verify the graph can run on Networkx.
        simulate_snn_on_networkx(G, 30)

        # Verify the graph can run on Lava.
        simulate_network_on_lava(G, 2)

        # Add the recurrent edge.
        G.add_edge(0, 0)

        with self.assertRaises(Exception) as context:
            # Verify running on Networkx throws error.
            simulate_snn_on_networkx(G, 30)

        self.assertTrue(
            "Not all synapse properties of edge: (0, 0) are specified. It only"
            + " contains attributes:dict_keys([])"
            in str(context.exception)
        )

        with self.assertRaises(Exception) as context:
            # Verify running on Lava throws error.
            simulate_network_on_lava(G, 2)

        self.assertTrue(
            "Not all synapse properties of edge: (0, 0) are specified. It only"
            + " contains attributes:dict_keys([])"
            in str(context.exception)
        )

    def test_verify_recurrent_edge_is_created_in_lava(self):
        """Creates an SNN consisting of 2 neurons, and verifies their behaviour
        over time.

        Then does the same again yet with a recurrent connection in the
        first neuron.
        """

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()

        # Add the recurrent edge.
        G.add_edge(0, 0, weight=-10)

        # Convert the networkx graph to lava graph by adding neurons to graph.
        add_lava_neurons_to_networkx_graph(G)

        # Assert static properties of neuron 0.
        self.assertEqual(G.nodes[0]["neuron"].bias.get(), 2)
        self.assertEqual(G.nodes[0]["neuron"].du.get(), 0.5)
        self.assertEqual(G.nodes[0]["neuron"].dv.get(), 0.5)
        self.assertEqual(G.nodes[0]["neuron"].vth.get(), 2.0)

        # Assert static properties of neuron 1.
        self.assertEqual(G.nodes[1]["neuron"].bias.get(), 0)
        self.assertEqual(G.nodes[1]["neuron"].du.get(), 0)
        self.assertEqual(G.nodes[1]["neuron"].dv.get(), 0)
        self.assertEqual(G.nodes[1]["neuron"].vth.get(), 10)

        # Assert neuron 0 properties at t=0
        self.assertEqual(G.nodes[0]["neuron"].u.get(), 0)
        self.assertEqual(G.nodes[0]["neuron"].v.get(), 0)

        # Assert neuron 1 properties at t=0
        self.assertEqual(G.nodes[1]["neuron"].u.get(), 0)
        self.assertEqual(G.nodes[1]["neuron"].v.get(), 0)

        # Verify the graph can run on Networkx.
        simulate_snn_on_networkx(G, 6)

        # Verify the graph can run on Lava.
        simulate_network_on_lava(G, 6)
