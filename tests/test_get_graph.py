"""Verifies 2 nodes are included in the networkx graph."""
import pickle
import unittest

from src.get_graph import get_networkx_graph_of_2_neurons


class Test_get_graph(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_returns_2_nodes(self):
        """Tests whether the get_networkx_graph_of_2_neurons function returns a
        graph with two nodes."""

        G = get_networkx_graph_of_2_neurons()
        self.assertEqual(len(G), 2)
