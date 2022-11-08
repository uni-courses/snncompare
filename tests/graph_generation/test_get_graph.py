"""Verifies 2 nodes are included in the networkx graph."""
import unittest

from typeguard import typechecked

from snnalgorithms.get_graph import (
    get_networkx_graph_of_2_neurons,
)


class Test_get_graph(unittest.TestCase):
    """Tests whether the get_networkx_graph_of_2_neurons of the get_graph file
    returns a graph with 2 nodes."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def test_returns_2_nodes(self) -> None:
        """Tests whether the get_networkx_graph_of_2_neurons function returns a
        graph with two nodes."""

        G = get_networkx_graph_of_2_neurons()
        self.assertEqual(len(G), 2)
