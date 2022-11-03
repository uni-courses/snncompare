"""Contains the list of graphs that are used for radiation testing."""
from typing import List

import networkx as nx
from typeguard import typechecked


class Used_graphs:
    """Creates graphs used for paper."""

    @typechecked
    def __init__(self) -> None:
        self.three = self.get_graphs_with_3_neurons()
        self.four = self.get_graphs_with_4_neurons()
        self.five = self.get_graphs_with_5_neurons()

    @typechecked
    def get_graphs(self, size: int) -> List[nx.Graph]:
        """Returns the graphs that are used for testing, per selected size.

        :param size: Nr of nodes in the original graph on which test is ran.
        """
        if size == 3:
            return self.three
        if size == 4:
            return self.four
        if size == 5:
            return self.five
        raise Exception("Graph size of: {size} is currently not supported.")

    @typechecked
    def get_graphs_with_3_neurons(self) -> List[nx.Graph]:
        """Returns list of graphs of size 3."""
        return [self.three_a()]

    @typechecked
    def get_graphs_with_4_neurons(self) -> List[nx.Graph]:
        """Returns list of graphs of size 4."""
        return [self.four_a(), self.four_b(), self.four_c()]

    @typechecked
    def get_graphs_with_5_neurons(self) -> List[nx.Graph]:
        """Returns list of graphs of size 5."""
        return [
            self.five_a(),
            self.five_b(),
            self.five_c(),
            self.five_d(),
            self.five_e(),
            self.five_f(),
        ]

    @typechecked
    def three_a(self) -> nx.Graph:
        """Creates two different graphs of size 3."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
            ]
        )
        return graph

    @typechecked
    def four_a(self) -> nx.Graph:
        """Straight line."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
            ]
        )
        return graph

    @typechecked
    def four_b(self) -> nx.Graph:
        """Y"""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 2),
                (1, 2),
                (2, 3),
            ]
        )
        return graph

    @typechecked
    def four_c(self) -> nx.Graph:
        """Square."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            ]
        )
        return graph

    @typechecked
    def five_a(self) -> nx.Graph:
        """Straight line."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
            ]
        )
        return graph

    @typechecked
    def five_b(self) -> nx.Graph:
        """Y-long-tail."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 2),
                (1, 2),
                (2, 3),
                (3, 4),
            ]
        )
        return graph

    @typechecked
    def five_c(self) -> nx.Graph:
        """Y with 3 arms."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 2),
                (1, 2),
                (3, 2),
                (4, 2),
            ]
        )
        return graph

    @typechecked
    def five_d(self) -> nx.Graph:
        """Pentagon."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
            ]
        )
        return graph

    @typechecked
    def five_e(self) -> nx.Graph:
        """Square-with-tail."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3, 4],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (3, 4),
            ]
        )
        return graph

    @typechecked
    def five_f(self) -> nx.Graph:
        """Square."""
        graph = nx.Graph()
        graph.add_nodes_from(
            [0, 1, 2, 3],
            color="w",
        )
        graph.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            ]
        )
        return graph


class Run:
    """Object that stores the results of a run of the experiment."""

    # pylint: disable=R0902
    # TODO: reduce 17/5 instance attributes to at most 15/7.

    @typechecked
    def __init__(
        self,
        dead_neuron_names: List[str],
        G: nx.DiGraph,
        get_degree: nx.DiGraph,
        has_adaptation: bool,
        iteration: int,
        neuron_death_probability: float,
        m: int,
        has_passed: bool,
        rand_ceil: int,
        rand_values: List[int],
        selected_alipour_nodes: List[int],
        selected_snn_nodes: List[int],
        sim_time: int,
        size: int,
    ) -> None:
        """Called at end of run."""
        # pylint: disable=R0913
        # TODO: reduce 15/5 arguments to at most 5/5.

        self.iteration = iteration
        self.neuron_death_probability = neuron_death_probability
        self.has_passed = has_passed
        self.selected_alipour_nodes = selected_alipour_nodes
        self.selected_snn_nodes = selected_snn_nodes
        self.rand_ceil = rand_ceil
        self.rand_values = rand_values
        self.m = m
        self.G = G
        self.has_adaptation = has_adaptation
        self.get_degree = get_degree
        self.dead_neuron_names = dead_neuron_names
        self.sim_time = sim_time
        self.size = size
        self.amount_of_neurons = self.get_amount_of_neurons(self.get_degree)
        self.amount_synapses = self.get_amount_synapses(self.get_degree)
        # pylint: disable=E1128
        # TODO: resolve.
        # self.nr_of_spikes = self.get_nr_of_spikes()

    @typechecked
    def get_amount_of_neurons(self, get_degree: nx.DiGraph) -> int:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.

        """
        return len(get_degree)

    @typechecked
    def get_amount_synapses(self, get_degree: nx.DiGraph) -> int:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.

        """
        return len(get_degree.edges)

    @typechecked
    def get_nr_of_spikes(self) -> None:
        """TODO: implement this method to store the number of spikes if it is
        found.

        :param get_degree: Graph with the MDSA SNN approximation solution.

        """
        return None
