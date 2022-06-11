import networkx as nx


class Used_graphs:
    """Creates graphs used for paper."""

    def __init__(self):
        self.three = self.get_graphs_with_3_neurons()
        self.four = self.get_graphs_with_4_neurons()
        self.five = self.get_graphs_with_5_neurons()

    def get_graphs(self, size):
        if size == 3:
            return self.three
        elif size == 4:
            return self.four
        elif size == 5:
            return self.five

    def get_graphs_with_3_neurons(self):
        return [self.three_a()]

    def get_graphs_with_4_neurons(self):
        return [self.four_a(), self.four_b(), self.four_c()]

    def get_graphs_with_5_neurons(self):
        return [
            self.five_a(),
            self.five_b(),
            self.five_c(),
            self.five_d(),
            self.five_e(),
            self.five_f(),
        ]

    def three_a(self):
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

    def four_a(self):
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

    def four_b(self):
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

    def four_c(self):
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

    def five_a(self):
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

    def five_b(self):
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

    def five_c(self):
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

    def five_d(self):
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

    def five_e(self):
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

    def five_f(self):
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
    def __init__(
        self,
        dead_neuron_names,
        G,
        get_degree,
        has_adaptation,
        iteration,
        neuron_death_probability,
        m,
        has_passed,
        rand_ceil,
        rand_values,
        selected_alipour_nodes,
        selected_snn_nodes,
        sim_time,
        size,
    ):
        """Called at end of run."""
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
        self.nr_of_spikes = self.get_nr_of_spikes(self.get_degree)

    def get_amount_of_neurons(self, get_degree):
        return len(get_degree)

    def get_amount_synapses(self, get_degree):
        return len(get_degree.edges)

    def get_nr_of_spikes(self, get_degree):
        return None
