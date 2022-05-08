import networkx as nx


def get_standard_graph_4_nodes() -> nx.DiGraph:
    """Y

    :param ) -> nx.DiGraph(:

    """
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [0, 1, 2, 3],
        color="w",
    )
    graph.add_edges_from(
        [
            (0, 2),
            (1, 2),
            (2, 3),
        ],
        weight=10,
    )
    return graph


def get_networkx_graph_of_2_neurons() -> nx.DiGraph:
    """Y

    :param ) -> nx.DiGraph(:

    """
    graph = nx.DiGraph()
    graph.add_nodes_from(
        [0, 1],
        color="w",
    )
    graph.add_edges_from(
        [
            (0, 1),
        ],
        weight=4,
    )

    # Specify neuron 0 properties.
    graph.nodes[0]["bias"] = 2.0
    graph.nodes[0]["du"] = 0.5
    graph.nodes[0]["dv"] = 0.5
    graph.nodes[0]["vth"] = 2.0

    graph.nodes[1]["bias"] = 0.0
    graph.nodes[1]["du"] = 0.0
    graph.nodes[1]["dv"] = 0.0
    graph.nodes[1]["vth"] = 10.0
    return graph
