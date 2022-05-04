import networkx as nx


def get_standard_graph_4_nodes() -> nx.Graph():
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
