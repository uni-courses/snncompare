"""File used to generate graph plots."""

import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import networkx as nx
from networkx.classes.digraph import DiGraph

from tests.exp_setts.unsorted.test_scope import Long_scope_of_tests


def plot_circular_graph(
    density: float,
    G: DiGraph,
    recurrent_edge_density: Union[int, float],
    test_scope: Long_scope_of_tests,
) -> None:
    """Generates a circular plot of a (directed) graph.

    :param density: param G:
    :param seed: The value of the random seed used for this test.
    :param export:  (Default value = True)
    :param show: Default value = True)
    :param G: The original graph on which the MDSA algorithm is ran.
    :param recurrent_edge_density:
    :param test_scope:
    """
    # the_labels = get_alipour_labels(G, configuration=configuration)
    the_labels = get_labels(G, "du")
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, labels=the_labels, with_labels=True)
    if test_scope.export:

        create_target_dir_if_not_exists("Images/", "graphs")
        plt.savefig(
            f"Images/graphs/graph_{test_scope.seed}_size{len(G)}_p{density}"
            + f"_p_recur{recurrent_edge_density}.png",
            dpi=200,
        )
    if test_scope.show:
        plt.show()
    plt.clf()
    plt.close()


def plot_uncoordinated_graph(G: nx.DiGraph, show: bool = True) -> None:
    """Generates a circular plot of a (directed) graph.

    :param density: param G:
    :param seed: The value of the random seed used for this test.
    :param show: Default value = True)
    :param G: The original graph on which the MDSA algorithm is ran.
    :param export:  (Default value = False)
    """
    # TODO: Remove unused method.
    # the_labels = get_alipour_labels(G, configuration=configuration)
    # the_labels =
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, with_labels=True)
    if show:
        plt.show()
    plt.clf()
    plt.close()


def create_target_dir_if_not_exists(path: str, new_dir_name: str) -> None:
    """Creates an output dir for graph plots.

    :param path: param new_dir_name:
    :param new_dir_name:
    """

    create_root_dir_if_not_exists(path)
    if not os.path.exists(f"{path}/{new_dir_name}"):
        os.makedirs(f"{path}/{new_dir_name}")


def create_root_dir_if_not_exists(root_dir_name: str) -> None:
    """

    :param root_dir_name:

    """
    if not os.path.exists(root_dir_name):
        os.makedirs(f"{root_dir_name}")
    if not os.path.exists(root_dir_name):
        raise Exception(f"Error, root_dir_name={root_dir_name} did not exist.")


def get_labels(G: DiGraph, configuration: str) -> Dict[int, str]:
    """Returns the labels for the plot nodes.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param configuration:
    """
    labels = {}
    for node_name in G.nodes:
        if configuration == "du":
            labels[node_name] = f"{node_name}"
            # ] = f'{node_name},R:{G.nodes[node_name]["lava_LIF"].du.get()}'
    return labels
