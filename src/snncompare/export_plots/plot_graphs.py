"""File used to generate graph plots."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import networkx as nx
from simplt.export_plot import create_target_dir_if_not_exists
from typeguard import typechecked

if TYPE_CHECKING:
    from snncompare.tests.test_scope import Long_scope_of_tests


@typechecked
def plot_circular_graph(
    *,
    density: float,
    G: nx.DiGraph,
    recurrent_edge_density: int | float,
    test_scope: Long_scope_of_tests,
) -> None:
    """Generates a circular plot of a (directed) graph.

    :param density: param G:
    :param seed: The value of the random seed used for this test.
    :param export: (Default value = True)
    :param show: Default value = True)
    :param G: The original graph on which the MDSA algorithm is ran.
    :param recurrent_edge_density:
    :param test_scope:
    """
    # the_labels = get_alipour_labels(G, configuration=configuration)
    the_labels = get_labels(G=G, configuration="du")
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, labels=the_labels, with_labels=True)
    if test_scope.export:
        create_target_dir_if_not_exists(some_path="latex/Images/graphs")
        plt.savefig(
            f"latex/Images/graphs/graph_{test_scope.seed}_size{len(G)}_"
            + f"p{density}_p_recur{recurrent_edge_density}.png",
            dpi=200,
        )
    if test_scope.show:
        plt.show()
    plt.clf()
    plt.close()


@typechecked
def create_root_dir_if_not_exists(*, root_dir_name: str) -> None:
    """:param root_dir_name:"""
    if not os.path.exists(root_dir_name):
        os.makedirs(f"{root_dir_name}")
    if not os.path.exists(root_dir_name):
        raise FileNotFoundError(
            f"Error, root_dir_name={root_dir_name} did not exist."
        )


@typechecked
def get_labels(*, G: nx.DiGraph, configuration: str) -> dict[int, str]:
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


@typechecked
def export_plot(  # type:ignore[misc]
    # some_plt: matplotlib.pyplot, # TODO: why is this not allowed.
    some_plt: Any,
    filename: str,
    extensions: list[str],
) -> None:
    """:param plt:

    :param filename:
    """
    create_target_dir_if_not_exists(some_path="latex/Images/graphs")
    for extension in extensions:
        some_plt.savefig(
            "latex/Images/graphs/" + filename + f".{extension}",
            dpi=200,
        )
