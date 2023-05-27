"""Contains helper functions that are used throughout this repository."""
from typing import TYPE_CHECKING, Dict, List

import networkx as nx
import pylab as plt
from typeguard import typechecked

from snncompare.export_plots.plot_graphs import export_plot

if TYPE_CHECKING:
    pass


@typechecked
def plot_alipour(
    *,
    configuration: str,
    seed: int,
    size: int,
    m: int,
    G: nx.DiGraph,
    export: bool = True,
    show: bool = False,
) -> None:
    """

    :param configuration:
    param iteration: The initialisation iteration that is used.
    :param seed: The value of the random seed used for this test.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    :param G: The original graph on which the MDSA algorithm is ran.
    :param export:  (Default value = True)
    :param show:  (Default value = False)

    """
    # pylint: disable=R0913
    # TODO: reduce 8/5 input arguments to at most 5/5.
    the_labels = get_alipour_labels(G=G, configuration=configuration)
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, labels=the_labels, with_labels=True)
    if show:
        plt.show()
    if export:
        export_plot(
            plt,
            f"alipour_{seed}_size{size}_m{m}_iter_combined_"
            + f"{configuration}",
            extensions=["png"],  # TODO: include run_config extensions.
        )

    plt.clf()
    plt.close()


@typechecked
def get_alipour_labels(*, G: nx.DiGraph, configuration: str) -> Dict[str, str]:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param configuration:

    """
    labels = {}
    for node_name in G.nodes:
        if configuration == "0rand_mark":
            labels[node_name] = (
                f'{node_name},R:{G.nodes[node_name]["random_number"]}, M:'
                + f'{G.nodes[node_name]["marks"]}'
            )
        elif configuration == "1weight":
            labels[
                node_name
            ] = f'{node_name}, W:{G.nodes[node_name]["weight"]}'
        else:
            raise NotImplementedError("Unsupported configuration.")
    return labels


@typechecked
def compute_marks_for_m_larger_than_one(
    *,
    input_graph: nx.Graph,
    m: int,
    seed: int,
    size: int,
    rand_ceil: int,
    export: bool = False,
    show: bool = False,
) -> None:
    """Stores the marks in the counter nodes of the graph.."""
    # pylint: disable=R0913
    # TODO: reduce 10/5 arguments to at most 5/5.
    # Don't compute for m=0
    for m_val in range(0, m + 1):
        for node_index in input_graph.nodes:
            if m_val == 0:
                compute_weight_for_m_0(
                    input_graph=input_graph, node_index=node_index
                )
            elif m_val > 0:
                compute_weight_for_m_1_and_up(
                    input_graph=input_graph, node_index=node_index
                )
            else:
                raise ValueError(f"{m_val} is not supported.")
            reset_marks_and_countermarks(
                input_graph=input_graph, node_index=node_index
            )

        add_marks_to_nodes_for_mdsa(
            input_graph=input_graph, rand_ceil=rand_ceil
        )

        if show or export:
            plot_alipour(
                configuration="0rand_mark",
                seed=seed,
                size=size,
                m=m_val,
                G=input_graph,
                show=show,
            )
            plot_alipour(
                configuration="1weight",
                seed=seed,
                size=size,
                m=m_val,
                G=input_graph,
                show=show,
            )
            plot_alipour(
                configuration="2inhib_weight",
                seed=seed,
                size=size,
                m=m_val,
                G=input_graph,
                show=show,
            )


@typechecked
def add_marks_to_nodes_for_mdsa(
    *,
    input_graph: nx.Graph,
    rand_ceil: int,
) -> None:
    """Compute the marks/score for the neighbours of each node."""

    for node in input_graph.nodes:
        # Compute for each node circuit what the highest score in the
        # neighbourhood is. (To determine which neighbour wins.)
        # The weight has been previously computed as: degree+randomness=weight
        # for m=0
        # TODO: Recompute the weight here to be sure it is correct.
        max_weight: int = max(
            input_graph.nodes[neighbour]["weight"]
            for neighbour in nx.all_neighbors(input_graph, node)
        )

        # Counter to ensure there is only 1 winner per neighbourhood.
        nr_of_max_weights: int = 0

        # Compute for each node circuit which neighbour has the highest
        # weight.
        for neighbour in nx.all_neighbors(input_graph, node):
            # If a neighbour has the highest score, give it a point.
            # A point is the rand_ceil+1 for the snn.
            # because the rand nrs are spread from 0 to rand_ceil, and the
            # mark point should exceed that to prevent duplicate values.
            if input_graph.nodes[neighbour]["weight"] == max_weight:
                # Keep mark count for snn.
                input_graph.nodes[neighbour]["marks"] += rand_ceil + 1
                # Keep countermarks for Alipour/neumann.
                input_graph.nodes[neighbour]["countermarks"] += 1

                # Verify there is only one max weight neuron.
                nr_of_max_weights += 1
                if nr_of_max_weights > 1:
                    raise ValueError("Two numbers with identical max weight.")


@typechecked
def compute_weight_for_m_0(*, input_graph: nx.Graph, node_index: int) -> None:
    """Computes the weight for the initial m=0 round, using the degree of the
    node instead of the (counter)marks from the previous round."""
    input_graph.nodes[node_index]["weight"] = (
        # The degree is substitute of the marks, and the marks are for the snn
        # which are multiplied with the nr of nodes to ensure all neighbours
        # have a unique weight.
        input_graph.degree[node_index] * len(input_graph.nodes)
        + input_graph.nodes[node_index]["random_number"]
    )


@typechecked
def compute_weight_for_m_1_and_up(
    *, input_graph: nx.Graph, node_index: int
) -> None:
    """Computes the weight for the rounds with m=>1, using the (counter)marks
    from the previous round, instead of the degrees of the nodes."""
    # Compute the weights for this round of m. The weights are used to
    # determine the winning neighbour of each node.
    input_graph.nodes[node_index]["weight"] = (
        input_graph.nodes[node_index]["marks"]
        + input_graph.nodes[node_index]["random_number"]
    )


@typechecked
def reset_marks_and_countermarks(
    *, input_graph: nx.Graph, node_index: int
) -> None:
    """Reset marks and countermarks, because they need to be recomputed for
    this round, using the weights."""
    input_graph.nodes[node_index]["marks"] = 0
    # Note marks are used in the snn make verifying the neumann vs snn
    # a bit easier. Countermarks are for the default Alipour/Neumann
    # version.
    input_graph.nodes[node_index]["countermarks"] = 0


@typechecked
def set_node_default_values(
    *,
    input_graph: nx.Graph,
    node: int,
    rand_ceil: float,
    uninhibited_spread_rand_nrs: List[float],
) -> None:
    """Initialises the starting values of the node attributes."""
    input_graph.nodes[node]["marks"] = input_graph.degree(node) * (
        rand_ceil + 1
    )
    input_graph.nodes[node]["countermarks"] = 0
    input_graph.nodes[node]["random_number"] = (
        1 * uninhibited_spread_rand_nrs[node]
    )
    input_graph.nodes[node]["weight"] = (
        input_graph.nodes[node]["marks"]
        + input_graph.nodes[node]["random_number"]
    )
