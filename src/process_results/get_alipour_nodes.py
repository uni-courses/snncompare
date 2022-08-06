"""Computes which nodes are selected by the MDSA algorithm presented by Alipour
et al."""

import networkx as nx

from src.helper import (
    compute_mark,
    compute_marks_for_m_larger_than_one,
    set_node_default_values,
)


def get_alipour_nodes(
    G: nx.Graph,
    m_val: int,
    rand_props,
):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param m: The amount of approximation iterations used in the MDSA
    approximation.
    :param rand_props:

    """
    delta = rand_props["delta"]
    inhibition = rand_props["inhibition"]
    rand_ceil = rand_props["rand_ceil"]
    # TODO: resolve this naming discrepancy.
    rand_nrs = rand_props["initial_rand_current"]

    # Reverse engineer uninhibited spread rand nrs:
    # TODO: read out from rand_props object.
    uninhibited_spread_rand_nrs = [(x + inhibition) for x in rand_nrs]

    for node in G.nodes:
        set_node_default_values(
            delta, G, inhibition, node, rand_ceil, uninhibited_spread_rand_nrs
        )

    # pylint: disable=R0801
    compute_mark(delta, G, rand_ceil)

    compute_marks_for_m_larger_than_one(
        delta,
        G,
        inhibition,
        None,
        m_val,
        None,
        None,
        rand_ceil,
        export=False,
        show=False,
    )
    counter_marks = {}
    for node_index in G.nodes:
        counter_marks[f"counter_{node_index}_{m_val}"] = G.nodes[node_index][
            "countermarks"
        ]
        print(
            f"node_index:{node_index}, ali-mark:"
            + f'{G.nodes[node_index]["countermarks"]}'
        )
    return counter_marks
