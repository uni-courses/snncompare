"""Takes an input graph and generates an SNN that solves the MDSA algorithm by
Alipour et al."""
from typing import List

import networkx as nx

from src.experiment_settings.create_testobject import Alipour_properties
from src.graph_generation.helper_network_structure import (
    create_synapses_and_spike_dicts,
)
from src.helper import get_y_position


def specify_mdsa_network_properties(
    input_graph: nx.DiGraph, m_val: int, seed: int
) -> nx.DiGraph:
    """Takes an input graph and generates an SNN that runs the MDSA algorithm
    by Alipour."""
    rand_props = Alipour_properties(input_graph, seed)

    # TODO: Rename all rand_nrs usages.
    rand_nrs = rand_props.initial_rand_current

    # TODO: Rename all rand_nrs usages.
    rand_ceil = rand_props.rand_ceil
    # delta = rand_props.delta

    # Convert the fully connected graph into a networkx graph that
    # stores the snn properties.
    # rand_ceil+1 because the maximum random number is rand_ceil which
    # should map to range 0<rand<1 when divided by the synaptic weight of
    # spike_once neurons. (and not to range 0<rand<=1 as it would without
    # the +1).
    mdsa_snn_graph = input_graph_to_mdsa_snn_graph(
        input_graph, rand_nrs, rand_ceil, m_val
    )

    # Specify the node x,y coordinate spacing distance for visualisation.
    mdsa_snn_graph.d = 0.25 * (m_val + 1)

    return mdsa_snn_graph


def input_graph_to_mdsa_snn_graph(
    input_graph: nx.DiGraph, rand_nrs: List[int], rand_ceil: int, m_val: int
) -> nx.DiGraph:
    """Returns a networkx graph that represents the snn that computes the
    spiking degree in the degree_receiver neurons. One node in the graph
    represents one neuron. A directional edge in the graph represents a synapse
    between two neurons.

    One spike once neuron is created per node in graph input_graph.
    One degree_receiver neuron is created per node in graph input_graph.
    A synapse is created from each spike_once neuron that represents node A
    to each of the degree_receiver that represents a neighbour of node A.

    :param input_graph: The original graph on which the MDSA algorithm is ran.
    :param rand_nrs: List of random numbers that are used.
    :param rand_ceil: Ceiling of the range in which rand nrs can be generated.
    :param m: The amount of approximation iterations used in the MDSA
     approximation.
    """
    # pylint: disable=R0914
    # 16/15 local variables is temporarily used here as there are quite a few
    # variables used in different combinations in various add_node() calls.
    # pylint: disable=R0912
    # TODO: reduce nr of branches.
    shifted_m = m_val + 1
    d = 0.25 * shifted_m  # specify grid distance size
    # Specify edge weight for recurrent inhibitory synapse.
    inhib_recur_weight = -10
    mdsa_snn_graph = nx.DiGraph()
    # Define list of m mappings for sets of tuples containing synapses
    left: List = [{} for _ in range(shifted_m)]
    right: List = [{} for _ in range(shifted_m)]

    # Create a node to make the graph connected. (Otherwise, recurrent snn
    # builder can not span/cross the network.)
    mdsa_snn_graph.add_node(
        "connecting_node",
        id=len(input_graph.nodes),
        du=0,
        dv=0,
        bias=0,
        vth=1,
        pos=(float(-d), float(d)),
    )

    # First create all the nodes in the mdsa_snn_graph graph.
    for node in input_graph.nodes:

        # One neuron per node named: spike_once_0-n
        mdsa_snn_graph.add_node(
            f"spike_once_{node}",
            id=node,
            du=0,
            dv=0,
            bias=2,
            vth=1,
            pos=(float(0), float(node * 4 * d)),
            recur=inhib_recur_weight,
        )

        for neighbour in nx.all_neighbors(input_graph, node):
            if node != neighbour:
                for loop in range(0, shifted_m):
                    mdsa_snn_graph.add_node(
                        f"degree_receiver_{node}_{neighbour}_{loop}",
                        id=node,
                        du=0,
                        dv=1,
                        bias=0,
                        vth=1,
                        pos=(
                            float(4 * d + loop * 9 * d),
                            get_y_position(input_graph, node, neighbour, d),
                        ),
                        recur=inhib_recur_weight,
                    )

        # One neuron per node named: rand
        if len(rand_nrs) < len(input_graph):
            raise Exception(
                "The range of random numbers does not allow for randomness"
                + " collision prevention."
            )

        for loop in range(0, shifted_m):
            mdsa_snn_graph.add_node(
                f"rand_{node}_{loop}",
                id=node,
                du=0,
                dv=0,
                bias=2,
                vth=1,
                pos=(float(d + loop * 9 * d), float(node * 4 * d) + d),
                recur=inhib_recur_weight,
            )

        # Add winner selector node
        for loop in range(0, shifted_m):
            if loop == 0:
                mdsa_snn_graph.add_node(
                    f"selector_{node}_{loop}",
                    id=node,
                    du=0,
                    dv=1,
                    bias=5,
                    vth=4,
                    pos=(float(7 * d + loop * 9 * d), float(node * 4 * d + d)),
                )
            elif loop > 0:
                mdsa_snn_graph.add_node(
                    f"selector_{node}_{loop}",
                    id=node,
                    du=0,
                    dv=1,
                    bias=4,
                    vth=4,
                    pos=(float(7 * d + loop * 9 * d), float(node * 4 * d + d)),
                )

        # Add winner selector node
        # for loop in range(0, m):
        mdsa_snn_graph.add_node(
            f"counter_{node}_{shifted_m-1}",
            id=node,
            du=0,
            dv=1,
            bias=0,
            vth=0,
            pos=(float(9 * d + loop * 9 * d), float(node * 4 * d)),
        )

        # Create next round connector neurons.
        for loop in range(1, shifted_m):
            mdsa_snn_graph.add_node(
                f"next_round_{loop}",
                id=node,
                du=0,
                dv=1,
                bias=0,
                vth=len(input_graph.nodes) - 1,
                pos=(float(6 * d + (loop - 1) * 9 * d), -2 * d),
            )

            mdsa_snn_graph.add_node(
                f"d_charger_{loop}",
                id=node,
                du=0,
                dv=1,
                bias=0,
                vth=0,
                pos=(float(9 * d + (loop - 1) * 9 * d), -2 * d),
            )

            mdsa_snn_graph.add_node(
                f"delay_{loop}",
                id=node,
                du=0,
                dv=1,
                bias=0,
                vth=2 * (len(input_graph)) - 1,
                pos=(float(12 * d + (loop - 1) * 9 * d), -2 * d),
            )

    # Ensure SNN graph is connected(Otherwise, recurrent snn builder can not
    # span/cross the network.)
    for circuit in input_graph.nodes:
        mdsa_snn_graph.add_edges_from(
            [
                (
                    "connecting_node",
                    f"spike_once_{circuit}",
                )
            ],
            weight=0,
        )

    # pylint: disable=R1702
    # Nested blocks are used here to lower runtime complexity. Rewriting the
    # two if statements to if A and B: would increase runtime because the
    # other_node loop would have to be executed for node==neighbour as well.
    for node in input_graph.nodes:
        for neighbour in nx.all_neighbors(input_graph, node):
            if node != neighbour:
                for other_node in input_graph.nodes:
                    if input_graph.has_edge(neighbour, other_node):

                        mdsa_snn_graph.add_edges_from(
                            [
                                (
                                    f"spike_once_{other_node}",
                                    f"degree_receiver_{node}_{neighbour}_0",
                                )
                            ],
                            weight=rand_ceil,
                        )

                        for loop in range(0, shifted_m - 1):
                            # Create list of outgoing edges from a certain
                            # counter neuron.
                            if (
                                f"counter_{other_node}_{loop}"
                                not in right[loop]
                            ):
                                right[loop][
                                    f"counter_{other_node}_{loop}"
                                ] = []
                            right[loop][f"counter_{other_node}_{loop}"].append(
                                f"degree_receiver_{node}_{neighbour}_{loop+1}"
                            )

    # Then create all edges between the nodes.
    for loop in range(1, shifted_m):
        mdsa_snn_graph.add_edges_from(
            [
                (
                    f"next_round_{loop}",
                    f"d_charger_{loop}",
                )
            ],
            weight=1,
        )
        mdsa_snn_graph.add_edges_from(
            [
                (
                    f"delay_{loop}",
                    f"d_charger_{loop}",
                )
            ],
            weight=-1,
        )
        mdsa_snn_graph.add_edges_from(
            [
                (
                    f"d_charger_{loop}",
                    f"delay_{loop}",
                )
            ],
            weight=+1,
        )

    for circuit in input_graph.nodes:
        for loop in range(1, shifted_m):
            # TODO
            mdsa_snn_graph.add_edges_from(
                [
                    (
                        f"delay_{loop}",
                        f"selector_{circuit}_{loop}",
                    )
                ],
                weight=1,  # TODO: doubt.
            )

        # Add synapse between random node and degree receiver nodes.
        for circuit_target in input_graph.nodes:
            if circuit != circuit_target:
                # Check if there is an edge from neighbour_a to neighbour_b.
                if circuit in nx.all_neighbors(input_graph, circuit_target):
                    for loop in range(0, shifted_m):
                        mdsa_snn_graph.add_edges_from(
                            [
                                (
                                    f"rand_{circuit}_{loop}",
                                    f"degree_receiver_{circuit_target}_"
                                    + f"{circuit}_{loop}",
                                )
                            ],
                            weight=rand_nrs[circuit],
                        )

                    # for loop in range(0, m):
                    # TODO: change to degree_receiver_x_y_z and update synapses
                    # for loop from 1,m to 0,m.
                    for loop in range(1, shifted_m):
                        mdsa_snn_graph.add_edges_from(
                            [
                                (
                                    f"degree_receiver_{circuit_target}_"
                                    + f"{circuit}_{loop-1}",
                                    f"next_round_{loop}",
                                )
                            ],
                            weight=1,
                        )

        # Add synapse from degree_selector to selector node.
        for neighbour_b in nx.all_neighbors(input_graph, circuit):
            if circuit != neighbour_b:
                mdsa_snn_graph.add_edges_from(
                    [
                        (
                            f"degree_receiver_{circuit}_{neighbour_b}_"
                            + f"{shifted_m-1}",
                            f"counter_{neighbour_b}_{shifted_m-1}",
                        )
                    ],
                    weight=+1,  # to disable bias
                )
                for loop in range(0, shifted_m):
                    mdsa_snn_graph.add_edges_from(
                        [
                            (
                                f"degree_receiver_{circuit}_{neighbour_b}"
                                + f"_{loop}",
                                f"selector_{circuit}_{loop}",
                            )
                        ],
                        weight=-5,  # to disable bias
                    )
                    # Create list of outgoing edges from a certain counter
                    # neuron.
                    if (
                        f"degree_receiver_{circuit}_{neighbour_b}_{loop}"
                        not in left[loop]
                    ):
                        left[loop][
                            f"degree_receiver_{circuit}_{neighbour_b}_{loop}"
                        ] = []
                    left[loop][
                        f"degree_receiver_{circuit}_{neighbour_b}_{loop}"
                    ].append(f"counter_{neighbour_b}_{loop}")

        # Add synapse from selector node back into degree selector.
        for neighbour_b in nx.all_neighbors(input_graph, circuit):
            if circuit != neighbour_b:
                for loop in range(0, shifted_m):
                    mdsa_snn_graph.add_edges_from(
                        [
                            (
                                f"selector_{circuit}_{loop}",
                                f"degree_receiver_{circuit}_{neighbour_b}_"
                                + f"{loop}",
                            )
                        ],
                        weight=1,  # To increase u(t) at every timestep.
                    )
    # TODO: verify indentation level.
    create_synapses_and_spike_dicts(
        input_graph, mdsa_snn_graph, left, shifted_m, rand_ceil, right
    )

    return mdsa_snn_graph
