from pprint import pprint
import networkx as nx
import pylab as plt  # TODO: verify not matplotlib.

from src.helper import get_y_position
from src.Plot_to_tex import Plot_to_tex


def get_degree_graph_with_separate_wta_circuits(G, rand_nrs, rand_ceil, m):
    m = m + 1
    d = 0.25 * m  # specify grid distance size
    """Returns a networkx graph that represents the snn that computes the
    spiking degree in the degree_receiver neurons.
    One node in the graph represents one neuron.
    A directional edge in the graph represents a synapse between two
    neurons.

    One spike once neuron is created per node in graph G.
    One degree_receiver neuron is created per node in graph G.
    A synapse is created from each spike_once neuron that represents node A
    to each of the degree_receiver that represents a neighbour of node A.
    """
    get_degree = nx.DiGraph()
    # Define list of m mappings for sets of tuples containing synapses
    left = [{} for _ in range(m)]
    right = [{} for _ in range(m)]

    # Create a node to make the graph connected. (Otherwise, recurrent snn
    # builder can not span/cross the network.)
    get_degree.add_node(
        "connecting_node",
        id=len(G.nodes),
        du=0,
        dv=0,
        bias=0,
        vth=1,
        pos=(float(-d), float(d)),
    )

    # First create all the nodes in the get_degree graph.
    for node in G.nodes:

        # One neuron per node named: spike_once_0-n
        get_degree.add_node(
            f"spike_once_{node}",
            id=node,
            du=0,
            dv=0,
            bias=2,
            vth=1,
            pos=(float(0), float(node * 4 * d)),
        )

        for neighbour in nx.all_neighbors(G, node):
            if node != neighbour:
                # One neuron per node named: degree_receiver_0-n.
                # get_degree.add_node(
                #    f"degree_receiver_{node}_{neighbour}",
                #    id=node,
                #    du=0,
                #    dv=1,
                #    bias=0,
                #    vth=1,
                #    pos=(float(1.0), get_y_position(G, node, neighbour)),
                # )

                for loop in range(0, m):
                    get_degree.add_node(
                        f"degree_receiver_{node}_{neighbour}_{loop}",
                        id=node,
                        du=0,
                        dv=1,
                        bias=0,
                        vth=1,
                        pos=(
                            float(4 * d + loop * 9 * d),
                            get_y_position(G, node, neighbour, d),
                        ),
                    )

        # One neuron per node named: rand
        if len(rand_nrs) < len(G):
            raise Exception(
                "The range of random numbers does not allow for randomness"
                + " collision prevention."
            )

        for loop in range(0, m):
            get_degree.add_node(
                f"rand_{node}_{loop}",
                id=node,
                du=0,
                dv=0,
                bias=2,
                vth=1,
                pos=(float(d + loop * 9 * d), float(node * 4 * d) + d),
            )

        # Add winner selector node
        for loop in range(0, m):
            if loop == 0:
                get_degree.add_node(
                    f"selector_{node}_{loop}",
                    id=node,
                    du=0,
                    dv=1,
                    bias=5,
                    vth=4,
                    pos=(float(7 * d + loop * 9 * d), float(node * 4 * d + d)),
                )
            elif loop > 0:
                get_degree.add_node(
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
        get_degree.add_node(
            f"counter_{node}_{m-1}",
            id=node,
            du=0,
            dv=1,
            bias=0,
            vth=0,
            pos=(float(9 * d + loop * 9 * d), float(node * 4 * d)),
        )
        # for loop in range(0, m):
        #    get_degree.add_node(
        #        f"depleter_{node}_{loop}",
        #        id=node,
        #        du=1,
        #        dv=1,
        #        bias=0,
        #        vth=0,
        #        pos=(float(9 * d + loop * 9 * d), float(node * 4 * d) - d),
        #    )

        # Create next round connector neurons.
        for loop in range(1, m):
            get_degree.add_node(
                f"next_round_{loop}",
                id=node,
                du=0,
                dv=1,
                bias=0,
                vth=len(G.nodes) - 1,
                pos=(float(6 * d + (loop - 1) * 9 * d), -2 * d),
            )

            get_degree.add_node(
                f"d_charger_{loop}",
                id=node,
                du=0,
                dv=1,
                bias=0,
                vth=0,
                pos=(float(9 * d + (loop - 1) * 9 * d), -2 * d),
            )

            get_degree.add_node(
                f"delay_{loop}",
                id=node,
                du=0,
                dv=1,
                bias=0,
                vth=2 * (len(G)) - 1,
                pos=(float(12 * d + (loop - 1) * 9 * d), -2 * d),
            )

    # Ensure SNN graph is connected(Otherwise, recurrent snn builder can not
    # span/cross the network.)
    for circuit in G.nodes:
        get_degree.add_edges_from(
            [
                (
                    "connecting_node",
                    f"spike_once_{circuit}",
                )
            ],
            weight=0,
        )
    for node in G.nodes:
        for neighbour in nx.all_neighbors(G, node):
            if node != neighbour:
                for other_node in G.nodes:
                    if G.has_edge(neighbour, other_node):

                        get_degree.add_edges_from(
                            [
                                (
                                    f"spike_once_{other_node}",
                                    f"degree_receiver_{node}_{neighbour}_0",
                                )
                            ],
                            weight=rand_ceil,
                        )

                        for loop in range(0, m - 1):
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
    for loop in range(1, m):
        get_degree.add_edges_from(
            [
                (
                    f"next_round_{loop}",
                    f"d_charger_{loop}",
                )
            ],
            weight=1,
        )
        get_degree.add_edges_from(
            [
                (
                    f"delay_{loop}",
                    f"d_charger_{loop}",
                )
            ],
            weight=-1,
        )
        get_degree.add_edges_from(
            [
                (
                    f"d_charger_{loop}",
                    f"delay_{loop}",
                )
            ],
            weight=+1,
        )

    for circuit in G.nodes:
        for loop in range(1, m):
            # TODO
            get_degree.add_edges_from(
                [
                    (
                        f"delay_{loop}",
                        f"selector_{circuit}_{loop}",
                    )
                ],
                weight=1,  # TODO: doubt.
            )

        # Add synapse between random node and degree receiver nodes.
        for circuit_target in G.nodes:
            if circuit != circuit_target:
                # Check if there is an edge from neighbour_a to neighbour_b.
                if circuit in nx.all_neighbors(G, circuit_target):
                    for loop in range(0, m):
                        get_degree.add_edges_from(
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
                    for loop in range(1, m):
                        get_degree.add_edges_from(
                            [
                                (
                                    f"degree_receiver_{circuit_target}_"
                                    + f"{circuit}_{loop-1}",
                                    f"next_round_{loop}",
                                )
                            ],
                            weight=1,
                        )

        # for loop in range(0, m):
        #    get_degree.add_edges_from(
        #        [
        #            (
        #                f"next_round_{loop}",
        #                f"depleter_{circuit}_{loop}",
        #            )
        #        ],
        #        weight=+1,
        #    )
        # for loop in range(0, m):
        #    get_degree.add_edges_from(
        #        [
        #            (
        #                f"counter_{circuit}_{loop}",
        #                f"depleter_{circuit}_{loop}",
        #            )
        #        ],
        #        weight=+1,
        #    )
        #    get_degree.add_edges_from(
        #        [
        #            (
        #                f"depleter_{circuit}_{loop}",
        #                f"counter_{circuit}_{loop}",
        #            )
        #        ],
        #        weight=+len(G),
        #    )

        # Add synapse from degree_selector to selector node.
        for neighbour_b in nx.all_neighbors(G, circuit):
            if circuit != neighbour_b:
                get_degree.add_edges_from(
                    [
                        (
                            f"degree_receiver_{circuit}_{neighbour_b}_{m-1}",
                            f"counter_{neighbour_b}_{m-1}",
                        )
                    ],
                    weight=+1,  # to disable bias
                )
                for loop in range(0, m):
                    get_degree.add_edges_from(
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

        # TODO:
        # Add synapse from selector node back into degree selector.
        for neighbour_b in nx.all_neighbors(G, circuit):
            if circuit != neighbour_b:
                for loop in range(0, m):
                    get_degree.add_edges_from(
                        [
                            (
                                f"selector_{circuit}_{loop}",
                                f"degree_receiver_{circuit}_{neighbour_b}_"
                                + f"{loop}",
                            )
                        ],
                        weight=1,  # To increase u(t) at every timestep.
                    )

    # Create replacement synapses.
    if m <= 1:
        get_degree = create_degree_synapses_for_m_is_zero(
            get_degree, left, m, rand_ceil, right
        )
    else:
        get_degree = retry_create_degree_synapses(G, get_degree, m, rand_ceil)

    # Create spike dictionaries with [t] as key, and boolean spike as value for
    # each node.
    for node in get_degree.nodes:
        get_degree.nodes[node]["spike"] = {}
    return get_degree


def create_degree_synapses_for_m_is_zero(
    get_degree, left, m, rand_ceil, right
):
    print(f"m={m},OLD")
    for id in range(m - 1):
        for l_key, l_value in left[id].items():
            for l_counter in l_value:
                for r_key, r_value in right[id].items():
                    for r_degree in r_value:
                        if l_counter == r_key:
                            get_degree.add_edges_from(
                                [
                                    (
                                        l_key,
                                        r_degree,
                                    )
                                ],
                                weight=rand_ceil,  # Increase u(t) at each t.
                            )
    return get_degree


def retry_create_degree_synapses(G, get_degree, m, rand_ceil):
    for loop in range(0, m):
        for x_l in G.nodes:
            for y in G.nodes:
                for x_r in G.nodes:
                    if f"degree_receiver_{x_l}_{y}_{loop}" in get_degree.nodes:
                        if (
                            f"degree_receiver_{x_r}_{y}_{loop+1}"
                            in get_degree.nodes
                        ):
                            # if not G.has_edge(x_l, v):
                            get_degree.add_edges_from(
                                [
                                    (
                                        f"degree_receiver_{x_l}_{y}_{loop}",
                                        f"degree_receiver_{x_r}_{y}_{loop+1}",
                                    )
                                ],
                                weight=rand_ceil,  # Increase u(t) at each t.
                            )
    return get_degree


def plot_neuron_behaviour_over_time(
    adaptation,
    filename,
    G,
    iteration,
    seed,
    size,
    m,
    t,
    show=False,
    current=True,
):

    # options = {"edgecolors": "red"}
    options = {}
    color_map, spiking_edges, unseen_edges = set_node_colours(G, t)
    edge_color_map = set_edge_colours(G, spiking_edges)

    nx.draw(
        G,
        nx.get_node_attributes(G, "pos"),
        with_labels=True,
        node_size=8,
        font_size=5,
        width=0.2,
        node_color=color_map,
        edge_color=edge_color_map,
        **options,
    )
    node_labels = get_labels(G, current)
    pos = {
        node: (x, y)
        for (node, (x, y)) in nx.get_node_attributes(G, "pos").items()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=5)

    plt.axis("off")
    axis = plt.gca()
    axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    axis.set_ylim([1.2 * y for y in axis.get_ylim()])

    if show:
        plt.show()

    plot_export = Plot_to_tex()
    plot_export.export_plot(
        plt,
        # f"test_object_seed_adapt_{adaptation}_{seed}_size{size}_m{m}_iter{iteration}_t{t}",
        filename,
    )
    # plt.savefig()
    plt.clf()
    plt.close()


def plot_coordinated_graph(G, iteration, size, show=False):
    # Width=edge width.
    pprint(nx.get_node_attributes(G, "pos"))
    nx.draw(
        G,
        nx.get_node_attributes(G, "pos"),
        with_labels=True,
        node_size=8,
        font_size=5,
        width=0.2,
    )
    node_labels = nx.get_node_attributes(G, "")
    pos = {
        node: (x, y)
        for (node, (x, y)) in nx.get_node_attributes(G, "pos").items()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=5)

    plt.axis("off")
    axis = plt.gca()
    axis.set_xlim([1.2 * x for x in axis.get_xlim()])
    axis.set_ylim([1.2 * y for y in axis.get_ylim()])
    # f = plt.figure()
    # f.set_figwidth(10)
    # f.set_figheight(10)
    # plt.subplots_adjust(left=0.0, right=4.0, bottom=0.0, top=4.0)
    if show:
        plt.show()

    plot_export = Plot_to_tex()
    plot_export.export_plot(plt, f"snn_{size}_{iteration}")
    # plt.savefig()
    plt.clf()
    plt.close()


def plot_unstructured_graph(G, iteration, size, show=False):
    nx.draw(G, with_labels=True)
    if show:
        plt.show()
    # plot_export = Plot_to_tex()
    # plot_export.export_plot(plt, f"G_{size}_{iteration}")
    plt.clf()
    plt.close()


def set_node_colours(G, t):
    color_map = []
    spiking_edges = []
    unseen_edges = []
    for node_name in G.nodes:
        if G.nodes[node_name]["spike"] != {}:
            # for node in G:
            if G.nodes[node_name]["spike"][t]:
                color_map.append("green")
                for neighbour in nx.all_neighbors(G, node_name):
                    spiking_edges.append((node_name, neighbour))
            else:
                color_map.append("white")
        else:
            if node_name[:11] != "connecting_":
                raise Exception(
                    f"Did not find spike dictionary for node:{node_name}"
                )
            else:
                color_map.append("yellow")
                for neighbour in nx.all_neighbors(G, node_name):
                    unseen_edges.append((node_name, neighbour))
    return color_map, spiking_edges, unseen_edges


def set_edge_colours(G, spiking_edges):
    edge_color_map = []
    for edge in G.edges:

        if edge in spiking_edges:
            edge_color_map.append("green")
        else:
            edge_color_map.append("black")
    return edge_color_map


def get_labels(G, current=True):
    node_labels = {}
    reset_labels = False
    if current:
        for node_name in G.nodes:
            if node_name != "connecting_node":
                # print u.
                if not G.nodes[node_name]["neuron"] is None:
                    node_labels[node_name] = G.nodes[node_name][
                        "neuron"
                    ].u.get()[0]
                else:
                    reset_labels = True
            else:
                node_labels[node_name] = "0"
    else:
        node_labels = nx.get_node_attributes(G, "")

    # If neurons were not stored in run, they are None, then get default
    # labels.
    if reset_labels:
        node_labels = nx.get_node_attributes(G, "")
    return node_labels
