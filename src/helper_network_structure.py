"""Assists the conversion from the input graph to an SNN graph that performs
the MDSA approximation."""
import networkx as nx
import pylab as plt  # TODO: verify not matplotlib.

from src.helper import get_y_position
from src.Plot_to_tex import Plot_to_tex


def get_degree_graph_with_separate_wta_circuits(G, rand_nrs, rand_ceil, m):
    """Returns a networkx graph that represents the snn that computes the
    spiking degree in the degree_receiver neurons. One node in the graph
    represents one neuron. A directional edge in the graph represents a synapse
    between two neurons.

    One spike once neuron is created per node in graph G.
    One degree_receiver neuron is created per node in graph G.
    A synapse is created from each spike_once neuron that represents node A
    to each of the degree_receiver that represents a neighbour of node A.

    :param G: The original graph on which the MDSA algorithm is ran.
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
    m = m + 1
    d = 0.25 * m  # specify grid distance size
    # Specify edge weight for recurrent inhibitory synapse.
    inhib_recur_weight = -10
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
            recur=inhib_recur_weight,
        )

        for neighbour in nx.all_neighbors(G, node):
            if node != neighbour:
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
                        recur=inhib_recur_weight,
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
                recur=inhib_recur_weight,
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

    # pylint: disable=R1702
    # Nested blocks are used here to lower runtime complexity. Rewriting the
    # two if statements to if A and B: would increase runtime because the
    # other_node loop would have to be executed for node==neighbour as well.
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
    # TODO: verify indentation level.
    create_synapses_and_spike_dicts(G, get_degree, left, m, rand_ceil, right)

    return get_degree


def create_synapses_and_spike_dicts(G, get_degree, left, m, rand_ceil, right):
    """Creates some synapses and the spike dictionary."""
    # pylint: disable=R0913
    # 6/5 arguments are currently used in the synapse creation method.
    # TODO: add recurrent synapses (as edges).
    add_recursive_edges_to_graph(get_degree)

    # Create replacement synapses.
    if m <= 1:
        # TODO: remove return get_degree
        get_degree = create_degree_synapses_for_m_is_zero(
            get_degree, left, m, rand_ceil, right
        )
    else:
        get_degree = retry_create_degree_synapses(G, get_degree, m, rand_ceil)

    # Create spike dictionaries with [t] as key, and boolean spike as value for
    # each node.
    for node in get_degree.nodes:
        get_degree.nodes[node]["spike"] = {}


def create_degree_synapses_for_m_is_zero(
    get_degree, left, m, rand_ceil, right
):
    """

    :param get_degree: Graph with the MDSA SNN approximation solution.
    :param left:
    :param m: The amount of approximation iterations used in the MDSA
     approximation.
    :param rand_ceil: Ceiling of the range in which rand nrs can be generated.
    :param right:

    """
    # pylint: disable=R1702
    # Currently no method is found to reduce the 6/5 nested blocks.
    print(f"m={m},OLD")
    for some_id in range(m - 1):
        for l_key, l_value in left[some_id].items():
            for l_counter in l_value:
                for r_key, r_value in right[some_id].items():
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
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param get_degree: Graph with the MDSA SNN approximation solution.
    :param m: The amount of approximation iterations used in the MDSA
     approximation.
    :param rand_ceil: Ceiling of the range in which rand nrs can be generated.

    """
    # pylint: disable=R0913
    # Currently no method is found to reduce the 6/5 nested blocks.
    for loop in range(0, m):
        for x_l in G.nodes:
            for y in G.nodes:
                for x_r in G.nodes:
                    if (
                        f"degree_receiver_{x_l}_{y}_{loop}" in get_degree.nodes
                        and (
                            f"degree_receiver_{x_r}_{y}_{loop+1}"
                            in get_degree.nodes
                        )
                    ):
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
    filename,
    G,
    t,
    show=False,
    current=True,
):
    """Plots the neuron behaviour at a specific step of the SNN simulation.

    :param adaptation: indicates if test uses brain adaptation or not.
    :param filename:
    :param G: The original graph on which the MDSA algorithm is ran.
    param iteration: The initialisation iteration that is used.
    :param seed: The value of the random seed used for this test.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param m: The amount of approximation iterations used in the MDSA
     approximation.
    :param t:
    :param show:  (Default value = False)
    :param current:  (Default value = True)
    """
    # TODO: remove unused function.
    # options = {"edgecolors": "red"}
    options = {}
    color_map, spiking_edges, _ = set_node_colours(G, t)
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


def plot_coordinated_graph(
    G,
    desired_properties,
    show=False,
    filename="no_filename",
):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    param iteration: The initialisation iteration that is used.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param desired_properties:  (Default value = [])
    :param show:  (Default value = False)
    :param filename:  (Default value = "no_filename")

    """
    if desired_properties is None:
        desired_properties = []
    color_map, spiking_edges = set_nx_node_colours(G)
    edge_color_map = set_edge_colours(G, spiking_edges)
    # Width=edge width.
    nx.draw(
        G,
        nx.get_node_attributes(G, "pos"),
        with_labels=True,
        node_size=10,
        font_size=6,
        width=0.2,
        node_color=color_map,
        edge_color=edge_color_map,
        # **options,
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

    add_neuron_properties_to_plot(axis, desired_properties, G, G.nodes, pos)

    # f = plt.figure()
    # f.set_figwidth(10)
    # f.set_figheight(10)
    # plt.subplots_adjust(left=0.0, right=4.0, bottom=0.0, top=4.0)
    if show:
        plt.show()

    plot_export = Plot_to_tex()
    print(f"export to:{filename}")
    plot_export.export_plot(plt, filename)
    # plt.savefig()
    plt.clf()
    plt.close()


def add_neuron_properties_to_plot(axis, desired_properties, G, nodenames, pos):
    """Adds a text (annotation) to each neuron with the desired neuron
    properties.

    :param axis:
    :param desired_properties:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param nodenames:
    :param pos:
    """
    for nodename in nodenames:

        # Shift the x-coordinates of the redundant neurons to right for
        # readability.
        if nodename[:3] == "red":
            shift_right = 0.15
        else:
            shift_right = 0

        annotation_text = get_annotation_text(desired_properties, G, nodename)
        # Include text in plot.
        axis.text(
            pos[nodename][0] + shift_right,
            pos[nodename][1],
            annotation_text,
            transform=axis.transData,
            fontsize=4,
        )


def get_annotation_text(desired_properties, G, nodename):
    """Returns a string with the annotation text.

    :param desired_properties:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param nodename: Node of the name of a networkx graph.
    """
    annotation = ""
    if "bias" in desired_properties:
        annotation = (
            annotation + f'bias={G.nodes[nodename]["nx_LIF"].bias.get()}\n'
        )
    if "du" in desired_properties:
        annotation = (
            annotation + f'du={G.nodes[nodename]["nx_LIF"].du.get()}\n'
        )
    if "dv" in desired_properties:
        annotation = (
            annotation + f'dv={G.nodes[nodename]["nx_LIF"].dv.get()}\n'
        )
    if "u" in desired_properties:
        annotation = annotation + f'u={G.nodes[nodename]["nx_LIF"].u.get()}\n'
    if "v" in desired_properties:
        annotation = annotation + f'v={G.nodes[nodename]["nx_LIF"].v.get()}\n'
    if "vth" in desired_properties:
        annotation = (
            annotation + f'vth={G.nodes[nodename]["nx_LIF"].vth.get()}\n'
        )
    if "a_in_next" in desired_properties:
        annotation = (
            annotation + f'a_in_next={G.nodes[nodename]["nx_LIF"].a_in_next}\n'
        )

    return annotation


def plot_unstructured_graph(G, show=False):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    param iteration: The initialisation iteration that is used.
    :param size: Nr of nodes in the original graph on which test is ran.
    :param show:  (Default value = False)

    """
    # TODO: Remove unused function
    nx.draw(G, with_labels=True)
    if show:
        plt.show()
    # plot_export = Plot_to_tex()
    # plot_export.export_plot(plt, f"G_{size}_{iteration}")
    plt.clf()
    plt.close()


def set_nx_node_colours(G):
    """Returns a list of node colours in order of G.nodes.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    color_map = []
    spiking_edges = []

    colour_dict = {}
    for node_name in G.nodes:
        if "nx_LIF" in G.nodes[node_name].keys():
            if "rad_death" in G.nodes[node_name].keys():
                if G.nodes[node_name]["rad_death"]:
                    colour_dict[node_name] = "red"
                    if G.nodes[node_name]["nx_LIF"].spikes:
                        raise Exception("Dead neuron can't spike.")
            if G.nodes[node_name]["nx_LIF"].spikes:
                colour_dict[node_name] = "green"
                for neighbour in nx.all_neighbors(G, node_name):
                    spiking_edges.append((node_name, neighbour))
            if node_name not in colour_dict:
                colour_dict[node_name] = "white"
        else:
            colour_dict[node_name] = "yellow"
    for node_name in G.nodes:
        color_map.append(colour_dict[node_name])
    return color_map, spiking_edges


def set_node_colours(G, t):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param t:

    """
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
                # raise Exception(
                print(f"Did not find spike dictionary for node:{node_name}")
            else:
                color_map.append("yellow")
                for neighbour in nx.all_neighbors(G, node_name):
                    unseen_edges.append((node_name, neighbour))
    return color_map, spiking_edges, unseen_edges


def set_edge_colours(G, spiking_edges):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param spiking_edges:

    """
    edge_color_map = []
    for edge in G.edges:

        if edge in spiking_edges:
            edge_color_map.append("green")
        else:
            edge_color_map.append("black")
    return edge_color_map


def get_labels(G, current=True):
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param current:  (Default value = True)

    """
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


def add_recursive_edges_to_graph(G):
    """Adds recursive edges to graph for nodes that have the recur attribute.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    for nodename in G.nodes:
        if "recur" in G.nodes[nodename].keys():
            G.add_edges_from(
                [
                    (
                        nodename,
                        nodename,
                    )
                ],
                weight=G.nodes[nodename]["recur"],
            )
