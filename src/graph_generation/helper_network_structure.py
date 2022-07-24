"""Assists the conversion from the input graph to an SNN graph that performs
the MDSA approximation."""
import networkx as nx
import pylab as plt  # TODO: verify not matplotlib.

from src.export_results.Plot_to_tex import Plot_to_tex


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
