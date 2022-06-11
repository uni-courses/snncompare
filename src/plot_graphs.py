"""File used to generate graph plots."""

import os

import matplotlib.pyplot as plt
import networkx as nx


def plot_circular_graph(density, G, recurrent_edge_density, test_scope):
    """Generates a circular plot of a (directed) graph.

    :param density:
    :param G:
    :param seed:
    :param export:  (Default value = True)
    :param show:  (Default value = True)
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


def plot_uncoordinated_graph(G, export=False, show=True):
    """Generates a circular plot of a (directed) graph.

    :param density:
    :param G:
    :param seed:
    :param export:  (Default value = True)
    :param show:  (Default value = True)
    """
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


def create_target_dir_if_not_exists(path, new_dir_name):
    """Creates an output dir for graph plots.

    :param path:
    :param new_dir_name:
    """

    create_root_dir_if_not_exists(path)
    if not os.path.exists(f"{path}/{new_dir_name}"):
        os.makedirs(f"{path}/{new_dir_name}")


def create_root_dir_if_not_exists(root_dir_name):
    if not os.path.exists(root_dir_name):
        os.makedirs(f"{root_dir_name}")
    if not os.path.exists(root_dir_name):
        raise Exception(f"Error, root_dir_name={root_dir_name} did not exist.")


def get_labels(G, configuration):
    """Returns the labels for the plot nodes.

    :param G:
    :param configuration:
    """
    labels = {}
    for node_name in G.nodes:
        if configuration == "du":
            labels[node_name] = f"{node_name}"
            # ] = f'{node_name},R:{G.nodes[node_name]["lava_LIF"].du.get()}'
    return labels
