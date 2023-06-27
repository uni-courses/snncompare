"""Stores the input graph in:
latex/Images/graphs/input_graph/isomorphicHash_run_config_id.svg"""
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from simplt.export_plot import create_target_dir_if_not_exists
from typeguard import typechecked

from snncompare.import_results.helper import get_isomorphic_graph_hash
from snncompare.optional_config import Output_config
from snncompare.run_config import Run_config


@typechecked
def output_input_graph(
    *,
    input_graph: nx.Graph,
    output_config: Output_config,
    run_config: Run_config,
) -> None:
    """Assumes the output directory already exists.

    Outputs the nx.graph as svg file.
    """

    xy_pos: Dict[int, Tuple[int, int]] = {}
    for node_index in input_graph.nodes:
        xy_pos[node_index] = (0, node_index)

    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=input_graph)
    print(f"isomorphic_hash={isomorphic_hash}")
    create_target_dir_if_not_exists(some_path=output_config.input_graph_dir)
    output_filename: str = (
        f"{output_config.input_graph_dir}/{len(input_graph)}"
        + f"_{isomorphic_hash}_{run_config.unique_id}.pdf"
    )

    draw_bend_input_graph_edges(input_graph=input_graph, xy_pos=xy_pos)

    plt.savefig(
        output_filename,
        dpi=200,
    )
    plt.show()
    plt.clf()
    plt.close()


@typechecked
def draw_bend_input_graph_edges(
    *, input_graph: nx.Graph, xy_pos: Dict[int, Tuple[int, int]]
) -> None:
    """Ensures the edges of the networkx graph are bent manually."""
    # alpha is how see through.
    nx.draw_networkx_nodes(
        input_graph, xy_pos, node_color="b", node_size=100, alpha=1
    )
    ax = plt.gca()

    for e in input_graph.edges:
        ax.annotate(
            "",
            xy=xy_pos[e[0]],
            xycoords="data",
            xytext=xy_pos[e[1]],
            textcoords="data",
            arrowprops={
                "arrowstyle": "-",
                "color": "0.5",
                "shrinkA": 5,
                "shrinkB": 5,
                "patchA": None,
                "patchB": None,
                "connectionstyle": "arc3,rad=0.4",
            },
        )

    # turn the axis on
    ax.set_axis_on()
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels([0, 1, 2, 3, 4], color="k", size=20)

    # plt.xlabel(x_axis_label)
    plt.ylabel("node number [-]")
