"""Stores the input graph in:
latex/Images/graphs/input_graph/isomorphicHash_run_config_id.svg"""
from pprint import pprint
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
    print("xy_pos")
    pprint(xy_pos)

    nx.draw(
        input_graph,
        xy_pos,
    )

    isomorphic_hash: str = get_isomorphic_graph_hash(some_graph=input_graph)
    create_target_dir_if_not_exists(some_path=output_config.input_graph_dir)
    output_filename: str = (
        f"{output_config.input_graph_dir}/{len(input_graph)}"
        + f"_{isomorphic_hash}_{run_config.unique_id}.svg"
    )
    plt.savefig(
        output_filename,
        dpi=200,
    )
    plt.clf()
    plt.close()
