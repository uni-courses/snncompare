"""Returns the updated plot data.

TODO: rename and restructure this function along with:
helper_network_structure.
"""
from typing import Dict, List, Tuple

import networkx as nx
from typeguard import typechecked


@typechecked
def set_nx_node_colours(
    *, G: nx.DiGraph, t: int
) -> Tuple[Dict[str, str], List, List]:
    """Returns a list of node colours in order of G.nodes.

    TODO: simplify function.
    """
    color_map = []
    colour_dict: Dict[str, str] = {}
    spiking_edges = []

    for node_name in G.nodes:
        if "nx_lif" in G.nodes[node_name].keys():
            if "rad_death" in G.nodes[node_name].keys():
                if G.nodes[node_name]["rad_death"]:
                    # colour_dict[node_name] = ["red",0.5]
                    colour_dict[node_name] = "rgb(255, 0, 0)"
                    if G.nodes[node_name]["nx_lif"][t].spikes:
                        raise Exception("Dead neuron can't spike.")
            if G.nodes[node_name]["nx_lif"][t].spikes:
                # colour_dict[node_name] = ["green",0.5]
                colour_dict[node_name] = "rgb(0, 255, 0)"
                for neighbour in nx.all_neighbors(G, node_name):
                    spiking_edges.append((node_name, neighbour))
            if node_name not in colour_dict:
                set_node_colours_with_redundancy(
                    colour_dict=colour_dict, node_name=node_name
                )
        else:
            colour_dict[node_name] = "rgb(255, 255, 0)"
    for node_name in G.nodes:
        color_map.append(colour_dict[node_name])
    return colour_dict, color_map, spiking_edges


@typechecked
def set_node_colours_with_redundancy(
    *, colour_dict: Dict, node_name: str
) -> None:
    """Sets the colour of the redundant node different than the original
    node."""

    # TODO: include redundancy level
    # if node_name[:4] == f"r_{red_level}_":
    if node_name[:2] == "r_":
        # colour_dict[node_name] = ["olive",0.5]
        colour_dict[node_name] = "rgb(128, 128, 0)"
    else:
        # colour_dict[node_name] = ["yellow",1]
        colour_dict[node_name] = "rgb(255, 255, 0)"
