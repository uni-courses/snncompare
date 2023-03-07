"""Returns the updated plot data."""
from typing import Dict

import networkx as nx
from typeguard import typechecked


@typechecked
def get_nx_node_colours(*, G: nx.DiGraph, t: int) -> Dict[str, str]:
    """Returns a list of node colours in order of G.nodes."""
    rgb_colours: Dict[str, str] = {
        "red": "rgb(255, 0, 0)",
        "yellow": "rgb(255, 0, 0)",
        "olive": "rgb(128, 128, 0)",
        "green": "rgb(0, 255, 0)",
        "gray": "rgb(128, 128, 128)",
    }
    colour_dict: Dict[str, str] = {}
    for node_name in G.nodes:
        if "nx_lif" in G.nodes[node_name].keys():
            set_radiation_death_colour(
                colour_dict=colour_dict,
                G=G,
                node_name=node_name,
                rgb_colours=rgb_colours,
                t=t,
            )
            set_spiking_neuron_colour(
                colour_dict=colour_dict,
                G=G,
                node_name=node_name,
                rgb_colours=rgb_colours,
                t=t,
            )
            set_remaining_node_colours(
                colour_dict=colour_dict,
                node_name=node_name,
                rgb_colours=rgb_colours,
            )
        else:
            colour_dict[node_name] = "rgb(255, 255, 0)"

    return colour_dict


@typechecked
def set_radiation_death_colour(
    *,
    colour_dict: Dict,
    G: nx.DiGraph,
    node_name: str,
    rgb_colours: Dict[str, str],
    t: int,
) -> None:
    """Adds the radiation death colour into the colour dict for the nodes."""
    if "rad_death" in G.nodes[node_name].keys():
        if G.nodes[node_name]["rad_death"]:
            colour_dict[node_name] = rgb_colours["red"]
            if G.nodes[node_name]["nx_lif"][t].spikes:
                print(f"DEAD NEURON:{node_name} spiked at {t}.")
                # TODO: restore error
                # raise ValueError("Dead neuron can't spike.")


@typechecked
def set_spiking_neuron_colour(
    *,
    colour_dict: Dict,
    G: nx.DiGraph,
    node_name: str,
    rgb_colours: Dict[str, str],
    t: int,
) -> None:
    """Adds the spiking colour into the colour dict for the nodes."""
    if G.nodes[node_name]["nx_lif"][t].spikes:
        colour_dict[node_name] = rgb_colours["green"]


@typechecked
def set_remaining_node_colours(
    *,
    colour_dict: Dict,
    node_name: str,
    rgb_colours: Dict[str, str],
) -> None:
    """Sets the colour of the neurons that neither died in radiation, nor
    spiked.

    It sets normal non-spiking neurons yellow, and redundant in olive
    colour/dark yellow.
    """
    if node_name not in colour_dict:
        if node_name[:2] == "r_":
            colour_dict[node_name] = rgb_colours["olive"]
        else:
            colour_dict[node_name] = rgb_colours["yellow"]
