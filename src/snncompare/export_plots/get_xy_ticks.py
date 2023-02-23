"""Computes the x-tick- and y-tick position- and values for the plot."""
from typing import Dict, List, Tuple

import networkx as nx
from snnbackends.networkx.LIF_neuron import LIF_neuron
from typeguard import typechecked


# pylint: disable=R0912
@typechecked
def store_xy_ticks(
    lif_neurons: List[LIF_neuron], plotted_graph: nx.DiGraph
) -> None:
    """Stores the x-axis and y-axis ticks into the object."""
    x_ticks, y_ticks = get_xy_tick_labels(lif_neurons)
    plotted_graph.graph["x_tics"] = x_ticks
    plotted_graph.graph["y_tics"] = y_ticks


@typechecked
def get_xy_tick_labels(
    lif_neurons: List[LIF_neuron],
) -> Tuple[Dict[float, str], Dict[float, str]]:
    """Computes the x-tick position and labels for the plot.

    TODO: filter which neuron types you want on x-axis.
    TODO: filter which neuron types you want on y-axis.
    TODO: make algorithm dependent.
    """
    x_tick_labels: Dict[float, str] = {}
    y_tick_labels: Dict[float, str] = {}
    sorted_neurons: Dict[str, List[LIF_neuron]] = get_sorted_neurons(
        lif_neurons=lif_neurons
    )
    for neuron_type, neurons in sorted_neurons.items():
        if neuron_type in ["connector_node", "counter", "terminator_node"]:
            x_tick_labels[neurons[0].pos[0]] = neuron_type
        if neuron_type in ["degree_receiver", "selector"]:
            for neuron in neurons:
                # pylint: disable=C0201
                if neuron.pos[0] not in x_tick_labels.keys():
                    if neuron_type == "degree_receiver":
                        m_val_identifier_index = 2
                    elif neuron_type == "selector":
                        m_val_identifier_index = 1
                    some_identifier = neuron.identifiers[
                        m_val_identifier_index
                    ]
                    x_tick_labels[neuron.pos[0]] = (
                        f"{neuron_type}, "
                        + f"{some_identifier.description}="
                        + f"{some_identifier.value}"
                    )
        if neuron_type in ["rand", "spike_once"]:
            for neuron in neurons:
                y_tick_labels[neuron.pos[1]] = neuron.full_name

    return x_tick_labels, y_tick_labels


@typechecked
def get_sorted_neurons(
    lif_neurons: List[LIF_neuron],
) -> Dict[str, List[LIF_neuron]]:
    """Sorts the LIF_neurons on neuron name."""
    sorted_neurons: Dict[str, List[LIF_neuron]] = {}
    for lif_neuron in lif_neurons:
        for neuron_name in [
            "spike_once",
            "rand",
            "degree_receiver",
            "selector",
            "counter",
            "next_round",
            "connector_node",
            "terminator_node",
        ]:
            put_neuron_into_sorted_dict(
                lif_neuron=lif_neuron,
                neuron_name=neuron_name,
                sorted_neurons=sorted_neurons,
            )
    return sorted_neurons


@typechecked
def put_neuron_into_sorted_dict(
    lif_neuron: LIF_neuron,
    neuron_name: str,
    sorted_neurons: Dict[str, List[LIF_neuron]],
) -> None:
    """Puts a neuron in its category/key/neuron type if it ."""
    if lif_neuron.name == neuron_name:
        if neuron_name in sorted_neurons.keys():
            sorted_neurons[neuron_name].append(lif_neuron)
        else:
            sorted_neurons[neuron_name] = [lif_neuron]
