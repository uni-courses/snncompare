"""Computes the x-tick- and y-tick position- and values for the plot."""
from typing import Dict, List

from snnbackends.networkx.LIF_neuron import LIF_neuron
from typeguard import typechecked


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
