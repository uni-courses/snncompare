"""Verifies the graph represents a connected and valid SNN, with all required
neuron and synapse properties specified."""

# Import the networkx module.
from pprint import pprint

import networkx as nx
import numpy as np


def verify_lava_neuron_properties_are_specified(
    node: nx.DiGraph.nodes,
) -> None:
    """

    :param node: nx.DiGraph.nodes:
    :param node: nx.DiGraph.nodes:

    """
    pprint(node)
    bias = node["lava_LIF"].bias_mant.get()
    if not isinstance(bias, (float, np.ndarray)):
        # TODO: include additional verifications on dimensions of bias.
        raise Exception(
            f"Bias is not a np.ndarray, it is of type:{type(bias)}."
        )

    if not isinstance(node["lava_LIF"].du.get(), float):
        raise Exception("du is not a float.")
    if not isinstance(node["lava_LIF"].dv.get(), float):
        raise Exception("dv is not a float.")
    if not isinstance(node["lava_LIF"].vth.get(), float):
        raise Exception("vth is not a float.")
