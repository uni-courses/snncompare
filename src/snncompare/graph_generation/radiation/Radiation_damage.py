"""Simulates radiation damage on SNN.

Currently simulates it through simulating neuron death by setting the
threshold voltage to 1000 V (which will not be reached used in the
current MDSA approximation). TODO: include other radiation effects such
as "unexpected/random" changes in neuronal and synaptic properties.
"""
import random
from typing import List

import networkx as nx


class Radiation_damage:
    """Creates expected properties of the spike_once neuron."""

    def __init__(self, probability: float):
        self.neuron_death_probability = (
            probability  # % of neurons that will decay.
        )

    def get_random_list_of_len_n(self, n: int, max_val: int) -> List[int]:
        """Does not include max, only below.

        :param n:
        :param max_val:
        """
        randomlist = []
        for _ in range(int(0), int(n)):
            n = random.randint(0, max_val)  # nosec - using a random seed.
        randomlist.append(n)

        return randomlist

    def inject_simulated_radiation(
        self, get_degree: dict, probability: float, seed: int
    ) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param probability:

        """
        # Get list of dead neurons.
        # dead_neurons = self.get_list_of_dead_neurons(get_degree)

        # Get random neurons from list.
        dead_neuron_names = self.get_random_neurons(
            get_degree, probability, seed
        )

        store_dead_neuron_names_in_graph(get_degree, dead_neuron_names)

        # Kill neurons.
        self.kill_neurons(get_degree, dead_neuron_names)

        return dead_neuron_names

    def get_random_neurons(
        self, get_degree: nx.DiGraph, probability: float, seed: int
    ) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param probability:
        :param adaptation_only:  (Default value = False)

        """

        # TODO: restore the probabilitiy  of firing instead of getting fraction
        # of neurons.
        nr_of_dead_neurons = int(len(get_degree) * probability)

        random.seed(seed)
        # Get a list of length nr_of_dead_neurons with random integers
        # These integers indicate which neurons die.

        rand_indices = random.sample(
            range(0, len(get_degree)), nr_of_dead_neurons
        )

        dead_neuron_names = []
        # TODO: fold instead of for.
        count = 0
        for nodename in get_degree:

            # for i,node_name in enumerate(get_degree):
            if count in rand_indices:
                dead_neuron_names.append(nodename)
            count = count + 1

        # for node_name in get_degree:
        # if self.kill_neuron(probability):
        # dead_neuron_names.append(node_name)
        return dead_neuron_names

    def kill_neuron(self, probability: float) -> bool:
        """probabiltiy: 0 to 1 (exc. 1)
        Returns bool true or false

        :param probability:

        """

        return random.random() < probability  # nosec - using a random seed.

    def get_list_of_dead_neurons(self, get_degree: nx.DiGraph) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.

        """
        spike_once_neurons = self.get_spike_once_nodes(get_degree)
        self.get_selector_nodes(get_degree)
        self.get_degree_receiver_nodes(get_degree)
        return spike_once_neurons

    def get_spike_once_nodes(self, get_degree: nx.DiGraph) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param m: The amount of approximation iterations used in the MDSA
        approximation.  (Default value = None)

        """
        spike_once_nodes = []
        for node_name in get_degree.nodes:
            if node_name[:11] == "spike_once_":
                spike_once_nodes.append(node_name)
        return spike_once_nodes

        # vth = get_degree.nodes[node_name]["vth"] + 1
        # if node_name == "spike_once_0":
        #    vth = 9999

    def get_degree_receiver_nodes(self, get_degree: nx.DiGraph) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param m: The amount of approximation iterations used in the MDSA
        approximation.  (Default value = None)

        """
        degree_receiver_nodes = []
        for node_name in get_degree.nodes:
            if node_name[:16] == "degree_receiver_":
                degree_receiver_nodes.append(node_name)
        return degree_receiver_nodes

    def get_selector_nodes(self, get_degree: nx.DiGraph) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param m: The amount of approximation iterations used in the MDSA
        approximation.  (Default value = None)

        """
        selector_nodes = []
        for node_name in get_degree.nodes:
            if node_name[:9] == "selector_":
                selector_nodes.append(node_name)
        return selector_nodes

    def kill_neurons(
        self, get_degree: nx.DiGraph, dead_node_names: List[str]
    ) -> None:
        """Simulates dead neurons by setting spiking voltage threshold to near
        infinity.

        Ensures neuron does not spike.

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param dead_node_names:
        """
        for node_name in dead_node_names:
            if node_name in dead_node_names:
                get_degree.nodes[node_name]["vth"] = 9999


def store_dead_neuron_names_in_graph(
    G: nx.DiGraph, dead_neuron_names: List[str]
) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param dead_neuron_names:

    """

    for nodename in G.nodes:
        if nodename in dead_neuron_names:
            G.nodes[nodename]["rad_death"] = True
        else:
            G.nodes[nodename]["rad_death"] = False


def verify_radiation_is_applied(
    some_graph: nx.DiGraph, dead_neuron_names: List[str], rad_type: str
) -> None:
    """Goes through the dead neuron names, and verifies the radiation is
    applied correctly."""

    # TODO: include check to see if store_dead_neuron_names_in_graph is
    # executed correctly by checking whether the:G.nodes[nodename]["rad_death"]
    # = True
    if rad_type == "neuron_death":
        for nodename in some_graph:
            if nodename in dead_neuron_names:
                if not some_graph.nodes[nodename]["rad_death"]:
                    raise Exception(
                        'Error, G.nodes[nodename]["rad_death"] not set'
                    )
                if some_graph.nodes[nodename]["vth"] != 9999:
                    raise Exception(
                        "Error, radiation is not applied to:{nodename}, even"
                        + f" though it is in:{dead_neuron_names}"
                    )
    else:
        raise Exception(
            f"Error, radiation type: {rad_type} is not yet supported."
        )
