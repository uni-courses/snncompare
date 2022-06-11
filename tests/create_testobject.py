"""Creates test object."""

import copy
from lava.proc.monitor.process import Monitor

from src.helper import fill_dictionary, generate_list_of_n_random_nrs
from src.helper_network_structure import (
    get_degree_graph_with_separate_wta_circuits,
    plot_coordinated_graph,
    plot_unstructured_graph,
)
from src.old_conversion import convert_networkx_graph_to_snn_with_one_neuron

# from src.networkx_to_snn import convert_networkx_graph_to_snn_with_one_neuron


def create_test_object(
    adaptation,
    G,
    iteration,
    m,
    rad_dam,
    plot_input_graph=False,
    plot_snn_graph=False,
    export=True,
):
    test_object = Test_properties()
    # Specify the expected neuron properties.
    # TODO: change this to make it a function of
    test_object.sample_selector_neuron = Selector_neuron()
    test_object.sample_spike_once_neuron = Spike_once_neuron()
    test_object.sample_rand_neuron = Rand_neuron()
    test_object.sample_degree_receiver_neuron = Degree_receiver()
    test_object.sample_counter_neuron = Counter_neuron(G)
    test_object.m = m

    # Specify the expected synaptic weights
    # TODO: Specify per synapse group. (except for the random synapses)
    test_object.incoming_selector_weight = -5

    # Move the graph on which the algorithm is ran.
    test_object.G = G

    if plot_input_graph or export:
        plot_unstructured_graph(
            test_object.G, iteration, len(G), plot_input_graph
        )

    # Generate the maximum random ceiling
    # +2 to allow selecting a larger range of numbers than the number of
    # nodes in the graph.
    test_object.rand_ceil = len(test_object.G) + 0
    # Get the list of random numbers.
    test_object.rand_nrs = generate_list_of_n_random_nrs(
        test_object.G, max=test_object.rand_ceil, seed=42
    )
    print(f"before delta={test_object.rand_nrs}")

    # Make the random numbers differ with at least delta>=2. This is to
    # prevent multiple degree_receiver_x_y neurons (that differ less than
    # delta) in a single WTA circuit to spike before they are inhibited by
    # the first winner. This inhibition goes via the selector neuron and
    # has a delay of 2. So a winner should have a difference of at least 2.
    test_object.delta = 2
    # Spread the random numbers with delta to ensure 1 winner in WTA
    # circuit.
    test_object.rand_nrs = [
        x * test_object.delta for x in test_object.rand_nrs
    ]
    print(f"after_delta={test_object.rand_nrs}")
    print(f"test_object.rand_ceil={test_object.rand_ceil}")
    # Add inhibition to rand_nrs to ensure the degree_receiver current u[1]
    # always starts negative. The a_in of the degree_receiver_x_y neuron is
    # : the incoming spike_once_x weights+rand_x neurons+selector_excitation
    # - There are at most n incoming spike signals.
    # - Each spike_once should have a weight of at least random_ceiling+1.
    # That is because the random value should map to 0<rand<1 with respect
    # to the difference of 1 spike_once more or less.
    # - The random_ceiling is specified.
    # - The excitatory neuron comes in at +1, a buffer of 1 yields+2.
    # Hence, the inhibition is computed as:
    test_object.inhibition = (
        len(test_object.G) * (test_object.rand_ceil * test_object.delta + 1)
        + (test_object.rand_ceil) * test_object.delta
        + 1
    )
    test_object.rand_nrs = [
        x - test_object.inhibition for x in test_object.rand_nrs
    ]
    print(
        f"After inhibition of:{test_object.inhibition}, rand_nrs="
        + f"{test_object.rand_nrs}"
    )

    # Convert the fully connected graph into a networkx graph that
    # stores the snn properties.
    # rand_ceil+1 because the maximum random number is rand_ceil which should
    # map to range 0<rand<1 when divided by the synaptic weight of spike_once
    # neurons. (and not to range 0<rand<=1 as it would without the +1).
    test_object.get_degree = get_degree_graph_with_separate_wta_circuits(
        test_object.G,
        test_object.rand_nrs,
        test_object.rand_ceil * test_object.delta + 1,
        m,
    )
    test_object.mdsa_graph = copy.deepcopy(test_object.get_degree)

    # Inject radiation
    dead_neuron_names = rad_dam.inject_simulated_radiation(
        test_object.get_degree, rad_dam.neuron_death_probability
    )
    test_object.first_rad_damage_graph = copy.deepcopy(test_object.get_degree)
    test_object.first_dead_neuron_names = copy.deepcopy(dead_neuron_names)

    if plot_snn_graph or export:
        plot_coordinated_graph(
            test_object.get_degree, iteration, len(G), plot_snn_graph
        )

    # Convert the snn networkx graph into a Loihi SNN if no adapted
    # version is generated.
    if not adaptation:
        (
            test_object.converted_nodes,
            test_object.lhs_neuron,
            test_object.neurons,
            test_object.lhs_node,
            test_object.neuron_dict,
        ) = convert_networkx_graph_to_snn_with_one_neuron(
            test_object.get_degree
        )

    return test_object, dead_neuron_names


def get_degree_receiver_previous_property_dicts(
    test_object, degree_receiver_neurons
):
    degree_receiver_previous_us = {}
    degree_receiver_previous_vs = {}
    degree_receiver_previous_has_spiked = {}
    (
        degree_receiver_previous_us,
        degree_receiver_previous_vs,
        degree_receiver_previous_has_spiked,
    ) = fill_dictionary(
        test_object.neuron_dict,
        degree_receiver_neurons,
        degree_receiver_previous_us,
        degree_receiver_previous_vs,
        previous_has_spiked=degree_receiver_previous_has_spiked,
    )

    return (
        degree_receiver_previous_has_spiked,
        degree_receiver_previous_us,
        degree_receiver_previous_vs,
    )


def get_selector_previous_property_dicts(test_object, selector_neurons):
    selector_previous_a_in = {}
    selector_previous_us = {}
    selector_previous_vs = {}
    (
        selector_previous_a_in,
        selector_previous_us,
        selector_previous_vs,
    ) = fill_dictionary(
        test_object.neuron_dict,
        selector_neurons,
        selector_previous_us,
        selector_previous_vs,
        selector_previous_a_in,
    )
    return selector_previous_a_in, selector_previous_us, selector_previous_vs


def get_counter_previous_property_dicts(test_object, counter_neurons):
    counter_previous_a_in = {}
    counter_previous_us = {}
    counter_previous_vs = {}
    (
        counter_previous_a_in,
        counter_previous_us,
        counter_previous_vs,
    ) = fill_dictionary(
        test_object.neuron_dict,
        counter_neurons,
        counter_previous_us,
        counter_previous_vs,
        counter_previous_a_in,
    )
    return counter_previous_a_in, counter_previous_us, counter_previous_vs


def add_monitor_to_dict(neuron, monitor_dict, sim_time):
    """Creates a dictionary monitors that monitor the outgoing spikes of LIF
    neurons."""
    if type(neuron) != str:
        monitor = Monitor()
        monitor.probe(neuron.out_ports.s_out, sim_time)
        monitor_dict[neuron] = monitor
    return monitor_dict


class Selector_neuron:
    """Creates expected properties of the selector neuron."""

    def __init__(self):
        self.first_name = "selector_0"
        self.bias = 5
        self.du = 0
        self.dv = 1
        self.vth = 4


class Spike_once_neuron:
    """Creates expected properties of the spike_once neuron."""

    def __init__(self):
        self.first_name = "spike_once_0"
        self.bias = 2
        self.du = 0
        self.dv = 0
        self.vth = 1


class Rand_neuron:
    """Creates expected properties of the rand neuron."""

    def __init__(self):
        self.first_name = "rand_0"
        self.bias = 2
        self.du = 0
        self.dv = 0
        self.vth = 1


class Counter_neuron:
    """Creates expected properties of the counter neuron."""

    def __init__(self, G):
        self.first_name = "counter_0"
        self.bias = 0
        self.du = 0
        self.dv = 1
        self.vth = 0


class Degree_receiver:
    """Creates expected properties of the spike_once neuron."""

    def __init__(self):
        self.first_name = "spike_once_0"
        self.bias = 0
        self.du = 0
        self.dv = 1
        self.vth = 1


class Test_properties:
    """Contains test parameters."""

    def __init__(self):
        pass
