"""Performs tests that verify lava simulation produces the same results as the
networkx simulation."""

import unittest

import networkx as nx
import numpy as np

from src.get_graph import (
    get_cyclic_graph_without_directed_path,
    set_rand_neuron_properties,
)
from src.LIF_neuron import print_neuron_properties_per_graph
from src.plot_graphs import plot_circular_graph
from src.run_on_lava import (
    add_lava_neurons_to_networkx_graph,
    simulate_snn_on_lava,
)
from src.run_on_networkx import (
    add_nx_neurons_to_networkx_graph,
    run_snn_on_networkx,
)
from src.Scope_of_tests import Long_scope_of_tests
from src.verify_graph_is_snn import (
    assert_no_duplicate_edges_exist,
    assert_synaptic_edgeweight_type_is_correct,
    verify_networkx_snn_spec,
)


class Test_propagation_with_recurrent_edges(unittest.TestCase):
    """Performs tests that verify lava simulation produces the same results as
    the networkx simulation."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.test_scope = Scope_of_tests()
        self.test_scope = Long_scope_of_tests(export=True, show=False)

    def test_random_networks_are_propagated_the_same_on_networkx_and_lava(
        self,
    ):
        """Tests whether the random_snn_networks that are generated yield the
        same behaviour on lava as on networkx."""
        # pylint: disable=R1702
        # This test tests all combinations of properties, hence 6 nested for
        # loops are considered acceptable in this case.
        # Get graph without edge to self.

        # Generate cyclic graph.
        G = get_cyclic_graph_without_directed_path()
        set_rand_neuron_properties(G, self.test_scope)
        size = len(G)

        for recurrent_density in np.arange(
            self.test_scope.min_recurrent_edge_density,
            self.test_scope.max_recurrent_edge_density,
            self.test_scope.recurrent_edge_density_stepsize,
        ):
            # Ensure the simulation works for all starter neurons.
            for starter_neuron in range(size):

                # Assert graph is connected.
                # self.assertTrue(nx.is_connected(G))
                self.assertFalse(
                    not nx.is_strongly_connected(G)
                    and not nx.is_weakly_connected(G)
                )

                # Assert size of graph.
                self.assertEqual(size, len(G))

                # Assert each edge has a weight.
                for edge in G.edges:

                    assert_synaptic_edgeweight_type_is_correct(G, edge)

                # Assert no duplicate edges exist.
                assert_no_duplicate_edges_exist(G)

                # Assert all neuron properties are specified.
                verify_networkx_snn_spec(G)

                # Generate networkx network.
                add_nx_neurons_to_networkx_graph(G)

                # Generate lava network.
                add_lava_neurons_to_networkx_graph(G)

                # Verify the simulations produce identical static
                # neuron properties.
                print("")
                compare_static_snn_properties(self, G)

                print_neuron_properties_per_graph(G, True)

                plot_circular_graph(
                    -1,
                    G,
                    recurrent_density,
                    self.test_scope,
                )

            run_simulation_for_t_steps(
                self, G, starter_neuron, sim_duration=20
            )

    def compare_dynamic_snn_properties(self, G):
        """Performs comparison of static neuron properties at each timestep.

        :param G: The original graph on which the MDSA algorithm is ran.
        """
        for node in G.nodes:
            lava_neuron = G.nodes[node]["lava_LIF"]
            nx_neuron = G.nodes[node]["nx_LIF"]

            # Assert u is equal.
            self.assertEqual(lava_neuron.u.get(), nx_neuron.u.get())

            # Assert v is equal.
            self.assertEqual(lava_neuron.v.get(), nx_neuron.v.get())


def run_simulation_for_t_steps(
    test_object, G, starter_neuron, sim_duration=20
):
    """Runs the SNN simulation on a graph for t timesteps."""
    for t in range(sim_duration):
        print("")
        # Run the simulation on networkx.
        run_snn_on_networkx(G, 1)

        # Run the simulation on lava.
        simulate_snn_on_lava(G, starter_neuron, 1)

        print(f"After t={t+1} simulation steps.")
        print_neuron_properties_per_graph(G, False)
        # Verify dynamic neuron properties.
        test_object.compare_dynamic_snn_properties(G)

    # Terminate Loihi simulation.
    G.nodes[starter_neuron]["lava_LIF"].stop()


def compare_static_snn_properties(test_object, G):
    """Performs comparison of static neuron properties at each timestep.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    for node in G.nodes:
        lava_neuron = G.nodes[node]["lava_LIF"]
        nx_neuron = G.nodes[node]["nx_LIF"]

        # Assert bias is equal.
        test_object.assertEqual(lava_neuron.bias.get(), nx_neuron.bias.get())

        # dicts
        # print(f"lava_neuron.__dict__={lava_neuron.__dict__}")
        # print(f"lava_neuron.__dict__={nx_neuron.__dict__}")

        # Assert du is equal.
        test_object.assertEqual(lava_neuron.du.get(), nx_neuron.du.get())
        #

        # Assert dv is equal.
        test_object.assertEqual(lava_neuron.dv.get(), nx_neuron.dv.get())

        # print(f"lava_neuron.name.get()={lava_neuron.name.get()}")
        # print(f"lava_neuron.name.get()={nx_neuron.name.get()}")
        # Assert name is equal.
        # self.assertEqual(lava_neuron.name, nx_neuron.name)

        # Assert vth is equal.
        test_object.assertEqual(lava_neuron.vth.get(), nx_neuron.vth.get())

        # Assert v_reset is equal. (Not yet implemented in Lava.)
        # self.assertEqual(
        #    lava_neuron.v_reset.get(), nx_neuron.v_reset.get()
        # )
