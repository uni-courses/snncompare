"""Performs tests that verify lava simulation produces the same results as the
networkx simulation."""
from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
from networkx.classes.digraph import DiGraph
from snnbackends.lava.run_on_lava import (
    add_lava_neurons_to_networkx_graph,
    simulate_snn_on_lava,
)
from snnbackends.networkx.LIF_neuron import print_neuron_properties_per_graph
from snnbackends.networkx.run_on_networkx import (
    add_nx_neurons_to_networkx_graph,
    run_snn_on_networkx,
)
from snnbackends.networkx.verify_graph_is_networkx_snn import (
    assert_no_duplicate_edges_exist,
    assert_synaptic_edgeweight_type_is_correct,
)
from snnbackends.verify_graph_is_snn import verify_networkx_snn_spec
from typeguard import typechecked

from tests.exp_setts.unsorted.test_scope import Long_scope_of_tests
from tests.simulation.test_rand_network_propagation import (
    Test_propagation_with_recurrent_edges,
)
from tests.simulation.tests_simulation_helper import (
    get_graph_for_cyclic_propagation,
)
from tests.tests_helper import (
    compare_static_snn_properties,
    get_cyclic_graph_without_directed_path,
)


class Test_cyclic_propagation_with_recurrent_edges(unittest.TestCase):
    """Performs tests that verify lava simulation produces the same results as
    the networkx simulation.

    Note, if you get error: "Too many files", you can run in your CLI:
    ulimit -n 800000 to circumvent it.
    """

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        # self.test_scope = Scope_of_tests()
        self.test_scope = Long_scope_of_tests(export=True, show=False)

    @typechecked
    def test_random_networks_are_propagated_the_same_on_networkx_and_lava(
        self,
    ) -> None:
        """Tests whether the random_snn_networks that are generated yield the
        same behaviour on lava as on networkx."""
        # pylint: disable=R1702
        # This test tests all combinations of properties, hence 6 nested for
        # loops are considered acceptable in this case.
        # Get graph without edge to self.

        # Generate cyclic graph.
        # TODO: move this into a separate function that returns a fresh new
        # graph at every timestep of simulation.
        # Then ensure the graphs is simulated for t timesteps at each step of
        # the simulation.

        for _ in np.arange(
            self.test_scope.min_recurrent_edge_density,
            self.test_scope.max_recurrent_edge_density,
            self.test_scope.recurrent_edge_density_stepsize,
        ):
            size = len(get_cyclic_graph_without_directed_path())
            # Ensure the simulation works for all starter neurons.
            for starter_neuron in range(size):
                G = get_graph_for_cyclic_propagation(self.test_scope)

                # pylint: disable=R0801
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
                verify_networkx_snn_spec(G, t=0, backend="lava")

                # Generate networkx network.
                add_nx_neurons_to_networkx_graph(G, t=0)

                # Generate lava network.
                add_lava_neurons_to_networkx_graph(G, t=0)

                # Verify the simulations produce identical static
                # neuron properties.
                compare_static_snn_properties(self, G, t=0)

                print_neuron_properties_per_graph(G, True, t=0)

                # plot_circular_graph(
                #    -1,
                #    G,
                #    recurrent_density,
                #    self.test_scope,
                # )

            run_simulation_for_t_steps(
                self, G, starter_neuron, sim_duration=20
            )

    @typechecked
    def compare_dynamic_snn_properties(self, G: DiGraph, t: int) -> None:
        """Performs comparison of static neuron properties at each timestep.

        :param G: The original graph on which the MDSA algorithm is ran.
        """
        for node in G.nodes:
            lava_neuron = G.nodes[node]["lava_LIF"]
            nx_neuron = G.nodes[node]["nx_LIF"][t]

            # Assert u is equal.
            self.assertEqual(lava_neuron.u.get(), nx_neuron.u.get())

            # Assert v is equal.
            self.assertEqual(lava_neuron.v.get(), nx_neuron.v.get())


def run_simulation_for_t_steps(
    test_object: (
        Test_propagation_with_recurrent_edges
        | Test_cyclic_propagation_with_recurrent_edges
    ),
    G: DiGraph,
    starter_neuron: int,
    sim_duration: int = 20,
) -> None:
    """Runs the SNN simulation on a graph for t timesteps."""
    for t in range(sim_duration):
        G = get_graph_for_cyclic_propagation(test_object.test_scope)
        print(f"t={t}")
        print("")
        # Run the simulation on networkx.
        run_snn_on_networkx(G, t)

        # Run the simulation on lava.
        simulate_snn_on_lava(G, starter_neuron, t)

        print(f"After t={t+1} simulation steps.")
        print_neuron_properties_per_graph(G, False, t)
        # Verify dynamic neuron properties.
        test_object.compare_dynamic_snn_properties(G, t)

        # Terminate Loihi simulation.
        G.nodes[starter_neuron]["lava_LIF"].stop()
