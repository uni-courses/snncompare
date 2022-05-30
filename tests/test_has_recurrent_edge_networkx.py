# -*- coding: utf-8 -*-
"""Performs tests that verify the behaviour of a recurrent edge in the
network."""

import unittest

from src.get_graph import get_networkx_graph_of_2_neurons
from src.run_on_lava import simulate_network_on_lava
from src.run_on_networkx import run_snn_on_networkx, simulate_snn_on_networkx


class Test_get_graph_on_networkx(unittest.TestCase):
    """Tests whether a graph has a recurrent edge.

    A recurrent edge is an edge that is connected to itself.
    """

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_returns_2_nodes(self):
        """Tests whether the get_networkx_graph_of_2_neurons function returns a
        graph with two nodes."""

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()
        self.assertEqual(len(G), 2)

        # Assert the graph has only 1 edge.
        self.assertEqual(len(G.edges), 1)

        # Assert the edge goes from node 0 to node 1.
        self.assertEqual(set(G.edges), {(0, 1)})

        # Add the recurrent edge.
        G.add_edge(0, 0)

        # Assert the graph has 2 edges.
        self.assertEqual(len(G.edges), 2)

        # Assert the edges go from node 0 to node 1, and from 0 to 0.
        self.assertEqual(set(G.edges), {(0, 0), (0, 1)})

    def test_verify_recurrent_edge_without_weight_throws_error(self):
        """Creates an SNN consisting of 2 neurons, and verifies their behaviour
        over time.

        Then does the same again yet with a recurrent connection in the
        first neuron.
        """

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()

        # Verify the graph can run on Networkx.
        simulate_snn_on_networkx(G, 30)

        # Verify the graph can run on Lava.
        run_snn_on_networkx(G, 2)

        # Add the recurrent edge.
        G.add_edge(0, 0)

        with self.assertRaises(Exception) as context:
            # Verify running on Networkx throws error.
            simulate_snn_on_networkx(G, 30)

        self.assertEqual(
            "Not all synapse properties of edge: (0, 0) are specified. It only"
            + " contains attributes:dict_keys([])",
            str(context.exception),
        )

        with self.assertRaises(Exception) as context:
            # Verify running on Lava throws error.
            simulate_network_on_lava(G, 2)

        print(f"str(context.exception)={str(context.exception)}")
        self.assertEqual(
            "Not all synapse properties of edge: (0, 0) are specified. It only"
            + " contains attributes:dict_keys([])",
            str(context.exception),
        )

    def test_neuron_properties_after_1_sec_without_recurrent_connection(self):
        """Creates an SNN consisting of 2 neurons, and verifies their behaviour
        over time.

        Then does the same again yet with a recurrent connection in the
        first neuron.
        """

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()

        # Assert static properties of neuron 0.
        self.assertEqual(G.nodes[0]["nx_LIF"].bias.get(), 3)
        self.assertEqual(G.nodes[0]["nx_LIF"].du.get(), 0.0)
        self.assertEqual(G.nodes[0]["nx_LIF"].dv.get(), 0.0)
        self.assertEqual(G.nodes[0]["nx_LIF"].vth.get(), 2.0)

        # Assert static properties of neuron 1.
        self.assertEqual(G.nodes[1]["nx_LIF"].bias.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].du.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].dv.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].vth.get(), 10)

        # Assert dynaic properties of neuron 0 at t=0.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[0]["nx_LIF"].v.get(), 0)

        # Assert dynaic properties of neuron 1 at t=0.
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

        # Simulate network for 1 timestep.
        # simulate_snn_on_networkx(G, 1)
        run_snn_on_networkx(G, 1)

        # Assert dynaic properties of neuron 0 at t=1.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(
            G.nodes[0]["nx_LIF"].v.get(), 0
        )  # Spikes, reset to 0.

        # Assert dynaic properties of neuron 1 at t=1.
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

    def test_neuron_properties_without_recurrent_connection(self):
        """Creates an SNN consisting of 2 neurons, and verifies their behaviour
        over time.

        Then does the same again yet with a recurrent connection in the
        first neuron.
        """

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()

        # Assert static properties of neuron 0.
        self.assertEqual(G.nodes[0]["nx_LIF"].bias.get(), 3)
        self.assertEqual(G.nodes[0]["nx_LIF"].du.get(), 0.0)
        self.assertEqual(G.nodes[0]["nx_LIF"].dv.get(), 0.0)
        self.assertEqual(G.nodes[0]["nx_LIF"].vth.get(), 2.0)

        # Assert static properties of neuron 1.
        self.assertEqual(G.nodes[1]["nx_LIF"].bias.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].du.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].dv.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].vth.get(), 10)

        # Assert dynaic properties of neuron 0 at t=0.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[0]["nx_LIF"].v.get(), 0)

        # Assert dynaic properties of neuron 1 at t=0.
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

        # Simulate network for 1 timestep.
        run_snn_on_networkx(G, 1)
        # Assert dynamic properties of neuron 0 at t=1.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(
            G.nodes[0]["nx_LIF"].v.get(), 0
        )  # Spikes, reset to 0.

        # Assert dynamic properties of neuron 1 at t=1.
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

        # TODO: assert dynamic properties per timestep.
        run_snn_on_networkx(G, 1)
        # Assert dynamic properties of neuron 0 at t=2.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(
            G.nodes[0]["nx_LIF"].v.get(), 0
        )  # Spikes, reset to 0.

        # Assert dynamic properties of neuron 1 at t=2.
        self.assertEqual(
            G.nodes[1]["nx_LIF"].u.get(), 6
        )  # Incoming spike with weight 6.
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

        # TODO: assert dynamic properties per timestep.
        run_snn_on_networkx(G, 1)
        # Assert dynamic properties of neuron 0 at t=3.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(
            G.nodes[0]["nx_LIF"].v.get(), 0
        )  # Spikes, reset to 0.

        # Assert dynamic properties of neuron 1 at t=3.
        # spike of t=2 has arrived at a_in, yielding: u(t=3)=u(t=2)*(1-du)+a_in
        # u(t=3)=6*(1-0)+6=12
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 12)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

    def test_neuron_properties_with_recurrent_connection(self):
        """Creates an SNN consisting of 2 neurons, adds an inhibitory recurrent
        synapse to the excitatory neuron (at node 0), and verifies the SNN
        behaviour over time.

        Then does the same again yet with a recurrent connection in the
        first neuron.
        """

        # Get graph without edge to self.
        G = get_networkx_graph_of_2_neurons()

        # Add the recurrent edge.
        G.add_edge(0, 0, weight=-20.0)

        # Assert static properties of neuron 0 at t=0.
        self.assertEqual(G.nodes[0]["nx_LIF"].bias.get(), 3)
        self.assertEqual(G.nodes[0]["nx_LIF"].du.get(), 0.0)
        self.assertEqual(G.nodes[0]["nx_LIF"].dv.get(), 0.0)
        self.assertEqual(G.nodes[0]["nx_LIF"].vth.get(), 2.0)

        # Assert static properties of neuron 1 at t=0.
        self.assertEqual(G.nodes[1]["nx_LIF"].bias.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].du.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].dv.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].vth.get(), 10)

        # Assert dynaic properties of neuron 0 at t=0.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[0]["nx_LIF"].v.get(), 0)

        # Assert dynaic properties of neuron 1 at t=0.
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

        # Simulate network for 1 timestep.
        run_snn_on_networkx(G, 1)

        # Assert dynamic properties of neuron 0 at t=1 (after 1 sec of sim).
        # spikes but recurrent inhibitory spike did not arrive yet.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), 0)
        self.assertEqual(
            G.nodes[0]["nx_LIF"].v.get(), 0
        )  # Spikes, reset to 0.

        # Assert dynamic properties of neuron 1 at t=1 (after 1 sec of sim).
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 0)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

        # Simulate network for 1 timestep.
        run_snn_on_networkx(G, 1)

        # Assert dynamic properties of neuron 0 at t=2.
        # spikes only the previous inhibitory spike has arrived.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), -20)
        self.assertEqual(
            G.nodes[0]["nx_LIF"].v.get(), 0
        )  # Spikes, reset to 0.

        # Assert dynamic properties of neuron 1 at t=2.
        self.assertEqual(
            G.nodes[1]["nx_LIF"].u.get(), 6
        )  # Incoming spike with weight 6.
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)

        # TODO: assert dynamic properties per timestep.
        run_snn_on_networkx(G, 1)
        # Assert dynamic properties of neuron 0 at t=3.
        self.assertEqual(G.nodes[0]["nx_LIF"].u.get(), -40)
        self.assertEqual(
            G.nodes[0]["nx_LIF"].v.get(), 0
        )  # Spikes, reset to 0.

        # Assert dynamic properties of neuron 1 at t=3.
        self.assertEqual(G.nodes[1]["nx_LIF"].u.get(), 12)
        self.assertEqual(G.nodes[1]["nx_LIF"].v.get(), 0)
