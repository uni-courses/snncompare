"""Stores the scope of the tests."""
from typeguard import typechecked


class Scope_of_tests:
    """Stores the ranges over which the random graphs and SNNs are created for
    testing purposes."""

    # pylint: disable=R0903
    # pylint: disable=too-many-instance-attributes
    # Eleven is considered is reasonable in this case.

    @typechecked
    def __init__(self, export: bool = True, show: bool = False) -> None:

        self.export = export  # Export the graph that is generated to .png

        # Specify graph size (nr. of nodes in graph).
        self.min_size = 7
        self.max_size = 8

        # Specify ratio of edges that are created w.r.t. fully connected.
        self.min_edge_density = 0.9
        self.max_edge_density = 1
        self.edge_density_stepsize = 0.1

        # Specify random neuron bias range.
        self.min_bias = -10
        self.max_bias = 10

        # Specify random neuron spike threshold voltage range.
        self.min_vth = 0
        self.max_vth = 10
        # Specify random synaptic edge weight range.
        self.min_edge_weight = -12
        self.max_edge_weight = 12

        # Ratio of nodes that have a recurrent edge (to node itself).
        self.min_recurrent_edge_density = 0.0
        self.max_recurrent_edge_density = 1
        self.recurrent_edge_density_stepsize = 0.1

        self.show = show  # Show the graph that is generated.

        # Specify random seed.
        self.seed = 42


class Long_scope_of_tests:
    """Stores the ranges over which the random graphs and SNNs are created for
    testing purposes."""

    # pylint: disable=R0903
    # pylint: disable=too-many-instance-attributes
    # Eleven is considered is reasonable in this case.

    @typechecked
    def __init__(self, export: bool = True, show: bool = False) -> None:

        self.export = export  # Export the graph that is generated to .png

        # Specify graph size (nr. of nodes in graph).
        self.min_size = 2
        self.max_size = 6

        # Specify ratio of edges that are created w.r.t. fully connected.
        self.min_edge_density = 0.6
        self.max_edge_density = 1
        self.edge_density_stepsize = 0.2

        # Specify random neuron bias range.
        self.min_bias = -10
        self.max_bias = 10

        # Specify random neuron spike threshold voltage range.
        self.min_vth = 0
        self.max_vth = 10
        # Specify random synaptic edge weight range.
        self.min_edge_weight = -12
        self.max_edge_weight = 12

        # TODO: change to range instead of number.
        self.recurrent_edge_density = 0.3

        # Ratio of nodes that have a recurrent edge (to node itself).
        self.min_recurrent_edge_density = 0.0
        self.max_recurrent_edge_density = 1
        self.recurrent_edge_density_stepsize = 0.5

        self.show = show  # Show the graph that is generated.

        # Specify random seed.
        self.seed = 42
