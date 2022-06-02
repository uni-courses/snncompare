"""File represents LIF neuron object."""


class Scope_of_tests:
    """Stores the ranges over which the random graphs and SNNs are created for
    testing purposes."""

    # pylint: disable=R0903
    # pylint: disable=too-many-instance-attributes
    # Eleven is considered is reasonable in this case.

    def __init__(self) -> None:

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

        # Specify random seed.
        self.seed = 42
