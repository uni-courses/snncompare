"""Contains the supported parameter ranges of the algorithms on which the
experiment can be ran."""


# pylint: disable=R0903
class MDSA:
    """Minimum Dominating Set Approximation by Alipour.

    The Graph Size
    """

    def __init__(
        self,
    ) -> None:

        # List of the parameters of this algorithm.
        self.algo_parameters = ["m_vals"]

        # The number of iterations for which the Alipour approximation is ran.
        self.m_vals = list(range(0, 1, 1))
