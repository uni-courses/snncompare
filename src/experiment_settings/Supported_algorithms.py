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

        # List of the algorithm parameters for an experiment settings dict.
        self.algo_parameters = ["m_vals"]

        # List of the algorithm parameters for a run settings dict.
        self.algo_parameters = ["m_val"]

        # List with values for m. m is the number of iterations for which the
        # Alipour approximation is ran.
        self.m_vals = list(range(0, 1, 1))

        # A values for m for a specific run setting.
        self.m_val = 0
