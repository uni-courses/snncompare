"""Contains the supported parameter ranges of the algorithms on which the
experiment can be ran."""


# pylint: disable=R0903
class MDSA:
    """Minimum Dominating Set Approximation by Alipour.

    TODO: determine whether self.m_val should be an undefined parameter because
    if follows from the iteration over self.mvals.
    """

    def __init__(
        self,
    ) -> None:

        # List of the algorithm parameters for a run settings dict.
        self.algo_parameters = ["m_val"]

        # List with values for m. m is the number of iterations for which the
        # Alipour approximation is ran.
        self.m_vals = list(range(0, 4, 1))

        # A values for m for a specific run setting.
        self.m_val = 0
