"""Contains the specification of and maximum values of the algorithm
settings."""


from typing import List

from src.snncompare.exp_setts.algos.algo_helper import assert_parameter_is_list


# pylint: disable=R0903
class MDSA_config:
    """Create a particular configuration for the MDSA algorithm."""

    def __init__(self, mdsa_config: dict) -> None:

        for some_property, value in mdsa_config.items():
            if some_property == "m_vals":
                # Verify type of parameters
                if not isinstance(value, int):
                    raise TypeError(
                        "m_val is not of type:int. Instead it is of "
                        + f"type:{type(value)}"
                    )

                # List of the algorithm parameters for a run settings dict.
                self.m_val = value
            else:
                raise KeyError(
                    f"Error, the key:{some_property} is not supported "
                    "for the MDSA configuration."
                )


# pylint: disable=R0903
class MDSA:
    """Specification of algorithm specification. Algorithm: Minimum Dominating
    Set Approximation by Alipour.

    Example usage: default_MDSA_alg=MDSA(m_vals=list(range(0, 4, 1)))
    """

    def __init__(
        self,
        m_vals: List[int],
    ) -> None:
        self.name = "MDSA"
        self.min_m_vals: int = 0
        self.max_m_vals: int = 3

        self.verify_m_vals(m_vals)

        # List of the algorithm parameters for a run settings dict.
        self.alg_parameters = {"m_vals": m_vals}

    def verify_m_vals(self, m_vals: List[int]) -> None:
        """Verifies the m_vals parameter setting of the algorithm."""
        assert_parameter_is_list(m_vals)

        # Verify values of parameters.
        for m_val in m_vals:
            # Verify type of parameters
            if not isinstance(m_val, int):
                raise TypeError(
                    "m_val is not of type:int. Instead it is of "
                    + f"type:{type(m_val)}"
                )
            if m_val < self.min_m_vals:
                raise ValueError(
                    "Error, the minimum supported value for m_vals is:"
                    + f"{self.min_m_vals}, yet we found:{m_vals}"
                )
            if m_val > self.max_m_vals:
                raise ValueError(
                    "Error, the maximum supported value for m_vals is:"
                    + f"{self.min_m_vals}, yet we found:{m_vals}"
                )
