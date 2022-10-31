"""Contains the specification of and maximum values of the algorithm
settings."""


from typing import List

from src.snncompare.exp_setts.algos.algo_helper import assert_parameter_is_list


# pylint: disable=R0903
# pylint: disable=R0801
class DUMMY_config:
    """Create a particular configuration for the MDSA algorithm."""

    def __init__(self, dummy_config: dict) -> None:

        for some_property, value in dummy_config.items():
            if some_property == "some_vals":
                # Verify type of parameters
                if not isinstance(value, int):
                    raise TypeError(
                        "some_val is not of type:int. Instead it is of "
                        + f"type:{type(value)}"
                    )

                # List of the algorithm parameters for a run settings dict.
                self.some_val = value
            elif some_property == "other_vals":
                # Verify type of parameters
                if not isinstance(value, str):
                    raise TypeError(
                        "other_vals is not of type:str. Instead it is of "
                        + f"type:{type(value)}"
                    )

                # List of the algorithm parameters for a run settings dict.
                self.other_val = value
            else:
                raise KeyError(
                    f"Error, the key:{some_property} is not supported "
                    "for the MDSA configuration."
                )


# pylint: disable=R0903
class DUMMY:
    """Specification of algorithm specification. Algorithm: Minimum Dominating
    Set Approximation by Alipour.

    Example usage: default_MDSA_alg=MDSA(some_vals=list(range(0, 4, 1)))
    """

    def __init__(
        self,
        some_vals: List[int],
        other_vals: List[str],
    ) -> None:
        self.name = "DUMMY"

        # Specify supported values for some_vals.
        self.min_some_vals: int = 4
        self.max_some_vals: int = 9

        # Specify supported values for ohter_vals.
        self.supported_other_vals: List[str] = ["onestring", "anotherstring"]

        self.verify_some_vals(some_vals)
        self.verify_other_vals(other_vals)

        # List of the algorithm parameters for a run settings dict.
        self.alg_parameters = {
            "some_vals": some_vals,
            "other_vals": other_vals,
        }

    def verify_some_vals(self, some_vals: List[int]) -> None:
        """Verifies the some_vals parameter setting of the algorithm."""
        assert_parameter_is_list(some_vals)

        # Verify values of parameters.
        for some_val in some_vals:
            # Verify type of parameters
            if not isinstance(some_val, int):
                raise TypeError(
                    "some_val is not of type:int. Instead it is of "
                    + f"type:{type(some_val)}"
                )
            if some_val < self.min_some_vals:
                raise ValueError(
                    "Error, the minimum supported value for some_vals is:"
                    + f"{self.min_some_vals}, yet we found:{some_vals}"
                )
            if some_val > self.max_some_vals:
                raise ValueError(
                    "Error, the maximum supported value for some_vals is:"
                    + f"{self.min_some_vals}, yet we found:{some_vals}"
                )

    def verify_other_vals(self, other_vals: List[str]) -> None:
        """Verifies the other_vals parameter setting of the algorithm."""
        assert_parameter_is_list(other_vals)

        # Verify values of parameters.
        for other_val in other_vals:
            # Verify type of parameters
            if not isinstance(other_val, str):
                raise TypeError(
                    "other_val is not of type:str. Instead it is of "
                    + f"type:{type(other_val)}"
                )
            if other_val not in self.supported_other_vals:
                raise ValueError(
                    f"Error, the value:{other_val} is not in the supported "
                    + f"other_vals values: {self.supported_other_vals}"
                )
