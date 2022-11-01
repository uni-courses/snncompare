"""Contains the specification of and maximum values of the algorithm
settings."""


from typing import List

from src.snncompare.exp_setts.algos.verify_algos import (
    verify_list_with_numbers,
)


# pylint: disable=R0903
# pylint: disable=R0801
class Radiation_config:
    """Create a particular configuration for the Radiation configuration."""

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
class Permanent_neuron_death:
    """Specification of the default simulated radiation model: permanent neuron
    death. Neuron death is modelled by setting the threshold voltage to 1000V.
    TODO: set to infinity instead (for backends that support it)
    TODO: also create object that applies this radiation to an incoming SNN.

    TODO: Example usage: default_radiation=Permanent_neuron_death()
    """

    def __init__(
        self,
        death_probability: List[float],
    ) -> None:
        self.name = "permanent_neuron_death"

        # Specify supported values for some_vals.
        self.min_death_probability: float = 0.0
        self.max_death_probability: float = 1.0

        verify_list_with_numbers(
            elem_type=float,
            min_val=self.min_death_probability,
            max_val=self.max_death_probability,
            some_vals=death_probability,
            var_name=self.name,
        )

        # List of the algorithm parameters for a run settings dict.
        self.alg_parameters = {
            "death_probability": death_probability,
        }
