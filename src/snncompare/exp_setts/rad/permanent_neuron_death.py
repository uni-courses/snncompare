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


def specify_supported_radiations_settings(test_object):
    """Specifies types of supported radiations settings. Some settings consist
    of a list of tuples, with the probability of a change occurring, followed
    by the average magnitude of the change.

    Others only contain a list of floats which represent the probability
    of radiations induced change occurring.
    """
    # List of tuples with x=probabiltity of change, y=average value change
    # in synaptic weights.
    test_object.delta_synaptic_w = [
        (0.01, 0.5),
        (0.05, 0.4),
        (0.1, 0.3),
        (0.2, 0.2),
        (0.25, 0.1),
    ]

    # List of tuples with x=probabiltity of change, y=average value change
    # in neuronal threshold.
    test_object.delta_vth = [
        (0.01, 0.5),
        (0.05, 0.4),
        (0.1, 0.3),
        (0.2, 0.2),
        (0.25, 0.1),
    ]

    # Create a supported radiations setting example.
    test_object.radiations = {
        # No radiations
        "None": [],
        # radiations effects are transient, they last for 1 or 10
        # simulation steps. If transient is 0., the changes are permanent.
        "transient": [0.0, 1.0, 10.0],
        # List of probabilities of a neuron dying due to radiations.
        "neuron_death": [
            0.01,
            0.05,
            0.1,
            0.2,
            0.25,
        ],
        # List of probabilities of a synapse dying due to radiations.
        "synaptic_death": [
            0.01,
            0.05,
            0.1,
            0.2,
            0.25,
        ],
        # List of: (probability of synaptic weight change, and the average
        # factor with which it changes due to radiations).
        "delta_synaptic_w": test_object.delta_synaptic_w,
        # List of: (probability of neuron threshold change, and the average
        # factor with which it changes due to radiations).
        "delta_vth": test_object.delta_vth,
    }
