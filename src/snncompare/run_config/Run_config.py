""""Stores the run config Dict type."""
import sys
from typing import Dict, Optional, Tuple, Union

from snnradiation.Rad_damage import Rad_damage
from typeguard import typechecked

from snncompare.export_results.helper import get_unique_run_config_id

if sys.version_info < (3, 11):
    from typing_extensions import TypedDict
else:
    pass


# pylint: disable=R0903
class Algorithm(TypedDict):
    """Example typed dict to make the property types explicit."""

    alg_name: Dict[str, int]


# pylint: disable=R0903
class Adaptation:
    """Adaptations."""

    @typechecked
    def __init__(
        self,
        name: str,
        int_param: Optional[int],
    ):
        if name == "Redundancy":
            self.name = name
            self.adaptation = Redundancy(int_param)
        else:
            raise TypeError("Error, adaptation:{name} not yet supported.")


# pylint: disable=R0903
class Redundancy:
    """Adaptation example."""

    @typechecked
    def __init__(self, redundancy: int):
        if redundancy < 2:
            raise ValueError("Error, redundancy should be 2 or larger.")
        if redundancy % 2 == 1:
            raise ValueError("Error, redundancy should be even integer.")
        self.redundancy = redundancy


# pylint: disable=R0903
class Neuron_death:
    """Adaptation example."""

    @typechecked
    def __init__(self, probability: float):
        if probability < 0:
            raise ValueError(
                "Error, neuron death probability should be 0 or larger."
            )
        if probability > 1:
            raise ValueError(
                "Error, neuron death probability should be 1 or smaller."
            )
        self.probability = probability


class Radiation(TypedDict):
    """Example typed dict to make the property types explicit."""

    # Permanent effects
    permanent_neuron_death: float
    permanent_synapse_death: float

    # Neuron property changes
    permanent_bias_change: float
    permanent_du_change: float
    permanent_dv_change: float
    permanent_vth_change: float

    # Synaptic property changes.
    permanent_weight_change: float

    # Transient effects [duration, absolute change in value]
    temp_neuron_death: Tuple[int, float]
    temp_synapse_death: Tuple[int, float]

    # Neuron property changes
    temp_bias_change: Tuple[int, float]
    temp_du_change: Tuple[int, float]
    temp_dv_change: Tuple[int, float]
    temp_vth_change: Tuple[int, float]

    # Synaptic property changes.
    temp_weight_change: Tuple[float]


# pylint: disable=R0902
# pylint: disable=R0903
class Run_config:
    """Stores the run configuration object."""

    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        adaptation: Union[None, Dict[str, int]],
        algorithm: Dict[str, Dict[str, int]],
        graph_size: int,
        graph_nr: int,
        radiation: Rad_damage,
        seed: int,
        simulator: str,
        max_duration: Optional[int] = None,
    ):
        """Stores run configuration settings for the experiment."""

        # Required properties
        self.adaptation: Union[None, Dict[str, int]] = adaptation
        self.algorithm: Dict[str, Dict[str, int]] = algorithm

        self.graph_size: int = graph_size
        self.graph_nr: int = graph_nr
        self.radiation: Rad_damage = radiation
        self.seed: int = seed
        self.simulator: str = simulator

        # Optional properties

        self.max_duration: Optional[int] = max_duration

        self.unique_id: str = get_unique_run_config_id(run_config=self)

        # TODO: Verify run config object.


def dict_to_run_config(*, some_dict: Dict) -> Run_config:
    """Converts a dict into a Run_config object."""
    run_config: Run_config = Run_config(
        adaptation=some_dict["adaptation"],
        algorithm=some_dict["algorithm"],
        graph_size=some_dict["graph_size"],
        graph_nr=some_dict["graph_nr"],
        radiation=some_dict["radiation"],
        seed=some_dict["seed"],
        simulator=some_dict["simulator"],
    )

    # TODO: only do optional values.
    for key, val in some_dict.items():
        setattr(run_config, key, val)
    return run_config
