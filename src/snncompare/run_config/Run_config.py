""""Stores the run config Dict type."""
import copy
import sys
from pprint import pprint
from typing import Dict, Optional, Union

from snnadaptation.Adaptation import Adaptation
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
class Neuron_death:
    """Adaptation example."""

    @typechecked
    def __init__(self, probability: float):
        """Creates a new NeuronDeathProbability object, an object that
        represents a probability of neuron death in a network.

        Args:
        :probability: (float), The probability of neuron death.
        Returns:
        A probability of neuron death object with the specified
        probability.
        """

        if probability < 0:
            raise ValueError(
                "Error, neuron death probability should be 0 or larger."
            )
        if probability > 1:
            raise ValueError(
                "Error, neuron death probability should be 1 or smaller."
            )
        self.probability = probability


# pylint: disable=R0902
# pylint: disable=R0903
class Run_config:
    """Stores the run configuration object."""

    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        adaptation: Adaptation,
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
        self.adaptation: Union[None, Adaptation] = adaptation
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

    @typechecked
    def print_run_config_dict(self) -> None:
        """Converts a run_config to a human readable dict and prints it."""
        run_config_dict: Dict = copy.deepcopy(self.__dict__)
        run_config_dict["radiation"] = self.radiation.__dict__
        run_config_dict["adaptation"] = self.adaptation.__dict__
        pprint(run_config_dict)


@typechecked
def run_configs_are_equal(*, left: Run_config, right: Run_config) -> bool:
    """Returns True if the left and right Run_config objects are equal. Returns
    False otherwise.

    TODO: Test function
    """
    if sorted(left.__dict__.keys()) != sorted(right.__dict__.keys()):
        return False
    for key, left_value in left.__dict__.items():
        if key in ["adaptation", "radiation"]:
            if left_value.get_hash() != right.__dict__[key].get_hash():
                return False
        elif key != "unique_id":
            if left_value != right.__dict__[key]:
                return False
    return True


@typechecked
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
