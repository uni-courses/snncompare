""""Stores the run config Dict type."""
import sys
from typing import Dict, List, Optional, Tuple, Union

from typeguard import typechecked

if sys.version_info < (3, 11):
    from typing_extensions import TypedDict
else:
    pass


class Algorithm(TypedDict):
    """Example typed dict to make the property types explicit."""

    alg_name: Dict[str, int]


class Adaptation(TypedDict):
    """Example typed dict to make the property types explicit."""

    redundancy: int
    population: int


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
        iteration: int,
        graph_size: int,
        graph_nr: int,
        radiation: Union[
            None, Union[Dict[str, float], Dict[str, Tuple[int, float]]]
        ],
        seed: int,
        simulator: str,
        export_images: Optional[bool] = None,
        export_types: Optional[List[str]] = None,
        max_duration: Optional[int] = None,
        recreate_s1: Optional[bool] = None,
        recreate_s2: Optional[bool] = None,
        overwrite_images_only: Optional[bool] = None,
        recreate_s4: Optional[bool] = None,
        show_snns: Optional[bool] = None,
        unique_id: Optional[str] = None,
    ):
        """Stores run configuration settings for the experiment."""

        # Required properties
        self.adaptation: Union[None, Dict[str, int]] = adaptation
        self.algorithm: Dict[str, Dict[str, int]] = algorithm
        self.iteration: int = iteration
        self.graph_size: int = graph_size
        self.graph_nr: int = graph_nr
        self.radiation: Union[
            None, Union[Dict[str, float], Dict[str, Tuple[int, float]]]
        ] = radiation
        self.seed: int = seed
        self.simulator: str = simulator

        # Optional properties
        self.export_images: Optional[bool] = export_images
        self.export_types: Optional[List[str]] = export_types
        self.max_duration: Optional[int] = max_duration
        self.recreate_s1: Optional[bool] = recreate_s1
        self.recreate_s2: Optional[bool] = recreate_s2
        self.overwrite_images_only: Optional[bool] = overwrite_images_only
        self.recreate_s4: Optional[bool] = recreate_s4
        self.show_snns: Optional[bool] = show_snns
        self.unique_id: Optional[str] = unique_id

        # Verify run config object.

        # Compute hash of run config object.


def dict_to_run_config(some_dict: Dict) -> Run_config:
    """Converts a dict into a Run_config object."""
    run_config: Run_config = Run_config(
        adaptation=some_dict["adaptation"],
        algorithm=some_dict["algorithm"],
        iteration=some_dict["iteration"],
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
