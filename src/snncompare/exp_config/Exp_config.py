""""Stores the run config Dict type."""
from typing import Dict, List, Optional, Union

from typeguard import typechecked


# pylint: disable=R0902
# pylint: disable=R0903
class Exp_config:
    """Stores the exp_configriment settings object."""

    # pylint: disable=R0913
    # pylint: disable=R0914
    @typechecked
    def __init__(
        self,
        adaptations: Union[None, Dict[str, int], dict],
        algorithms: Dict[str, List[Dict[str, int]]],
        max_graph_size: int,
        max_max_graphs: int,
        min_graph_size: int,
        min_max_graphs: int,
        neuron_models: List,
        recreate_s1: bool,
        recreate_s2: bool,
        overwrite_images_only: bool,
        recreate_s4: bool,
        radiations: Dict,
        seeds: List[int],
        simulators: List,
        size_and_max_graphs: List,
        synaptic_models: List,
        export_images: Optional[bool] = False,
        export_types: Optional[List[str]] = None,
        unique_id: Optional[str] = None,
    ):
        """Stores run configuration settings for the exp_configriment."""

        # Required properties
        self.adaptations: Union[None, Dict[str, int]] = adaptations
        self.algorithms: Dict[str, List[Dict[str, int]]] = algorithms
        self.max_graph_size: int = max_graph_size
        self.max_max_graphs: int = max_max_graphs
        self.min_graph_size: int = min_graph_size
        self.min_max_graphs: int = min_max_graphs
        self.neuron_models: List = neuron_models
        self.recreate_s1: bool = recreate_s1
        self.recreate_s2: bool = recreate_s2
        self.overwrite_images_only: bool = overwrite_images_only
        self.recreate_s4: bool = recreate_s4
        self.radiations: Dict = radiations
        self.seeds: List[int] = seeds
        self.simulators: List = simulators
        self.size_and_max_graphs: List = size_and_max_graphs
        self.synaptic_models: List = synaptic_models

        # Optional properties
        self.export_images: bool = bool(export_images)
        if self.export_images:
            if export_types is not None:
                self.export_types: List[str] = export_types

        if unique_id is not None:
            self.unique_id: str = unique_id

        # Verify run config object.
