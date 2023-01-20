""""Stores the run config Dict type."""
import copy
import hashlib
import json
from typing import Dict, List, Optional, Union

from typeguard import typechecked

from snncompare.exp_config.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_config.verify_experiment_settings import verify_exp_config


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
        supp_exp_config = Supported_experiment_settings()
        verify_exp_config(
            supp_exp_config=supp_exp_config,
            exp_config=self,
            has_unique_id=False,
            allow_optional=False,
        )


@typechecked
def remove_optional_args_exp_config(
    supported_experiment_settings: "Supported_experiment_settings",
    copied_exp_config: "Exp_config",
) -> "Exp_config":
    """removes the optional arguments from a run config."""
    non_unique_attributes = [
        "recreate_s1",
        "recreate_s2",
        "overwrite_images_only",
        "recreate_s4",
        "export_images",
        "export_types",
        "unique_id",
    ]
    for attribute_name in non_unique_attributes:
        # TODO: set to default value instead
        setattr(copied_exp_config, attribute_name, None)
    verify_exp_config(
        supp_exp_config=supported_experiment_settings,
        exp_config=copied_exp_config,
        has_unique_id=False,
        allow_optional=False,
    )
    return copied_exp_config


@typechecked
def append_unique_exp_config_id(
    exp_config: "Exp_config",
) -> "Exp_config":
    """Checks if an experiment configuration dictionary already has a unique
    identifier, and if not it computes and appends it.

    If it does, throws an error.

    :param exp_config: Exp_config:
    """
    if "unique_id" in exp_config.__dict__.keys():
        raise Exception(
            f"Error, the exp_config:{exp_config}\n"
            + "already contains a unique identifier."
        )

    # Compute a unique code belonging to this particular experiment
    # configuration.
    # TODO: remove optional arguments from config.
    supported_experiment_settings = Supported_experiment_settings()
    exp_config_without_unique_id: "Exp_config" = (
        remove_optional_args_exp_config(
            supported_experiment_settings=supported_experiment_settings,
            copied_exp_config=copy.deepcopy(exp_config),
        )
    )

    unique_id = str(
        hashlib.sha256(
            json.dumps(exp_config_without_unique_id).encode("utf-8")
        ).hexdigest()
    )
    exp_config.unique_id = unique_id
    return exp_config
