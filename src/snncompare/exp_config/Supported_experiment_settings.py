"""Contains the supported experiment settings.

(The values of the settings may vary, yet the values of an experiment
setting should be within the ranges specified in this file, and the
setting types should be identical.)
"""
import copy
import hashlib
import json
from typing import TYPE_CHECKING, Dict

from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from typeguard import typechecked

from snncompare.exp_config.verify_experiment_settings import (
    verify_exp_config,
    verify_min_max,
)
from snncompare.helper import remove_optional_args_exp_config

if TYPE_CHECKING:
    from snncompare.exp_config.Exp_config import Exp_config


# pylint: disable=R0902
class Supported_experiment_settings:
    """Contains the settings that are supported for the exp_config."""

    @typechecked
    def __init__(
        self,
    ) -> None:

        self.seed = 5

        # Create dictionary with algorithm name as key, and algorithm settings
        # object as value.
        mdsa_min = MDSA([]).min_m_vals
        mdsa_max = MDSA([]).max_m_vals
        self.algorithms = get_algo_configs(
            MDSA(list(range(mdsa_min, mdsa_max, 1))).__dict__
        )

        # The number of times the experiment is repeated.
        self.iterations = list(range(0, 3, 1))

        # Specify the maximum number of: (maximum number of graphs per run
        # size).
        self.min_max_graphs = 1
        self.max_max_graphs = 15
        verify_min_max(self.min_max_graphs, self.max_max_graphs)

        # Specify the maximum graph size.
        self.min_graph_size = 3
        self.max_graph_size = 20
        verify_min_max(self.min_graph_size, self.max_graph_size)

        # The size of the graph and the maximum number of used graphs of that
        # size.
        self.size_and_max_graphs = [
            (self.min_graph_size, self.max_max_graphs),
            (5, 4),  # Means: get 4 graphs of size 5 for experiment.
            (self.max_graph_size, self.max_max_graphs),
        ]

        # Overwrite the simulation results or not.
        self.recreate_s4 = True
        # Overwrite the visualisation of the SNN behaviour or not.
        self.overwrite_images_only = True

        # The backend/type of simulator that is used.
        self.simulators = ["nx", "lava"]

        # Generate the supported adaptation settings.
        self.specify_supported_adaptation_settings()

        # Generate the supported radiations settings.
        self.specify_supported_radiations_settings()

        # Specify the supported image export file extensions.
        self.export_types = ["pdf", "png"]

    @typechecked
    def specify_supported_radiations_settings(self) -> None:
        """Specifies types of supported radiations settings. Some settings
        consist of a list of tuples, with the probability of a change
        occurring, followed by the average magnitude of the change.

        Others only contain a list of floats which represent the
        probability of radiations induced change occurring.
        """
        # List of tuples with x=probabiltity of change, y=average value change
        # in synaptic weights.
        self.delta_synaptic_w = [
            (0.01, 0.5),
            (0.05, 0.4),
            (0.1, 0.3),
            (0.2, 0.2),
            (0.25, 0.1),
        ]

        # List of tuples with x=probabiltity of change, y=average value change
        # in neuronal threshold.
        self.delta_vth = [
            (0.01, 0.5),
            (0.05, 0.4),
            (0.1, 0.3),
            (0.2, 0.2),
            (0.25, 0.1),
        ]

        # Create a supported radiations setting example.
        self.radiations = {
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
            "delta_synaptic_w": self.delta_synaptic_w,
            # List of: (probability of neuron threshold change, and the average
            # factor with which it changes due to radiations).
            "delta_vth": self.delta_vth,
        }

    @typechecked
    def specify_supported_adaptation_settings(self) -> None:
        """Specifies all the supported types of adaptation settings."""

        # Specify the (to be) supported adaptation types.
        self.adaptations = {
            "None": [],
            "redundancy": [
                1.0,
            ],  # Create 1 redundant neuron per neuron.
            "population": [
                10.0
            ],  # Create a population of 10 neurons to represent a
            # single neuron.
            "rate_coding": [
                5.0
            ],  # Multiply firing frequency with 5 to limit spike decay
            # impact.
        }

    @typechecked
    def has_unique_config_id(self, some_config: "Exp_config") -> bool:
        """

        :param exp_config:

        """
        if "unique_id" in some_config.keys():
            return True
        return False

    @typechecked
    def append_unique_exp_config_id(
        self,
        exp_config: "Exp_config",
        allow_optional: bool = True,
    ) -> Dict:
        """Checks if an experiment configuration dictionary already has a
        unique identifier, and if not it computes and appends it.

        If it does, throws an error.

        :param exp_config: Exp_config:
        """
        if "unique_id" in exp_config.keys():
            raise Exception(
                f"Error, the exp_config:{exp_config}\n"
                + "already contains a unique identifier."
            )

        verify_exp_config(
            self,
            exp_config,
            has_unique_id=False,
            allow_optional=allow_optional,
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
        exp_config["unique_id"] = unique_id
        verify_exp_config(
            self,
            exp_config,
            has_unique_id=True,
            allow_optional=allow_optional,
        )
        return exp_config
