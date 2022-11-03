"""Contains the supported experiment settings.

(The values of the settings may vary, yet the values of an experiment
setting should be within the ranges specified in this file, and the
setting types should be identical.)
"""


import copy

from src.snncompare.exp_setts.algos.get_alg_configs import get_algo_configs
from src.snncompare.exp_setts.algos.MDSA import MDSA
from src.snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
    verify_min_max,
)


# pylint: disable=R0903
# The settings object contains all the settings as a dictionary, hence no
# hierarchy is used, leading to 10/7 instance attributes.
class Exp_setts_typing:
    """Stores the supported experiment setting parameter ranges.

    An experiment can consist of multiple runs. A run is a particular
    combination of experiment setting parameters.
    """

    def __init__(
        self,
    ) -> None:

        # experiment_config dictionary keys:
        self.parameters = {
            "adaptations": dict,
            "algorithms": dict,
            "iterations": list,
            "max_graph_size": int,
            "max_max_graphs": int,
            "min_graph_size": int,
            "min_max_graphs": int,
            "neuron_models": list,
            "overwrite_snn_creation": bool,
            "overwrite_snn_propagation": bool,
            "overwrite_visualisation": bool,
            "overwrite_sim_results": bool,
            "radiations": dict,
            "seed": int,
            "simulators": list,
            "size_and_max_graphs": list,
            "synaptic_models": list,
        }
        self.optional_parameters = {
            "show_snns": bool,
            "export_images": bool,
            "unique_id": int,
        }


# pylint: disable=R0902
class Supported_experiment_settings:
    """Contains the settings that are supported for the experiment_config."""

    def __init__(
        self,
    ) -> None:
        self.parameters = Exp_setts_typing().parameters
        self.optional_parameters = Exp_setts_typing().optional_parameters

        self.seed = 5

        # Create dictionary with algorithm name as key, and algorithm settings
        # object as value.
        self.algorithms = get_algo_configs(MDSA(list(range(0, 4, 1))).__dict__)

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
        self.overwrite_sim_results = True
        # Overwrite the visualisation of the SNN behaviour or not.
        self.overwrite_visualisation = True

        # The backend/type of simulator that is used.
        self.simulators = ["nx", "lava"]

        # Generate the supported adaptation settings.
        self.specify_supported_adaptation_settings()

        # Generate the supported radiations settings.
        self.specify_supported_radiations_settings()

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

    def has_unique_config_id(self, experiment_config: dict) -> bool:
        """

        :param experiment_config:

        """
        if "unique_id" in experiment_config.keys():
            return True
        return False

    def append_unique_config_id(self, experiment_config: dict) -> dict:
        """Checks if an experiment configuration dictionary already has a
        unique identifier, and if not it computes and appends it.

        If it does, throws an error.

        :param experiment_config: dict:
        """
        if "unique_id" in experiment_config.keys():
            raise Exception(
                f"Error, the experiment_config:{experiment_config}\n"
                + "already contains a unique identifier."
            )

        verify_experiment_config(
            self, experiment_config, has_unique_id=False, strict=True
        )

        # Compute a unique code belonging to this particular experiment
        # configuration.
        hash_set = dict_to_frozen_set(experiment_config)

        unique_id = hash(hash_set)
        experiment_config["unique_id"] = unique_id
        verify_experiment_config(
            self, experiment_config, has_unique_id=True, strict=False
        )
        return experiment_config


def dict_to_frozen_set(experiment_config: dict) -> frozenset:
    """Converts a dictionary into a frozenset, such that a hash code of the
    dict can be computed."""
    some_dict = copy.deepcopy(experiment_config)
    for value in some_dict.values():
        if isinstance(value, dict):
            value = dict_to_frozen_set(value)
    return frozenset(some_dict)
