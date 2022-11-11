"""Contains a default run configuration used to test the MDSA algorithm."""

import json

import jsons
from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA

from snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from snncompare.export_results.export_json_results import (
    encode_tuples,
    write_dict_to_json,
)
from snncompare.helper import file_exists


def store_experiment_config_to_file(exp_config: dict, filename: str) -> None:
    """Verifies the experiment setting and then exports it to a dictionary."""

    # Verify the experiment experiment_config are complete and valid.
    # pylint: disable=R0801
    verify_experiment_config(
        Supported_experiment_settings(),
        exp_config,
        has_unique_id=False,
        allow_optional=False,
    )

    # epxort to file.
    filepath = (
        f"src/snncompare/exp_setts/custom_setts/exp_setts/{filename}.json"
    )
    new_dict = encode_tuples(exp_config)
    write_dict_to_json(filepath, jsons.dump(new_dict))


def load_experiment_config_from_file(filename: str) -> dict:
    """Loads an experiment config from file, then verifies and returns it."""
    filepath = (
        f"src/snncompare/exp_setts/custom_setts/exp_setts/{filename}.json"
    )
    if file_exists(filepath):
        with open(filepath, encoding="utf-8") as json_file:
            encoded_exp_config = json.load(json_file)
            exp_config = encode_tuples(encoded_exp_config, decode=True)

        # Verify the experiment experiment_config are complete and valid.
        # pylint: disable=R0801
        verify_experiment_config(
            Supported_experiment_settings(),
            exp_config,
            has_unique_id=False,
            allow_optional=False,
        )
        return exp_config
    raise FileNotFoundError(f"Error, {filepath} was not found.")


def experiment_config_for_mdsa_testing() -> dict:
    """Contains a default experiment configuration used to test the MDSA
    algorithm."""
    # Create prerequisites
    # supp_exp_setts = Supported_experiment_settings()

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    mdsa_creation_only_size_3_4: dict = {
        "adaptations": None,
        # TODO: set using a verification setting.
        "algorithms": {
            "MDSA": get_algo_configs(MDSA(list(range(0, 1, 1))).__dict__)
        },
        "iterations": list(range(0, 1, 1)),
        # TODO: Change into list with "Seeds"
        "seed": 7,
        # TODO: merge into: "input graph properties object
        # TODO: include verification."
        "min_max_graphs": 1,
        "max_max_graphs": 2,
        "min_graph_size": 3,
        "max_graph_size": 4,
        # "size_and_max_graphs": [(3, 1), (4, 3)],
        # "size_and_max_graphs": [(3, 1),(4, 1)],
        "size_and_max_graphs": [(3, 1), (4, 1)],
        # Move into "overwrite options"
        "overwrite_snn_creation": True,
        "overwrite_snn_propagation": True,
        "overwrite_visualisation": True,
        "overwrite_sim_results": True,
        "radiations": None,
        # TODO: pass algo to see if it is compatible with the algorithm.
        # TODO: move into "Backend options"
        "simulators": ["nx"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }
    return mdsa_creation_only_size_3_4
