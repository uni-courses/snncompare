"""Contains a default run configuration used to test the MDSA algorithm."""

import json
from typing import Dict

import jsons
from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.exp_config.run_config.Run_config import Run_config
from snncompare.exp_config.run_config.Supported_run_settings import (
    Supported_run_settings,
)
from snncompare.exp_config.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from snncompare.exp_config.verify_experiment_settings import verify_exp_config
from snncompare.export_results.export_json_results import (
    encode_tuples,
    write_dict_to_json,
)
from snncompare.helper import file_exists


def store_exp_config_to_file(
    custom_config_path: str, exp_config: Exp_config, filename: str
) -> None:
    """Verifies the experiment setting and then exports it to a dictionary."""

    # Verify the experiment exp_config are complete and valid.
    # pylint: disable=R0801
    verify_exp_config(
        Supported_experiment_settings(),
        exp_config,
        has_unique_id=False,
        allow_optional=False,
    )

    # epxort to file.
    filepath = f"{custom_config_path}{filename}.json"
    new_dict = encode_tuples(exp_config)
    write_dict_to_json(filepath, jsons.dump(new_dict))


def load_exp_config_from_file(custom_config_path: str, filename: str) -> Dict:
    """Loads an experiment config from file, then verifies and returns it."""
    filepath = f"{custom_config_path}{filename}.json"
    if file_exists(filepath):
        with open(filepath, encoding="utf-8") as json_file:
            encoded_exp_config = json.load(json_file)
            exp_config = encode_tuples(encoded_exp_config, decode=True)
            json_file.close()

        # Verify the experiment exp_config are complete and valid.
        # pylint: disable=R0801
        verify_exp_config(
            Supported_experiment_settings(),
            exp_config,
            has_unique_id=False,
            allow_optional=False,
        )
        return exp_config
    raise FileNotFoundError(f"Error, {filepath} was not found.")


def long_exp_config_for_mdsa_testing() -> Dict:
    """Contains a default experiment configuration used to test the MDSA
    algorithm."""
    # Create prerequisites
    # supp_exp_config = Supported_experiment_settings()

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    long_mdsa_testing: Dict = {
        "adaptations": None,
        # TODO: set using a verification setting.
        "algorithms": {
            "MDSA": get_algo_configs(MDSA(list(range(0, 6, 1))).__dict__)
        },
        "iterations": list(range(0, 1, 1)),
        # TODO: Change into list with "Seeds"
        "seed": 7,
        # TODO: merge into: "input graph properties object
        # TODO: include verification."
        "min_max_graphs": 1,
        "max_max_graphs": 2,
        "min_graph_size": 3,
        "max_graph_size": 5,
        # "size_and_max_graphs": [(3, 1), (4, 3)],
        # "size_and_max_graphs": [(3, 1),(4, 1)],
        "size_and_max_graphs": [(3, 1), (4, 3), (5, 6)],
        # Move into "overwrite options"
        "recreate_s1": True,
        "recreate_s2": True,
        "overwrite_images_only": True,
        "recreate_s4": True,
        "radiations": None,
        # TODO: pass algo to see if it is compatible with the algorithm.
        # TODO: move into "Backend options"
        "simulators": ["nx"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }

    verify_exp_config(
        Supported_experiment_settings(),
        long_mdsa_testing,
        has_unique_id=False,
        allow_optional=True,
    )
    return long_mdsa_testing


def minimal_mdsa_test_exp_config() -> Dict:
    """Returns a experiment config for minimal MDSA testing."""
    minimal_mdsa_testing = long_exp_config_for_mdsa_testing()
    minimal_mdsa_testing["size_and_max_graphs"] = [(3, 1)]
    minimal_mdsa_testing["algorithms"] = (
        {"MDSA": get_algo_configs(MDSA(list(range(0, 1, 1))).__dict__)},
    )
    return minimal_mdsa_testing


def short_mdsa_test_exp_config() -> Dict:
    """Returns a experiment config for short MDSA testing."""
    short_mdsa_testing = long_exp_config_for_mdsa_testing()
    short_mdsa_testing["size_and_max_graphs"] = [(3, 1), (5, 1)]
    short_mdsa_testing["algorithms"] = (
        {"MDSA": get_algo_configs(MDSA(list(range(0, 2, 1))).__dict__)},
    )
    return short_mdsa_testing


def run_config_with_error() -> Run_config:
    """Returns run_config for which error is found."""
    some_run_config: Run_config = Run_config(
        adaptation=None,
        algorithm={"MDSA": {"m_val": 0}},
        export_images=False,
        graph_nr=2,
        graph_size=5,
        iteration=0,
        recreate_s4=True,
        overwrite_images_only=True,
        radiation=None,
        seed=7,
        show_snns=False,
        simulator="nx",
    )
    Supported_run_settings().append_unique_run_config_id(
        some_run_config, allow_optional=True
    )
    return some_run_config


def get_exp_config_mdsa_size5_m4() -> Dict:
    """Returns a default experiment setting with  graph size 7, m=4."""
    mdsa_creation_only_size_7_m_4: Dict = long_exp_config_for_mdsa_testing()
    mdsa_creation_only_size_7_m_4["algorithms"] = {
        "MDSA": get_algo_configs(MDSA(list(range(4, 5, 1))).__dict__)
    }
    mdsa_creation_only_size_7_m_4["size_and_max_graphs"] = [(5, 1)]
    return mdsa_creation_only_size_7_m_4
