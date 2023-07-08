"""Contains a default run configuration used to test the MDSA algorithm."""
import json
from typing import Dict

from snnadaptation.Adaptation import Adaptation
from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from snnradiation.Rad_damage import Rad_damage
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_results.export_json_results import encode_tuples
from snncompare.helper import file_exists
from snncompare.run_config.Run_config import Run_config


@typechecked
def load_exp_config_from_file(
    *, custom_config_path: str, filename: str
) -> "Exp_config":
    """Loads an experiment config from file, then verifies and returns it."""
    filepath = f"{custom_config_path}exp_config/{filename}.json"
    if file_exists(filepath=filepath):
        with open(filepath, encoding="utf-8") as json_file:
            encoded_exp_config = json.load(json_file)
            exp_config_dict = encode_tuples(
                some_dict=encoded_exp_config, decode=True
            )
            json_file.close()
        # The ** loads the dict into the object.
        if "unique_id" in exp_config_dict:
            exp_config_dict.pop("unique_id")
        exp_config = Exp_config(**exp_config_dict)
        return exp_config
    raise FileNotFoundError(f"Error, {filepath} was not found.")


@typechecked
def load_run_config_from_file(
    *, custom_config_path: str, filename: str, from_unique_id: bool
) -> "Run_config":
    """Loads a run config from file, then verifies and returns it.

    TODO: Remove duplicate code with load_exp_config_from_file().
    TODO: Allow for calling with run_config file from results/stage1.
    """
    if not from_unique_id:
        filepath = f"{custom_config_path}run_configs/{filename}.json"
        if not file_exists(filepath=filepath):
            raise FileNotFoundError(f"Error, {filepath} was not found.")
    else:
        filepath = f"{filename}.json"
        if not file_exists(filepath=filepath):
            raise FileNotFoundError(f"Error, file:{filepath} not found")

    with open(filepath, encoding="utf-8") as json_file:
        encoded_exp_config = json.load(json_file)
        run_config_dict = encode_tuples(
            some_dict=encoded_exp_config, decode=True
        )
        json_file.close()

    # Convert adaptation_dict into adaptation_object.
    run_config_dict["adaptation"] = Adaptation(**run_config_dict["adaptation"])
    run_config_dict["radiation"] = Rad_damage(**run_config_dict["radiation"])

    # The ** loads the dict into the object.
    if "unique_id" in run_config_dict:
        run_config_dict.pop("unique_id")
    exp_config = Run_config(**run_config_dict)
    return exp_config


@typechecked
def long_exp_config_for_mdsa_testing_without_adaptation() -> "Exp_config":
    """Contains a default experiment configuration used to test the MDSA
    algorithm."""

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    long_mdsa_testing: Dict = {
        "adaptations": None,
        "algorithms": {
            "MDSA": get_algo_configs(
                algo_spec=MDSA(list(range(0, 6, 1))).__dict__
            )
        },
        # TODO: Change into list with "Seeds"
        "seeds": [7],
        # TODO: merge into: "input graph properties object
        # TODO: include verification."
        "min_max_graphs": 1,
        "max_max_graphs": 2,
        "min_graph_size": 3,
        "max_graph_size": 5,
        "size_and_max_graphs": [(3, 1), (4, 3), (5, 6)],
        # Move into "overwrite options"
        "radiations": {},
        # TODO: pass algo to see if it is compatible with the algorithm.
        # TODO: move into "Backend options"
        "simulators": ["nx"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }

    # The ** loads the dict into the object.
    exp_config = Exp_config(**long_mdsa_testing)
    return exp_config


@typechecked
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
        recreate_s3=True,
        radiation=None,
        seed=7,
        simulator="nx",
    )
    return some_run_config
