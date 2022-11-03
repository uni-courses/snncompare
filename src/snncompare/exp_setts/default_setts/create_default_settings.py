"""Used to generate the default experiment configuration json, run
configuration settings, default input graph and default input graphs."""

import jsons
import networkx as nx

from src.snncompare.arg_parser.arg_verification import verify_input_graph_path
from src.snncompare.exp_setts.adapt.Adaptation_Rad_settings import (
    Adaptations_settings,
    Radiation_settings,
)
from src.snncompare.exp_setts.algos.get_alg_configs import get_algo_configs
from src.snncompare.exp_setts.algos.MDSA import MDSA
from src.snncompare.exp_setts.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.snncompare.exp_setts.verify_experiment_settings import (
    verify_adap_and_rad_settings,
)
from src.snncompare.export_results.export_json_results import (
    write_dict_to_json,
)
from src.snncompare.export_results.export_nx_graph_to_json import (
    digraph_to_json,
)
from src.snncompare.graph_generation.Used_graphs import Used_graphs


def create_default_graph_json() -> None:
    """Generates a default input graph and exports it to a json file."""
    used_graphs = Used_graphs()
    default_nx_graph: nx.DiGraph = used_graphs.five[0]

    # Convert nx.DiGraph to dict.
    default_json_graph = digraph_to_json(default_nx_graph)

    graphs_json_filepath = (
        "src/snncompare/exp_setts/default_setts/default_graph_MDSA.json"
    )
    write_dict_to_json(graphs_json_filepath, jsons.dump(default_json_graph))

    # Verify file exists and that it has a valid content.
    verify_input_graph_path(graphs_json_filepath)

    # Verify file content.


def create_default_exp_setts() -> None:
    """Generates the default experiment settings json file."""


def default_experiment_config() -> dict:
    """Creates example experiment configuration setting."""
    # Create prerequisites
    supp_exp_setts = Supported_experiment_settings()
    adap_sets = Adaptations_settings()
    rad_sets = Radiation_settings()

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    with_adaptation_with_radiation = {
        # TODO: set using a verification setting.
        "algorithms": {
            "MDSA": get_algo_configs(MDSA(list(range(0, 1, 1))).__dict__)
        },
        # TODO: pass algo to see if it is compatible with the algorithm.
        "adaptations": verify_adap_and_rad_settings(
            supp_exp_setts, adap_sets.with_adaptation, "adaptations"
        ),
        "radiations": verify_adap_and_rad_settings(
            supp_exp_setts, rad_sets.with_radiation, "radiations"
        ),
        # "iterations": list(range(0, 3, 1)),
        "iterations": list(range(0, 1, 1)),
        # TODO: Change into list with "Seeds"
        "seed": 42,
        # TODO: merge into: "input graph properties object
        # TODO: include verification."
        "min_max_graphs": 1,
        "max_max_graphs": 15,
        "min_graph_size": 3,
        "max_graph_size": 20,
        # "size_and_max_graphs": [(3, 1), (4, 3)],
        "size_and_max_graphs": [(3, 1)],
        # Move into "overwrite options"
        "overwrite_snn_creation": False,
        "overwrite_snn_propagation": False,
        "overwrite_visualisation": False,
        "overwrite_sim_results": False,
        # TODO: pass algo to see if it is compatible with the algorithm.
        # TODO: move into "Backend options"
        "simulators": ["nx"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }
    return with_adaptation_with_radiation
