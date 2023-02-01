"""Used to generate the default experiment configuration json, run
configuration settings, default input graph and default input graphs."""
from typing import Dict

import jsons
import networkx as nx
from snnadaptation.redundancy.Adaptation_Rad_settings import (
    Adaptations_settings,
    Radiation_settings,
)
from snnalgorithms.get_alg_configs import get_algo_configs
from snnalgorithms.sparse.MDSA.alg_params import MDSA
from snnalgorithms.Used_graphs import Used_graphs
from typeguard import typechecked

from snncompare.exp_config.Exp_config import (
    Exp_config,
    Supported_experiment_settings,
    verify_adap_and_rad_settings,
)

from ..arg_parser.arg_verification import verify_input_graph_path
from ..export_results.export_json_results import write_dict_to_json
from ..export_results.export_nx_graph_to_json import digraph_to_json


@typechecked
def create_default_graph_json() -> None:
    """Generates a default input graph and exports it to a json file."""
    used_graphs = Used_graphs()
    default_nx_graph: nx.Graph = used_graphs.five[0]

    # Convert nx.DiGraph to dict.
    default_json_graph = digraph_to_json(G=default_nx_graph)

    graphs_json_filepath = (
        "src/snncompare/json_configurations/default_graph_MDSA.json"
    )
    write_dict_to_json(
        output_filepath=graphs_json_filepath,
        some_dict=jsons.dump(default_json_graph),
    )

    # Verify file exists and that it has a valid content.
    verify_input_graph_path(graph_path=graphs_json_filepath)

    # Verify file content.


@typechecked
def default_exp_config() -> Exp_config:
    """Creates example experiment configuration setting."""
    # Create prerequisites
    supp_exp_config = Supported_experiment_settings()
    adap_sets = Adaptations_settings()
    rad_sets = Radiation_settings()

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    with_adaptation_with_radiation: Dict = {
        # TODO: set using a verification setting.
        "algorithms": {
            "MDSA": get_algo_configs(
                algo_spec=MDSA(list(range(0, 1, 1))).__dict__
            )
        },
        # TODO: pass algo to see if it is compatible with the algorithm.
        "adaptations": verify_adap_and_rad_settings(
            supp_exp_config=supp_exp_config,
            some_dict=adap_sets.with_adaptation,
            check_type="adaptations",
        ),
        "radiations": verify_adap_and_rad_settings(
            supp_exp_config=supp_exp_config,
            some_dict=rad_sets.with_radiation,
            check_type="radiations",
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
        "recreate_s1": False,
        "recreate_s2": False,
        "recreate_s3": False,
        "recreate_s4": False,
        # TODO: pass algo to see if it is compatible with the algorithm.
        # TODO: move into "Backend options"
        "simulators": ["nx"],
        "neuron_models": ["LIF"],
        "synaptic_models": ["LIF"],
    }
    # The ** loads the dict into the object.
    exp_config = Exp_config(**with_adaptation_with_radiation)

    return exp_config
