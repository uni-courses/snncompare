"""Code to load and parse the simulation results dict consisting of the
experiment config, run config and json graphs, from a json dict.

Appears to also be used to partially convert json graphs back into nx
graphs.
"""
import json
from pathlib import Path
from typing import Dict, Optional

from typeguard import typechecked

from snncompare.import_results.json_dict_into_nx_snn import (
    load_json_graph_to_snn,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def load_results_from_json(
    *,
    json_filepath: str,
    run_config: Run_config,
) -> Dict:
    """Loads the results from a json file, and then converts the graph dicts
    back into a nx.DiGraph object."""
    # Load the json dictionary of results.
    results_loaded_graphs: Dict = load_json_file_into_dict(
        json_filepath=json_filepath
    )

    # Verify the dict contains a key for the graph dict.
    if "graphs_dict" not in results_loaded_graphs:
        raise KeyError(
            "Error, the graphs dict key was not in the stage_1_Dict:"
            + f"{results_loaded_graphs}"
        )

    # Verify the graphs dict is of type dict.
    if results_loaded_graphs["graphs_dict"] == {}:
        raise ValueError("Error, the graphs dict was an empty dict.")

    # TODO: call load snn
    load_json_graph_to_snn(
        run_config=run_config, json_graphs=results_loaded_graphs["graphs_dict"]
    )
    return results_loaded_graphs


@typechecked
def load_json_file_into_dict(
    *,
    json_filepath: str,
) -> Dict[str, Optional[Dict]]:
    """TODO: make this into a private function that cannot be called by
    any other object than some results loader.
    Loads a json file into dict from a filepath."""
    if not Path(json_filepath).is_file():
        raise FileNotFoundError("Error, filepath does not exist:{filepath}")
    # TODO: verify extension.
    # TODO: verify json formatting is valid.
    with open(json_filepath, encoding="utf-8") as json_file:
        the_dict = json.load(json_file)
        json_file.close()
    return the_dict
