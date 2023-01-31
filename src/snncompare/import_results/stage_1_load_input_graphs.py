"""Parses the graph json files to recreate the graphs."""
from pprint import pprint
from typing import Dict

from typeguard import typechecked

from snncompare.exp_config.run_config.Run_config import Run_config

from ..export_results.helper import run_config_to_filename
from ..export_results.verify_stage_1_graphs import assert_graphs_are_in_dict
from .read_json import load_results_from_json


@typechecked
def load_results_stage_1(
    *,
    run_config: Run_config,
) -> Dict:
    """Loads the experiment config, run config and graphs from the json file.

    # TODO: ensure it only loads the graphs of stage 1. OR: make all
    dict loading the same.
    """
    # Get the json filename.
    filename = run_config_to_filename(run_config=run_config)
    relative_output_dir = "results/"
    json_filepath = relative_output_dir + filename + ".json"
    stage_1_dict = load_results_from_json(
        json_filepath=json_filepath, run_config=run_config
    )

    # Split the dictionary into three separate dicts.
    # The ** loads the dict into the object.
    stage_1_dict["run_config"] = Run_config(**stage_1_dict["run_config"])

    # Verify the run_dict is valid.
    if run_config.unique_id != stage_1_dict["run_config"].unique_id:
        print("Current run_config:")
        pprint(run_config.__dict__)
        print("Loaded run_config:")
        pprint(stage_1_dict["run_config"].__dict__)
        raise Exception("Error, difference in run configs, see above.")

    # Verify the graph names are as expected for the graph name.
    assert_graphs_are_in_dict(
        run_config=run_config, graphs=stage_1_dict["graphs_dict"], stage=1
    )

    return stage_1_dict
