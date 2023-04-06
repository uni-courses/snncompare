"""Parses the graph json files to recreate the graphs."""
from typing import Dict

from typeguard import typechecked

from snncompare.run_config.Run_config import Run_config

from ..export_results.helper import run_config_to_filename


@typechecked
def get_run_config_filepath(
    *,
    run_config: Run_config,
) -> str:
    """Returns the run_config filepath as it is to be exported."""
    filename = run_config_to_filename(run_config_dict=run_config.__dict__)
    relative_output_dir = "results/stage1/run_configs/"
    json_filepath = relative_output_dir + filename + ".json"
    return json_filepath


# @typechecked
# def load_results_stage_1(
#     *,
#     run_config: Run_config,
# ) -> Dict:
#     """Loads the experiment config, run config and graphs from the json file.

#     # TODO: ensure it only loads the graphs of stage 1. OR: make all
#     dict loading the same.
#     """
#     stage_1_dict:Dict={}


#     # Get the run_config filename.
#     json_filepath:str=get_run_config_filepath(run_config=run_config)
#     run_config = load_results_from_json(
#         json_filepath=json_filepath, run_config=run_config
#     )

#     # Pop the unique_id if it was in.
#     pop_unique_id(config=run_config)

#     # Split the dictionary into three separate dicts.
#     # The ** loads the dict into the object.
#     stage_1_dict["run_config"] = Run_config(**stage_1_dict["run_config"])

#     # Verify the run_dict is valid.
#     if run_config.unique_id != stage_1_dict["run_config"].unique_id:
#         print("Current run_config:")
#         pprint(run_config.__dict__)
#         print("Loaded run_config:")
#         pprint(stage_1_dict["run_config"].__dict__)
#         raise ValueError("Error, difference in run configs, see above.")

#     # Verify the graph names are as expected for the graph name.
#     assert_graphs_are_in_dict(
#         run_config=run_config,
#         graphs=stage_1_dict["graphs_dict"],
#         stage_index=1,
#     )

#     return stage_1_dict


@typechecked
def pop_unique_id(
    *,
    config: Dict,
) -> str:
    """Removes unique_id, if it was in."""
    if "unique_id" in config:
        unique_id: str = config["unique_id"]
        config.pop("unique_id")
        return unique_id
    return ""
