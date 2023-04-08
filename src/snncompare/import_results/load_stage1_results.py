"""Parses the graph json files to recreate the graphs."""

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
