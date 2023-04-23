"""Parses the graph json files to recreate the graphs."""

from typeguard import typechecked

from snncompare.run_config.Run_config import Run_config


@typechecked
def get_run_config_filepath(
    *,
    run_config: Run_config,
) -> str:
    """Returns the run_config filepath as it is to be exported."""

    relative_output_dir = "results/stage1/run_configs/"
    json_filepath = relative_output_dir + run_config.unique_id + ".json"
    return json_filepath
