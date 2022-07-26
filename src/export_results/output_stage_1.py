"""Outputs the experiment config, run config and graphs of stage 1, as a
dictionary to a .json file.

Then it outputs the graphs as plots.
"""

from src.export_results.export_json_results import write_dict_to_json
from src.export_results.Output import (
    merge_experiment_and_run_config_with_graphs,
)


def output_stage_1_json(
    experiment_config: dict,
    filename: str,
    run_config: dict,
    graphs_stage_1: dict,
) -> None:
    """Merges the experiment config, run config and graphs of stage 1 into a
    single dict and exports that dict to a json file."""
    # TODO: include stage index
    output_dict = merge_experiment_and_run_config_with_graphs(
        experiment_config, run_config, graphs_stage_1
    )
    # run_config["stage_1_graphs"] = graphs_stage_1

    # TODO: Optional: ensure output files exists.
    output_filepath = f"results/stage_1/{filename}.json"
    write_dict_to_json(output_filepath, output_dict)

    # TODO: Ensure output file exists.
    # TODO: Verify the correct graphs is passed by checking the graph tag.
    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write experiment_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.
