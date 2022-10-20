"""Methods used to verify the graphs.

TODO: merge with from src.import_results.check_completed_stages
"""
from typing import List

from src.export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)


def verify_loaded_results_from_json(run_result: dict, run_config: dict):
    """Verifies the results that are loaded from json file are of the expected
    format."""
    stage_1_graph_names = get_expected_stage_1_graph_names(run_config)
    # Verify the 3 dicts are in the result dict.
    if "experiment_config" not in run_result.keys():
        raise Exception(
            f"Error, experiment_config not in run_result keys:{run_result}"
        )

    if "run_config" not in run_result.keys():
        raise Exception(
            f"Error, run_config not in run_result keys:{run_result}"
        )
    if "graphs_dict" not in run_result.keys():
        raise Exception(
            f"Error, graphs_dict not in run_result keys:{run_result}"
        )

    # Verify the right graphs are within the graphs_dict.
    for graph_name in stage_1_graph_names:
        if graph_name not in run_result["graphs_dict"].keys():
            raise Exception(
                f"Error, {graph_name} not in run_result keys:{run_result}"
            )

    # Verify each graph has the right completed stages attribute.
    for graph_name in run_result["graphs_dict"].keys():
        verify_completed_stages_list(
            run_result["graphs_dict"][graph_name].graph["completed_stages"]
        )
        # TODO: verify the stage index is in completed_stages.


def verify_completed_stages_list(completed_stages: List) -> None:
    """Verifies the completed stages list is a list of consecutive positive
    integers.

    TODO: test this function.
    """
    start_stage = completed_stages[0]
    for next_stage in completed_stages[1:]:
        if start_stage != next_stage - 1:
            raise Exception(
                f"Stage indices are not consecutive:{completed_stages}."
            )
        start_stage = next_stage
    for stage in completed_stages:
        if stage < 1:
            raise Exception(
                "completed_stages contained non positive integer:"
                + f"{completed_stages}"
            )
