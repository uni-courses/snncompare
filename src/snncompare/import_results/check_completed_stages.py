"""Method used to perform checks on whether the input is loaded correctly."""
from typing import Dict, List

from snnbackends.verify_nx_graphs import verify_completed_stages_list
from typeguard import typechecked

from snncompare.run_config.Run_config import Run_config

from ..export_results.verify_stage_1_graphs import (
    get_expected_stage_1_graph_names,
)


@typechecked
def has_outputted_stage_jsons(
    *,
    expected_stages: List[int],
    run_config: Run_config,
    stage_index: int,
) -> bool:
    """Checks if a stage has been outputted or not.

    TODO: delete?
    """
    print(expected_stages)
    print(run_config)
    print(stage_index)
    return True


@typechecked
def nx_graphs_have_completed_stage(
    *,
    run_config: Run_config,
    results_nx_graphs: Dict,
    stage_index: int,
) -> bool:
    """Checks whether all expected graphs have been completed for the stages:

    <stage_index>. This check is performed by loading the graph dict
    from the graph Dict, and checking whether the graph dict contains a
    list with the completed stages, and checking this list whether it
    contains the required stage number.
    """

    # Loop through expected graph names for this run_config.
    for graph_name in get_expected_stage_1_graph_names(run_config=run_config):
        graph = results_nx_graphs["graphs_dict"][graph_name]
        if graph_name not in results_nx_graphs["graphs_dict"]:
            return False
        if ("completed_stages") not in graph.graph:
            return False
        if not isinstance(
            graph.graph["completed_stages"],
            List,
        ):
            raise TypeError(
                "Error, completed stages parameter type is not a list."
            )
        if stage_index not in graph.graph["completed_stages"]:
            return False
        verify_completed_stages_list(
            completed_stages=graph.graph["completed_stages"]
        )
    return True
