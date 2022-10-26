"""Methods used to verify the json graphs."""
from typing import List


def verify_results_json_graphs_contain_correct_stages(
    results_json_graphs: dict, expected_stages: List[int]
) -> None:
    """Checks whether the loaded graphs from json contain at least the expected
    stages for this stage of the experiment."""

    if "experiment_config" not in results_json_graphs:
        raise KeyError("Error, key: experiment_config not in output_dict.")
    if "run_config" not in results_json_graphs:
        raise KeyError("Error, key: run_config not in output_dict.")
    if "graphs_dict" not in results_json_graphs:
        raise KeyError("Error, key: graphs_dict not in output_dict.")
    verify_json_graphs_dict_contain_correct_stages(
        results_json_graphs["graphs_dict"],
        expected_stages,
        results_json_graphs["run_config"],
    )


def verify_json_graphs_dict_contain_correct_stages(
    json_graphs: dict, expected_stages: List[int], run_config: dict
) -> None:
    """Verifies the json graphs dict contains the expected stages in each
    graph."""
    for expected_stage in expected_stages:
        for graph_name, json_graph in json_graphs.items():
            if expected_stages[-1] == 1:
                completed_stages = json_graph["graph"]["completed_stages"]

            elif expected_stages[-1] in [2, 4]:
                # TODO: determine why this is a list of graphs, instead of a
                # graph with list of nodes.
                # Completed stages are only stored in the last timestep of the
                # graph.
                if not isinstance(json_graph, List):
                    raise TypeError(
                        "Error, the json_graph is of type:"
                        f"{type(json_graph)}, with content:"
                    )
                # The non input SNN graphs are lists of graphs, 1 per
                # timestep, so get the completed stages property from the
                # last one of that list.
                completed_stages = json_graph[-1]["graph"]["completed_stages"]
            else:
                raise Exception(
                    f"Error, stage:{expected_stages[-1]} is "
                    "not yet supported in this check."
                )
            if expected_stage not in completed_stages:
                raise ValueError(
                    "Error, for run_config: "
                    + f"{run_config}, the expected "
                    + f"stage:{expected_stage}, was not found in "
                    + "the completed stages:"
                    + f"{completed_stages} "
                    + f"that were loaded from graph: {graph_name}."
                )
