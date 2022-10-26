"""Methods used to verify the json graphs."""
from typing import List


def verify_json_graphs_dict_contain_correct_stages(
    json_graphs: dict, expected_stages: List[int], run_config: dict
) -> None:
    """Verifies the json graphs dict contains the expected stages in each
    graph."""
    for expected_stage in expected_stages:
        for graph_name, graph in json_graphs.items():
            #            print(f'graph={graph}')
            print(f"{graph_name}, type={type(graph)}")
            for elem in graph:
                print(type(elem))
            if graph["graph"]["completed_stages"]:
                if expected_stage not in graph["graph"]["completed_stages"]:
                    raise ValueError(
                        "Error, for run_config: "
                        + f"{run_config}, the expected "
                        + f"stage:{expected_stage}, was not found in "
                        + "the completed stages:"
                        + f'{graph["graph"]["completed_stages"]} '
                        + f"that were loaded from graph: {graph_name}."
                    )


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
