"""Methods used to verify the json graphs."""
import pprint
from typing import Dict, List

from typeguard import typechecked

from snncompare.exp_config.run_config.Run_config import dict_to_run_config


@typechecked
def verify_results_safely_check_json_graphs_contain_expected_stages(
    results_json_graphs: Dict, expected_stages: List[int]
) -> None:
    """Checks whether the loaded graphs from json contain at least the expected
    stages for this stage of the experiment."""

    if "exp_config" not in results_json_graphs:
        raise KeyError("Error, key: exp_config not in output_dict.")
    if "run_config" not in results_json_graphs:
        raise KeyError("Error, key: run_config not in output_dict.")
    if "graphs_dict" not in results_json_graphs:
        raise KeyError("Error, key: graphs_dict not in output_dict.")

    if not isinstance(results_json_graphs["run_config"], Dict):
        raise TypeError(
            "run_config was not of type Dict:"
            + f'{results_json_graphs["run_config"]}'
        )

    results_json_graphs["run_config"] = dict_to_run_config(
        results_json_graphs["run_config"]
    )

    verify_json_graphs_dict_contain_correct_stages(
        results_json_graphs["graphs_dict"],
        expected_stages,
    )


@typechecked
def verify_json_graphs_dict_contain_correct_stages(
    json_graphs: Dict,
    expected_stages: List[int],
) -> None:
    """Verifies the json graphs dict contains the expected stages in each
    graph."""
    for expected_stage in expected_stages:
        for graph_name, json_graph in json_graphs.items():
            if not isinstance(json_graph, Dict):
                raise TypeError(
                    "Error, the json_graph is of type:"
                    f"{type(json_graph)}, with content:{json_graph}"
                )
            completed_stages = json_graph["graph"]["completed_stages"]
            if expected_stage not in completed_stages:
                print(f"expected_stages={expected_stages}")
                print(f"For:{graph_name}, json_graph.graph:")
                # pprint.pformat(json_graph["graph"], indent=4)
                raise ValueError(
                    "Error, for the above run_config, the expected stage:"
                    + f"{expected_stage}, in:"
                    + f"expected_stages={expected_stages},"
                    + "was not found in the completed "
                    + f"stages:{completed_stages} that were loaded from graph"
                    + f": {graph_name}, with json_graph.graph="
                    + pprint.pformat(json_graph["graph"], indent=4)
                )
