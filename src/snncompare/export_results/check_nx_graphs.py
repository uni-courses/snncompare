"""Methods used to safely check nx_graph properties."""


from typing import List


def json_graphs_contain_expected_stages(
    json_graphs: dict,
    expected_graph_names: List[str],
    expected_stages: List[int],
) -> bool:
    """Verifies for each of the expected graph names is in the json_graphs
    dict, and then verifies each of those graphs contain the completed
    stages."""
    for expected_graph_name in expected_graph_names:
        if expected_graph_name not in json_graphs.keys():
            return False
        for expected_stage in expected_stages:
            # More completed stages than expected is ok.
            if (
                expected_stage
                not in json_graphs[expected_graph_name]["completed_stages"]
            ):
                return False
    return True
