"""Methods used to safely check json graph properties."""
from typing import List

from typeguard import typechecked

from snncompare.exp_setts.run_config.Run_config import Run_config

from .verify_json_graphs import verify_json_graphs_dict_contain_correct_stages


@typechecked
def safely_check_json_graphs_contain_expected_stages(
    json_graphs: dict,
    expected_stages: List[int],
    run_config: Run_config,
) -> bool:
    """Returns True if the json graphs are valid, False otherwise."""
    try:
        verify_json_graphs_dict_contain_correct_stages(
            json_graphs, expected_stages, run_config
        )
        return True
    # pylint: disable=W0702
    except KeyError:
        return False
    except ValueError:
        return False
    except TypeError:
        return False


@typechecked
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
