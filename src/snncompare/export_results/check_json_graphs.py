"""Methods used to safely check json graph properties."""

from typing import List

from src.snncompare.export_results.verify_json_graphs import (
    verify_json_graphs_dict_contain_correct_stages,
)


def json_graphs_contain_correct_stages(
    json_graphs: dict, expected_stages: List[int], run_config: dict
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
