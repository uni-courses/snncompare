"""Converts radiation damage dicts from Exp_config dict, into a list Rad_damage
objects."""
from typing import Dict, List

from snnadaptation.Adaptation import Adaptation

# from snncompare.export_results.load_json_to_nx_graph import dicts_are_equal
from typeguard import typechecked


@typechecked
def get_adaptations_from_exp_config_dict(
    *,
    adaptations: Dict[str, List[int]],
) -> List[Adaptation]:
    """Converts the experiment settings radiation dictionaries into list of
    Rad_damage objects.

    Each Rad_damage object contains the settings for 1 run_config.
    """
    adaptation_objs: List[Adaptation] = []
    for adaptation_type, redundancies in adaptations.items():
        for redundancy in redundancies:
            adaptation_objs.append(
                Adaptation(
                    adaptation_type=adaptation_type,
                    redundancy=redundancy,
                )
            )
    return adaptation_objs
