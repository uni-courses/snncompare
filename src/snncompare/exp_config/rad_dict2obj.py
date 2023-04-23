"""Converts radiation damage dicts from Exp_config dict, into a list Rad_damage
objects."""
from math import inf
from typing import Dict, List, Union

from snnradiation.Rad_damage import Rad_damage

# from snncompare.export_results.load_json_to_nx_graph import dicts_are_equal
from typeguard import typechecked


@typechecked
def get_radiations_from_exp_config_dict(
    *,
    radiations: Dict[
        str, Dict[str, Union[str, List[Union[bool, int, float]]]]
    ],
) -> List[Rad_damage]:
    """Converts the experiment settings radiation dictionaries into list of
    Rad_damage objects.

    Each Rad_damage object contains the settings for 1 run_config.
    """
    radiation_objs: List[Rad_damage] = []
    for effect_type, rad_settings in radiations.items():
        print(f"rad_settings={rad_settings}")
        if effect_type == "neuron_death":
            for probability_per_t in rad_settings["probability_per_t"]:
                rad_damage = Rad_damage(
                    amplitude=inf,
                    effect_type=effect_type,
                    excitatory=False,
                    inhibitory=True,
                    probability_per_t=probability_per_t,
                )
                radiation_objs.append(rad_damage)
        else:
            add_generic_exp_config_rad_dict(
                effect_type=effect_type,
                rad_settings=rad_settings,
                radiation_objs=radiation_objs,
            )
    return radiation_objs


@typechecked
def add_generic_exp_config_rad_dict(
    *,
    effect_type: str,
    rad_settings: Dict[str, Union[str, List[Union[bool, int, float]]]],
    radiation_objs: List[Rad_damage],
) -> None:
    """Converts the experiment settings radiation dictionaries into list of
    Rad_damage objects.

    Each Rad_damage object contains the settings for 1 run_config.
    """

    for amplitude in rad_settings["amplitude"]:
        for excitatory in rad_settings["excitatory"]:
            for inhibitory in rad_settings["inhibitory"]:
                for probability_per_t in rad_settings["probability_per_t"]:
                    rad_damage = Rad_damage(
                        amplitude=amplitude,
                        effect_type=effect_type,
                        excitatory=excitatory,
                        inhibitory=inhibitory,
                        probability_per_t=probability_per_t,
                    )
                radiation_objs.append(rad_damage)
