"""Contains experiment settings."""
# pylint: disable=R0801
from src.Supported_settings import Supported_settings

supported_settings = Supported_settings()

without_adaptation: dict = {
    "none": [],
}
without_radiation: dict = {
    "none": [],
}

with_adaptation = {
    "redundancy": [
        1.0,
    ],
}
with_radiation = {
    "neuron_death": [
        0.01,
        0.05,
        0.1,
        0.2,
        0.25,
    ],
}

with_and_without_adaptation = {
    "none": [],
    "redundancy": [
        1.0,
    ],
}
with_and_without_radiation = {
    "none": [],
    "neuron_death": [
        0.01,
        0.05,
        0.1,
        0.2,
        0.25,
    ],
}


with_adaptation_with_radiation = {
    "m": list(range(0, 1, 1)),
    "iterations": list(range(0, 3, 1)),
    "size,max_graphs": [(3, 15), (4, 15)],
    "adaptation": supported_settings.verify_config_setting(
        with_adaptation, "adaptation"
    ),
    "radiation": supported_settings.verify_config_setting(
        with_radiation, "radiation"
    ),
    "overwrite": True,
    "simulators": ["nx"],
}
