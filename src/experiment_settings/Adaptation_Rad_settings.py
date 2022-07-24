"""Contains experiment settings."""
# pylint: disable=R0801
from src.experiment_settings.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.experiment_settings.verify_experiment_settings import (
    verify_adap_and_rad_settings,
)


class Adaptations_settings:
    """Stores all used adaptation settings."""

    # pylint: disable=R0903

    def __init__(
        self,
    ) -> None:

        self.without_adaptation: dict = {
            "None": [],
        }

        self.with_adaptation = {
            "redundancy": [
                1.0,
                # 2.0, # TODO: also support
            ],
        }

        self.with_and_without_adaptation = {
            "None": [],
            "redundancy": [
                1.0,
            ],
        }


class Radiation_settings:
    """Stores all used radiation settings."""

    # pylint: disable=R0903

    def __init__(
        self,
    ) -> None:
        self.without_radiation: dict = {
            "None": [],
        }
        self.with_radiation = {
            "neuron_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
        }
        self.with_and_without_radiation = {
            "None": [],
            "neuron_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
        }


adaptation_settings = Adaptations_settings()
radiation_settings = Radiation_settings()
supported_settings = Supported_experiment_settings()

with_and_without_adaptation_and_radiation = {
    "m": list(range(0, 1, 1)),
    "iterations": list(range(0, 3, 1)),
    "size,max_graphs": [(3, 15), (4, 15)],
    "adaptations": verify_adap_and_rad_settings(
        supported_settings,
        adaptation_settings.with_and_without_adaptation,
        "adaptations",
    ),
    "radiations": verify_adap_and_rad_settings(
        supported_settings,
        radiation_settings.with_and_without_radiation,
        "radiations",
    ),
    "overwrite": True,
    "simulators": ["nx"],
}
