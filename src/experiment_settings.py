"""Contains experiment settings."""
# pylint: disable=R0801
from src.Supported_settings import Supported_settings


class Adaptation_settings:
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


adaptation_settings = Adaptation_settings()
radiation_settings = Radiation_settings()
supported_settings = Supported_settings()

with_and_without_adaptation_and_radiation = {
    "m": list(range(0, 1, 1)),
    "iterations": list(range(0, 3, 1)),
    "size,max_graphs": [(3, 15), (4, 15)],
    "adaptation": supported_settings.verify_adap_and_rad_settings(
        adaptation_settings.with_and_without_adaptation, "adaptation"
    ),
    "radiation": supported_settings.verify_adap_and_rad_settings(
        radiation_settings.with_and_without_radiation, "radiation"
    ),
    "overwrite": True,
    "simulators": ["nx"],
}
