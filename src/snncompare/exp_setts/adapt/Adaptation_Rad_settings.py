"""Contains experiment settings."""
from typeguard import typechecked

# pylint: disable=R0801
from ..Supported_experiment_settings import Supported_experiment_settings


class Adaptations_settings:
    """Stores all used adaptation settings."""

    # pylint: disable=R0903

    @typechecked
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

    @typechecked
    def __init__(
        self,
    ) -> None:
        self.without_radiation: dict = {
            "None": [],
        }
        self.with_radiation = {
            "neuron_death": [
                # 0.01,
                # 0.05,
                # 0.1,
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
