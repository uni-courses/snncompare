"""Contains the supported experiment settings.

(The values of the settings may vary, yet the values of an experiment
setting should be within the ranges specified in this file, and the
setting types should be identical.)
"""


# pylint: disable=R0902
# The settings object contains all the settings as a dictionary, hence no
# hierarchy is used,leading to 10/7 instance attributes.
# pylint: disable=R0801
# pylint: disable=R0903


class Supported_run_settings:
    """Stores the supported experiment setting parameter ranges.

    An experiment can consist of multiple runs. A run is a particular
    combination of experiment setting parameters.
    """

    def __init__(
        self,
    ) -> None:
        # Config_settings dictionary keys:
        self.config_setting_parameters = [
            "adaptation",
            "algorithms",
            "iterations",
            "max_graph_size",
            "max_max_graphs",
            "min_graph_size",
            "min_max_graphs",
            "radiation",
            "overwrite_sim_results",
            "overwrite_visualisation",
            "seed",
            "simulators",
            "size_and_max_graphs",
        ]
