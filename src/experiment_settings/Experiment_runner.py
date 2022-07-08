"""Contains a single setting of the experiment configuration settings.

(The values of the settings may vary, yet the types should be the same.)
"""


from src.experiment_settings.Adaptation_Rad_settings import (
    Adaptations_settings,
    Radiation_settings,
)
from src.experiment_settings.Supported_experiment_settings import (
    Supported_experiment_settings,
)
from src.experiment_settings.verify_experiment_settings import (
    verify_adap_and_rad_settings,
    verify_experiment_config,
    verify_has_unique_id,
)


class Experiment_runner:
    """Stores the configuration of a single run."""

    # pylint: disable=R0903

    def __init__(self, experi_config: dict, export: bool, show: bool) -> None:

        # Store the experiment configuration settings.
        self.experi_config = experi_config

        # Load the ranges of supported settings.
        self.supp_experi_setts = Supported_experiment_settings()

        # Verify the experiment experi_config are complete and valid.
        verify_experiment_config(
            self.supp_experi_setts, experi_config, has_unique_id=False
        )

        # If the experiment experi_config does not contain a hash-code,
        # create the unique hash code for this configuration.
        if not self.supp_experi_setts.has_unique_config_id(self.experi_config):
            self.supp_experi_setts.append_unique_config_id(
                self, self.experi_config
            )

        # Verify the unique hash code for this configuration is valid.
        verify_has_unique_id(self.experi_config)

        # Append the export and show arguments.
        self.experi_config["export"] = export
        self.experi_config["show"] = show

        # determine_what_to_run

        # TODO: Perform run accordingly.
        # __perform_run

    def determine_what_to_run(self):
        """Scans for existing output and then combines the run configuration
        settings to determine what still should be computed."""
        # Determine which of the 4 stages have been performed.

        # Check if the run is already performed without exporting.

        # Check if the run is already performed with exporting.

    # pylint: disable=W0238
    def __perform_run(self):
        """Private method that runs the experiment.

        The 2 underscores indicate it is private. This method executes
        the run in the way the processed configuration settings specify.
        """


def example_experi_config():
    """Creates example experiment configuration settings."""
    # Create prerequisites
    supp_experi_setts = Supported_experiment_settings()
    adap_sets = Adaptations_settings()
    rad_sets = Radiation_settings()

    # Create the experiment configuration settings for a run with adaptation
    # and with radiation.
    with_adaptation_with_radiation = {
        "algorithms": {
            "MDSA": {
                "m_vals": list(range(0, 1, 1)),
            }
        },
        "adaptations": verify_adap_and_rad_settings(
            supp_experi_setts, adap_sets.with_adaptation, "adaptations"
        ),
        "iterations": list(range(0, 3, 1)),
        "min_max_graphs": 1,
        "max_max_graphs": 15,
        "min_graph_size": 3,
        "max_graph_size": 20,
        "overwrite_sim_results": True,
        "overwrite_visualisation": True,
        "radiations": verify_adap_and_rad_settings(
            supp_experi_setts, rad_sets.with_radiation, "radiations"
        ),
        "size_and_max_graphs": [(3, 15), (4, 15)],
        "simulators": ["nx"],
    }
    return with_adaptation_with_radiation
