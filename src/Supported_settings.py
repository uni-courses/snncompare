"""Contains the supported experiment settings.

(The values of the settings may vary, yet the types should be the same.)
"""


# pylint: disable=R0902
# The settings object contains all the settings as a dictionary, hence no
# hierarchy is used, leading to 10/7 instance attributes.
class Supported_settings:
    """Stores examples of the supported experiment settings, such as radiation
    and adaptation settings.

    Also verifies the settings that are created.
    """

    def __init__(
        self,
    ) -> None:
        # The number of iterations for which the Alipour approximation is ran.
        self.m = list(range(0, 1, 1))
        # The number of times the experiment is repeated.
        self.iterations = list(range(0, 3, 1))

        # Specify the maximum number of: (maximum number of graphs per run
        # size).
        self.max_max_graphs = 15
        # The size of the graph and the maximum number of used graphs of that
        # size.
        self.size_and_max_graphs = [
            (3, self.max_max_graphs),
            (4, self.max_max_graphs),
        ]
        # Overwrite the simulation results or not.
        self.overwrite_sim_results = True
        # Overwrite the visualisation of the SNN behaviour or not.
        self.overwrite_visualisation = True
        # The backend/type of simulator that is used.
        self.simulators = ["nx"]

        # Generate the supported adaptation settings.
        self.specify_supported_adaptation_settings()
        # Generate the supported radiation settings.
        self.specify_supported_radiation_settings()

    def specify_supported_adaptation_settings(self):
        """Specifies all the supported types of adaptation settings."""

        # Specify the (to be) supported adaptation types.
        self.adaptation = {
            "None": [],
            "redundancy": [
                1.0,
            ],  # Create 1 redundant neuron per neuron.
            "population": [
                10.0
            ],  # Create a population of 10 neurons to represent a
            # single neuron.
            "rate_coding": [
                5.0
            ],  # Multiply firing frequency with 5 to limit spike decay
            # impact.
        }

    def specify_supported_radiation_settings(self):
        """Specifies types of supported radiation settings. Some settings
        consist of a list of tuples, with the probability of a change
        occurring, followed by the average magnitude of the change.

        Others only contain a list of floats which represent the
        probability of radiation induced change occurring.
        """
        # List of tuples with x=probabiltity of change, y=average value change
        # in synaptic weights.
        self.delta_synaptic_w = [
            (0.01, 0.5),
            (0.05, 0.4),
            (0.1, 0.3),
            (0.2, 0.2),
            (0.25, 0.1),
        ]

        # List of tuples with x=probabiltity of change, y=average value change
        # in neuronal threshold.
        self.delta_vth = [
            (0.01, 0.5),
            (0.05, 0.4),
            (0.1, 0.3),
            (0.2, 0.2),
            (0.25, 0.1),
        ]

        # Create a supported radiation setting example.
        self.radiation = {
            # No radiation
            "None": [],
            # Radiation effects are transient, they last for 1 or 10 simulation
            # steps. If transient is 0., the changes are permanent.
            "transient": [0.0, 1.0, 10.0],
            # List of probabilities of a neuron dying due to radiation.
            "neuron_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            # List of probabilities of a synapse dying due to radiation.
            "synaptic_death": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.25,
            ],
            # List of: (probability of synaptic weight change, and the average
            # factor with which it changes due to radiation).
            "delta_synaptic_w": self.delta_synaptic_w,
            # List of: (probability of neuron threshold change, and the average
            # factor with which it changes due to radiation).
            "delta_vth": self.delta_vth,
        }
