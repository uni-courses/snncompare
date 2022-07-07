"""Contains the output of an experiment run at 4 different stages.

Input: Experiment configuration.
    SubInput: Run configuration within an experiment.
        Stage 1: The networkx graphs that will be propagated.
        Stage 2: The propagated networkx graphs (at least one per timestep).
        Stage 3: Visaualisation of the networkx graphs over time.
        Stage 4: Post-processed performance data of algorithm and adaptation
        mechanism.
"""

from src.export_results.verify_stage_1_graphs import verify_stage_1_graphs


# pylint: disable=R0903
class Stage_1_graphs:
    """Stage 1: The networkx graphs that will be propagated."""

    def __init__(
        self, experiment_config: dict, graphs: dict, run_setts: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_setts = run_setts
        self.graphs = graphs
        verify_stage_1_graphs()
        # G_original
        # G_SNN_input
        # G_SNN_adapted
        # G_SNN_rad_damage
        # G_SNN_adapted_rad_damage


# pylint: disable=R0903
class Stage_2_graphs:
    """Stage 2: The propagated networkx graphs (at least one per timestep)."""

    def __init__(self) -> None:
        pass


# pylint: disable=R0903
class Stage_3_graphs:
    """Stage 3: Visaualisation of the networkx graphs over time."""

    def __init__(self) -> None:
        pass


# pylint: disable=R0903
class Stage_4_graphs:
    """Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism."""

    def __init__(self) -> None:
        pass
