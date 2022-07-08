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
from src.export_results.verify_stage_2_graphs import verify_stage_2_graphs
from src.export_results.verify_stage_3_graphs import verify_stage_3_graphs
from src.export_results.verify_stage_4_graphs import verify_stage_4_graphs


# pylint: disable=R0903
class Stage_1_graphs:
    """Stage 1: The networkx graphs that will be propagated."""

    def __init__(
        self, experiment_config: dict, graphs_stage1: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs = graphs_stage1
        verify_stage_1_graphs()
        # G_original
        # G_SNN_input
        # G_SNN_adapted
        # G_SNN_rad_damage
        # G_SNN_adapted_rad_damage


# pylint: disable=R0903
class Stage_2_graphs:
    """Stage 2: The propagated networkx graphs (at least one per timestep)."""

    def __init__(
        self, experiment_config: dict, graphs_stage2: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs = graphs_stage2
        verify_stage_2_graphs()


# pylint: disable=R0903
class Stage_3_graphs:
    """Stage 3: Visaualisation of the networkx graphs over time."""

    def __init__(
        self, experiment_config: dict, graphs_stage3: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs = graphs_stage3
        verify_stage_3_graphs()


# pylint: disable=R0903
class Stage_4_graphs:
    """Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism."""

    def __init__(
        self, experiment_config: dict, graphs_stage4: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs = graphs_stage4
        verify_stage_4_graphs()
