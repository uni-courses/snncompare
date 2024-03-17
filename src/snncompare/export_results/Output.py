"""Contains the output of an experiment run at 4 different stages.

Input: Experiment configuration.
SubInput: Run configuration within an experiment.
    Stage 1: The networkx graphs that will be propagated.
    Stage 2: The propagated networkx graphs (at least one per timestep).
    Stage 3: Visaualisation of the networkx graphs over time.
    Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism.
"""
from typing import Dict

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.run_config.Run_config import Run_config

from .verify_stage_1_graphs import verify_stage_1_graphs
from .verify_stage_2_graphs import verify_stage_2_graphs
from .verify_stage_3_graphs import verify_stage_3_graphs
from .verify_stage_4_graphs import verify_stage_4_graphs

# pylint: disable=W0613 # work in progress.


with_adaptation_with_radiation = {
    "adaptation": {"redundancy": 1.0},
    "algorithm": {
        "MDSA": {
            "m_val": 2,
        }
    },
    "graph_size": 4,
    "graph_nr": 5,
    "iteration": 4,
    "recreate_s4": True,
    "recreate_s3": True,
    "radiation": {
        "delta_synaptic_w": (0.05, 0.4),
    },
    "simulator": "lava",
}


# def output_files_stage_3(exp_config, run_config, graphs_stage_3):
# This only outputs the visualisation of the desired graphs.

# If the graphs are simulated for 50 timesteps, 50 pictures per graph
# will be outputted. For naming scheme and taging, see documentation
# of function output_files_stage_1_and_2 or output_files_stage_2.

# :param exp_config: param run_config:
# :param graphs_stage_3:
# :param run_config:
# """

# TODO: Optional: ensure output files exists.

# TODO: loop through graphs and create visualisation.
# TODO: ensure the run parameters are in a legend
# TODO: loop over the graphs (t), and output them.
# TODO: append tags to output file(s).


# pylint: disable=R0903
class Stage_1_graphs:
    """Stage 1: The networkx graphs that will be propagated."""

    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        stage_1_graphs: Dict,
        run_config: Run_config,
    ) -> None:
        """Initializes an object for performing stage 2 computations.

        Args:
        :exp_config: (Exp_config), The experiment configuration.
        :stage_1_graphs: (Dict), A dictionary containing the graphs
        computed in stage 1.
        :run_config: (Run_config), The run configuration.
        """

        self.exp_config = exp_config
        self.run_config = run_config
        self.stage_1_graphs: Dict = stage_1_graphs
        verify_stage_1_graphs(
            exp_config=exp_config,
            run_config=run_config,
            graphs=self.stage_1_graphs,
        )
        # G_original
        # G_SNN_input
        # G_SNN_adapted
        # G_SNN_rad_damage
        # G_SNN_adapted_rad_damage


# pylint: disable=R0903
class Stage_2_graphs:
    """Stage 2: The propagated networkx graphs (at least one per timestep)."""

    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        graphs_stage_2: Dict,
        run_config: Run_config,
    ) -> None:
        """Initializes a `ModelTrainer` object.

        Args:
        :exp_config: (`Exp_config`), The experiment configuration.
        :graphs_stage_2: (`Dict`), The graphs for the second stage of
        training.
        :run_config: (`Run_config`), The run configuration.
        """

        self.exp_config = exp_config
        self.run_config = run_config
        self.graphs_stage_2 = graphs_stage_2
        verify_stage_2_graphs(
            exp_config=exp_config,
            run_config=run_config,
            graphs=self.graphs_stage_2,
        )


# pylint: disable=R0903
class Stage_3_graphs:
    """Stage 3: Visaualisation of the networkx graphs over time."""

    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        graphs_stage_3: Dict,
        run_config: Run_config,
    ) -> None:
        """Initializes the graphs for stage 3.

        Args:
        :exp_config: (Exp_config), The experiment configuration.
        :graphs_stage_3: (Dict), The graphs for stage 3.
        :run_config: (Run_config), The run configuration.
        Returns:
        None.
        """

        self.exp_config = exp_config
        self.run_config = run_config
        self.graphs_stage_3 = graphs_stage_3
        verify_stage_3_graphs(
            exp_config=exp_config,
            run_config=run_config,
            graphs_stage_3=self.graphs_stage_3,
        )


# pylint: disable=R0903
class Stage_4_graphs:
    """Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism."""

    @typechecked
    def __init__(
        self,
        exp_config: Exp_config,
        graphs_stage_4: Dict,
        run_config: Run_config,
    ) -> None:
        """This class performs the 5th stage of the experiment (training the
        stage 4 graphs) in the graph based network pipeline.

        Args:
        :exp_config: (Exp_config), The experiment configurations.
        :run_config: (Run_config), The run configurations.
        :graphs_stage_4: (Dict), The graphs of stage 4 of the pipeline.
        """

        self.exp_config = exp_config
        self.run_config = run_config
        self.graphs_stage_4 = graphs_stage_4
        verify_stage_4_graphs(
            exp_config=exp_config,
            run_config=run_config,
            graphs_stage_4=self.graphs_stage_4,
        )
