"""Contains the output of an experiment run at 4 different stages.

Input: Experiment configuration.
    SubInput: Run configuration within an experiment.
        Stage 1: The networkx graphs that will be propagated.
        Stage 2: The propagated networkx graphs (at least one per timestep).
        Stage 3: Visaualisation of the networkx graphs over time.
        Stage 4: Post-processed performance data of algorithm and adaptation
        mechanism.
"""
# pylint: disable=W0613 # work in progress.
from src.export_results.helper import run_config_to_filename
from src.export_results.verify_stage_1_graphs import verify_stage_1_graphs
from src.export_results.verify_stage_2_graphs import verify_stage_2_graphs
from src.export_results.verify_stage_3_graphs import verify_stage_3_graphs
from src.export_results.verify_stage_4_graphs import verify_stage_4_graphs

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
    "overwrite_sim_results": True,
    "overwrite_visualisation": True,
    "radiation": {
        "delta_synaptic_w": (0.05, 0.4),
    },
    "seed": 5,
    "simulator": "lava",
}


def output_files_stage_1(experiment_config, run_config, graphs_stage_1):
    """Merges the experiment configuration dict, run configuration dict and
    graphs into a single dict.

    The graphs have not yet been simulated, hence they should be
    convertible into dicts. This merged dict is then written to file.
    The unique_id of the experiment is added to the file as a filetag,
    as well as the unique_id of the run. Furthermore, all run parameter
    values are added as file tags, to make it easier to filter certain
    runs to manually inspect the results.
    """
    run_config_to_filename(run_config)
    # TODO: Ensure output file exists.
    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write experiment_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.


def output_files_stage_2(experiment_config, run_config, graphs_stage_2):
    """Merges the experiment configuration dict, run configuration dict into a
    single dict.

    If the networkx (nx) simulator is used, the graphs should be
    convertible into dicts. This merged dict is then written to file. If
    the lava simulator is used, the graphs cannot (easily) be converted
    into dicts, hence in that case, only the experiment and run settings
    are merged into a single dict that is exported. The lava graphs will
    be terminated and exported as pickle. If the graphs are not
    exported, pickle throws an error because some
    pipe/connection/something is still open. The unique_id of the
    experiment is added to the file as a filetag, as well as the
    unique_id of the run. Furthermore, all run parameter values are
    added as file tags, to make it easier to filter certain runs to
    manually inspect the results.
    """
    run_config_to_filename(run_config)
    # TODO: Ensure output file exists.

    # TODO: merge experiment config, run_config into single dict.
    if run_config["simulator"] == "nx":
        # TODO: append graphs to dict.

        pass
    elif run_config["simulator"] == "lava":
        # TODO: terminate simulation.
        # TODO: write simulated lava graphs to pickle.
        pass
    else:
        raise Exception("Simulator not supported.")
    # TODO: write merged dict to file.

    # TODO: append tags to output file(s).


def output_files_stage_3(experiment_config, run_config, graphs_stage_3):
    """This only outputs the visualisation of the desired graphs.

    If the graphs are simulated for 50 timesteps, 50 pictures per graph
    will be outputted. For naming scheme and taging, see documentation
    of function output_files_stage_1 or output_files_stage_2.
    """
    run_config_to_filename(run_config)
    # TODO: Optional: ensure output files exists.

    # TODO: merge experiment config, run_config into single dict.
    if run_config["simulator"] == "nx":
        # TODO: append graphs to dict.

        pass
    elif run_config["simulator"] == "lava":
        # TODO: terminate simulation.
        # TODO: write simulated lava graphs to pickle.
        pass
    else:
        raise Exception("Simulator not supported.")
    # TODO: write merged dict to file.

    # TODO: append tags to output file(s).


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
