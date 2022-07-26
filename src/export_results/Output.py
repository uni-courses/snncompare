"""Contains the output of an experiment run at 4 different stages.

Input: Experiment configuration.
SubInput: Run configuration within an experiment.
    Stage 1: The networkx graphs that will be propagated.
    Stage 2: The propagated networkx graphs (at least one per timestep).
    Stage 3: Visaualisation of the networkx graphs over time.
    Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism.
"""
import json
from pathlib import Path
from pprint import pprint
from typing import List

from src.export_results.export_json_results import (
    digraph_to_json,
    write_dict_to_json,
)
from src.export_results.helper import run_config_to_filename
from src.export_results.plot_graphs import (
    create_root_dir_if_not_exists,
    create_target_dir_if_not_exists,
)
from src.export_results.verify_stage_1_graphs import verify_stage_1_graphs
from src.export_results.verify_stage_2_graphs import verify_stage_2_graphs
from src.export_results.verify_stage_3_graphs import verify_stage_3_graphs
from src.export_results.verify_stage_4_graphs import verify_stage_4_graphs

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
    "overwrite_sim_results": True,
    "overwrite_visualisation": True,
    "radiation": {
        "delta_synaptic_w": (0.05, 0.4),
    },
    "seed": 5,
    "simulator": "lava",
}


def create_results_directories():
    """Results directory structure is: <repository root_dir>/results/stage_1.

    <repository root_dir>/results/stage_2 <repository
    root_dir>/results/stage_3 <repository root_dir>/results/stage_4
    """
    create_root_dir_if_not_exists("results")
    for stage_index in range(1, 5):  # Indices 1 to 4
        create_target_dir_if_not_exists("results/", f"stage_{stage_index}")

    # TODO: assert directory: <repo root dir>/results/stage_1" exists


def output_files_stage_1(
    experiment_config: dict, run_config: dict, graphs_stage_1: dict
):
    """Merges the experiment configuration dict, run configuration dict and
    graphs into a single dict. This method assumes only the graphs that are to
    be exported are passed into this method.

    The graphs have not yet been simulated, hence they should be
    convertible into dicts. This merged dict is then written to file.
    The unique_id of the experiment is added to the file as a filetag,
    as well as the unique_id of the run. Furthermore, all run parameter
    values are added as file tags, to make it easier to filter certain
    runs to manually inspect the results.

    :param experiment_config: param run_config:
    :param graphs_stage_1:
    :param run_config:
    """
    filename = run_config_to_filename(run_config)
    print(f"filename={filename}")
    pprint(run_config)

    # TODO: include stage index
    merge_run_config_and_graphs(run_config, graphs_stage_1)
    # run_config["stage_1_graphs"] = graphs_stage_1

    # TODO: Optional: ensure output files exists.
    output_filepath = f"results/stage_1/{filename}.json"
    print(f"output_filepath={output_filepath}")
    write_dict_to_json(output_filepath, run_config)

    # TODO: Ensure output file exists.
    # TODO: Verify the correct graphs is passed by checking the graph tag.
    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write experiment_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.


def merge_run_config_and_graphs(run_config: dict, graphs: dict) -> None:
    """Adds the networkx graphs of the graphs dictionary into the run config
    dictionary."""
    for graph_name, graph in graphs.items():
        run_config[graph_name] = digraph_to_json(graph)


def output_files_stage_2(experiment_config, run_config, graphs_stage_2):
    """Merges the experiment configuration dict, run configuration dict into a
    single dict. This method assumes only the graphs that are to be exported
    are passed into this method.

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

    :param experiment_config: param run_config:
    :param graphs_stage_2:
    :param run_config:
    """
    run_config_to_filename(run_config)
    # TODO: Ensure output file exists.
    # TODO: Verify the correct graphs is passed by checking the graph tag.

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

    :param experiment_config: param run_config:
    :param graphs_stage_3:
    :param run_config:
    """
    run_config_to_filename(run_config)
    # TODO: Optional: ensure output files exists.

    # TODO: loop through graphs and create visualisation.
    # TODO: ensure the run parameters are in a legend
    # TODO: loop over the graphs (t), and output them.
    # TODO: append tags to output file(s).


def output_files_stage_4(
    correct: bool,
    experiment_config: dict,
    nodes_alipour: List[int],
    nodes_snn: List[int],
    run_config: dict,
):
    """This only outputs the algorithm performance and adaptation performance
    on the specific graph.

    It does so by outputting the nodes selected by Alipour, the nodes
    selected by the SNN graph, and a boolean indicating whether they are
    the same or not.

    :param correct: bool:
    :param experiment_config: dict:
    :param nodes_alipour: List[int]:
    :param nodes_snn: List[int]:
    :param run_config: dict:
    :param correct: bool:
    :param experiment_config: dict:
    :param nodes_alipour: List[int]:
    :param nodes_snn: List[int]:
    :param run_config: dict:
    """

    run_config_to_filename(run_config)
    # TODO: ensure the run parameters are in a legend
    # TODO: loop over the graphs (t), and output them.
    # TODO: append tags to output file(s).


# pylint: disable=R0903
class Stage_1_graphs:
    """Stage 1: The networkx graphs that will be propagated."""

    def __init__(
        self, experiment_config: dict, graphs_stage_1: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs_stage_1 = graphs_stage_1
        verify_stage_1_graphs(
            experiment_config, run_config, self.graphs_stage_1
        )
        # G_original
        # G_SNN_input
        # G_SNN_adapted
        # G_SNN_rad_damage
        # G_SNN_adapted_rad_damage


# pylint: disable=R0903
class Stage_2_graphs:
    """Stage 2: The propagated networkx graphs (at least one per timestep)."""

    def __init__(
        self, experiment_config: dict, graphs_stage_2: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs_stage_2 = graphs_stage_2
        verify_stage_2_graphs(
            experiment_config, run_config, self.graphs_stage_2
        )


# pylint: disable=R0903
class Stage_3_graphs:
    """Stage 3: Visaualisation of the networkx graphs over time."""

    def __init__(
        self, experiment_config: dict, graphs_stage_3: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs_stage_3 = graphs_stage_3
        verify_stage_3_graphs(
            experiment_config, run_config, self.graphs_stage_3
        )


# pylint: disable=R0903
class Stage_4_graphs:
    """Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism."""

    def __init__(
        self, experiment_config: dict, graphs_stage_4: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.graphs_stage_4 = graphs_stage_4
        verify_stage_4_graphs(
            experiment_config, run_config, self.graphs_stage_4
        )


def get_extensions_dict(run_config, stage_index) -> dict:
    """Returns the file extensions of the output types.
    TODO: support .json as well as .txt for the dictionaries.

    :param run_config: param stage_index:
    :param stage_index:

    """
    if stage_index == 1:
        return {"config_and_graphs": ".txt"}
    if stage_index == 2:
        if run_config["simulator"] == "lava":
            return {"config": ".txt", "graphs": ".pkl"}

        return {"config_and_graphs": ".txt"}
    if stage_index == 3:
        # TODO: support .eps and/or .pdf.
        return {"graphs": ".png"}
    if stage_index == 4:
        return {"config_and_results": ".txt"}
    raise Exception("Unsupported experiment stage.")


def get_extensions_list(run_config, stage_index) -> list:
    """

    :param run_config: param stage_index:
    :param stage_index:

    extensions = list(get_extensions_dict(run_config, stage_index).values())
    """
    return list(get_extensions_dict(run_config, stage_index).values())


def performed_stage(run_config, stage_index: int) -> bool:
    """Verifies the required output files exist for a given simulation.

    :param run_config: param stage_index:
    :param stage_index:
    """
    print("")
    expected_filenames = []

    filename = run_config_to_filename(run_config)
    relative_output_dir = f"results/stage_{stage_index}/"
    extensions = get_extensions_list(run_config, stage_index)
    for extension in extensions:
        if stage_index in [1, 2, 4]:

            expected_filenames.append(
                relative_output_dir + filename + extension
            )
            # TODO: append expected_filepath to run_config per stage.

        if stage_index == 3:

            # Check if output file(s) of stage 2 exist, otherwise return False.
            if not Path(relative_output_dir + filename + extension).is_file():
                return False

            # If the expected output files containing the adapted graphs exist,
            # get the number of simulation steps.
            nr_of_simulation_steps = get_nr_of_simulation_steps(
                relative_output_dir, filename
            )
            for t in range(0, nr_of_simulation_steps):
                # Generate graph filenames
                expected_filenames.append(
                    relative_output_dir + filename + f"t_{t}" + extension
                )

    # Check if the expected output files already exist.
    for filename in expected_filenames:
        if not Path(relative_output_dir + filename).is_file():
            return False
    return True


def load_stage_2_output_dict(relative_output_dir, filename) -> dict:
    """Loads the stage_2 output dictionary from a file.

    # TODO: Determine why the file does not yet exist at this positinoc.
    # TODO: Output dict to json format.

    :param relative_output_dir: param filename:
    :param filename:
    """
    stage_2_output_dict_filepath = relative_output_dir + filename
    print(f"stage_2_output_dict_filepath={stage_2_output_dict_filepath}")
    with open(stage_2_output_dict_filepath, encoding="utf-8") as json_file:
        stage_2_output_dict = json.load(json_file)
    return stage_2_output_dict


def get_nr_of_simulation_steps(relative_output_dir, filename) -> int:
    """Reads the amount of simulation steps from the stage2 run configuration.

    :param relative_output_dir: param filename:
    :param filename:
    """
    stage_2_output_dict = load_stage_2_output_dict(
        relative_output_dir, filename
    )
    run_config = stage_2_output_dict["run_config"]
    return run_config["duration"]
