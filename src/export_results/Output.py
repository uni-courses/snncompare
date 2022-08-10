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
from typing import List

import jsons
import networkx as nx

from src.export_results.export_json_results import (
    digraph_to_json,
    write_dict_to_json,
)
from src.export_results.helper import run_config_to_filename
from src.export_results.load_pickles_get_results import (
    get_desired_properties_for_graph_printing,
)
from src.export_results.verify_stage_1_graphs import verify_stage_1_graphs
from src.export_results.verify_stage_2_graphs import verify_stage_2_graphs
from src.export_results.verify_stage_3_graphs import verify_stage_3_graphs
from src.export_results.verify_stage_4_graphs import verify_stage_4_graphs
from src.graph_generation.helper_network_structure import (
    plot_coordinated_graph,
)

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
    if run_config["export_snns"]:
        filename = run_config_to_filename(run_config)
        output_stage_json(
            experiment_config,
            graphs_stage_1,
            filename,
            run_config,
            1,
        )


def output_stage_files(
    experiment_config: dict,
    run_config: dict,
    graphs_stage_2: dict,
    stage_index: int,
):
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

        if run_config["export_snns"]:
            # Output the json dictionary of the files.
            filename = run_config_to_filename(run_config)

            output_stage_json(
                experiment_config,
                graphs_stage_2,
                filename,
                run_config,
                stage_index,
            )

        # TODO: Check if plots are already generated and if they must be
        # overwritten.
        if run_config["show_snns"]:
            # Output graph behaviour for stage stage_index.
            plot_stage_2_graph_behaviours(filename, graphs_stage_2, run_config)

    elif run_config["simulator"] == "lava":
        # TODO: terminate simulation.
        # TODO: write simulated lava graphs to pickle.
        raise Exception("Error, lava export method not yet implemented.")
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
        self.graphs_stage_1: dict = graphs_stage_1
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
        return {"config_and_graphs": ".json"}
    if stage_index == 2:
        if run_config["simulator"] == "lava":
            return {"config": ".json", "graphs": ".png"}

        return {"config_and_graphs": ".txt"}
    if stage_index == 3:
        # TODO: support .eps and/or .pdf.
        return {"graphs": ".png"}
    if stage_index == 4:
        return {"config_and_results": ".json"}
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
    expected_filepaths = []

    filename = run_config_to_filename(run_config)
    relative_output_dir = "results/"
    extensions = get_extensions_list(run_config, stage_index)
    for extension in extensions:
        if stage_index in [1, 2, 4]:

            expected_filepaths.append(
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
                expected_filepaths.append(
                    relative_output_dir + filename + f"t_{t}" + extension
                )

    # Check if the expected output files already exist.
    for filepath in expected_filepaths:
        if not Path(filepath).is_file():
            print(f"filepath={filepath} not found for stage:{stage_index}")
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


def merge_experiment_and_run_config_with_graphs(
    experiment_config: dict, run_config: dict, graphs: dict, stage_index: int
) -> dict:
    """Adds the networkx graphs of the graphs dictionary into the run config
    dictionary."""
    # Convert incoming graphs to dictionary.
    graphs_dict = {}
    for graph_name, graph_container in graphs.items():
        if stage_index == 1:
            graphs_dict[graph_name] = digraph_to_json(graph_container)
        elif stage_index == 2:
            graphs_per_type = []
            if isinstance(graph_container, (nx.DiGraph, nx.Graph)):
                graphs_per_type.append(digraph_to_json(graph_container))
            elif isinstance(graph_container, List):
                for graph in graph_container:
                    graphs_per_type.append(digraph_to_json(graph))
            else:
                raise Exception(
                    f"Error, unsupported type:{type(graph_container)}"
                )
            graphs_dict[graph_name] = graphs_per_type

    output_dict = {
        "experiment_config": experiment_config,
        "run_config": run_config,
        "graphs_dict": graphs_dict,
    }
    return output_dict


def output_stage_json(
    experiment_config: dict,
    graphs_of_stage: dict,
    filename: str,
    run_config: dict,
    stage_index: int,
) -> None:
    """Merges the experiment config, run config and graphs of stage 1 into a
    single dict and exports that dict to a json file."""
    # TODO: include stage index
    output_dict = merge_experiment_and_run_config_with_graphs(
        experiment_config, run_config, graphs_of_stage, stage_index
    )

    print(f"filename={filename}")
    # TODO: Optional: ensure output files exists.
    output_filepath = f"results/{filename}.json"
    write_dict_to_json(output_filepath, jsons.dump(output_dict))

    # TODO: Ensure output file exists.
    # TODO: Verify the correct graphs is passed by checking the graph tag.
    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write experiment_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.


def plot_stage_2_graph_behaviours(
    filepath: str, graphs: dict, run_config: dict
):
    """Exports the plots of the graphs per time step of the run
    configuration."""

    desired_props = get_desired_properties_for_graph_printing()

    # Loop over the graph types

    for graph_name, graph_list in graphs.items():
        for i, graph in enumerate(graph_list):
            # if graph_name == "rad_snn_algo_graph":
            # TODO: include check for only rad dead things.

            # TODO plot a single graph.

            # pylint: disable=R0913
            # TODO: reduce the amount of arguments from 6/5 to at most 5/5.
            plot_coordinated_graph(
                graph,
                desired_props,
                False,
                f"{graph_name}_{filepath}_{i}",
                title=create_custom_plot_titles(
                    graph_name, i, run_config["seed"]
                ),
            )


def print_dead_neuron_names(some_graph: nx.DiGraph):
    """Prints the dead neuron names."""
    for nodename in some_graph:
        if "rad_death" in some_graph.nodes[nodename].keys():
            # if nodename in dead_neuron_names:
            if some_graph.nodes[nodename]["rad_death"]:
                print(nodename)


# pylint: disable=R0912
# pylint: disable=R0915
def create_custom_plot_titles(graph_name, t: int, seed: int):
    """Creates custom titles for the SNN graphs for seed = 42."""
    if seed == 42:
        title = None
        if graph_name == "snn_algo_graph":
            if t == 0:
                title = (
                    "Initialisation:\n spike_once and random neurons spike, "
                    + "selector starts."
                )
            if t == 1:
                title = (
                    "Selector neurons continue exciting\n degree_receivers "
                    + "(WTA-circuits)."
                )
            if t == 21:
                title = "Selector about to create degree_receiver winner."
            if t == 22:
                title = (
                    "Degree_receiver neurons spike and inhibit selector,\n"
                    + " excite counter neuron."
                )
            if t == 23:
                title = "WTA circuits completed, score stored in counter."
            if t == 24:
                title = "Remaining WTA circuit being completed."
            if t == 25:
                title = (
                    "Algorithm completed, counter neuron amperage read out."
                )
            if t == 26:
                title = "."

        if graph_name == "adapted_snn_graph":
            if t == 0:
                title = (
                    "Initialisation Adapted SNN:\n All neurons have a "
                    + "redundant neuron."
                )
            if t == 1:
                title = (
                    "Selector neurons have inhibited redundant selector"
                    + " neurons."
                )
            if t == 21:
                title = "Selector about to create degree_receiver winner."
            if t == 22:
                title = (
                    "Degree_receiver neurons spike and inhibit selector,\n"
                    + "excite counter neuron."
                )
            if t == 23:
                title = "WTA circuits completed, score stored in counter."
            if t == 24:
                title = "Remaining WTA circuit being completed."
            if t == 25:
                title = (
                    "Algorithm completed, counter neuron amperage read out."
                )
            if t == 26:
                title = "."

        if graph_name == "rad_adapted_snn_graph":
            if t == 0:
                title = (
                    "Simulated Radiation Damage:\n Red neurons died, "
                    + "(don't spike)."
                )
            if t == 1:
                title = (
                    "Redundant spike_once and selector neurons take over.\n "
                    + "Working neurons inhibited redundant neurons (having "
                    + "delay=1)."
                )
            if t == 20:
                title = "Selector_2_0 about to create degree_receiver winner."
            if t == 21:
                title = (
                    "Degree_receiver_2_1_0 neuron spike and inhibits selector"
                    + ",\n excites counter neuron."
                )
            if t == 22:
                title = (
                    "First WTA circuits completed, score stored in counter. "
                    + "2nd WTA\n circuit is has winner, inhibits selectors, "
                    + "stores result in counter."
                )
            if t == 23:
                title = "Third WTA circuit has winner."
            if t == 24:
                title = (
                    "Algorithm completed, counter neuron amperage read out.\n"
                    + " Redundancy saved the day."
                )
            if t == 25:
                title = "."
    return title
