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
import pathlib
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
from src.helper import get_sim_duration

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
    experiment_config: dict, run_config: dict, stage_1_graphs: dict
):
    """TODO: generalise to stage 1 and/or stage2.
    Merges the experiment configuration dict, run configuration dict and
    graphs into a single dict. This method assumes only the graphs that are to
    be exported are passed into this method.

    The graphs have not yet been simulated, hence they should be
    convertible into dicts. This merged dict is then written to file.
    The unique_id of the experiment is added to the file as a filetag,
    as well as the unique_id of the run. Furthermore, all run parameter
    values are added as file tags, to make it easier to filter certain
    runs to manually inspect the results.

    :param experiment_config: param run_config:
    :param stage_1_graphs:
    :param run_config:
    """
    # TODO: make it clear that export_snns means export the images of the
    # graphs. The graphs should always be exported if they do not yet exist.
    if run_config["export_snns"]:
        filename = run_config_to_filename(run_config)
        output_stage_json(
            experiment_config,
            stage_1_graphs,
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
    are merged into a single dict that is exported.

    % TODO: The lava graphs will be terminated and exported as pickle.
    If the graphs are not exported, pickle throws an error because some
    pipe/connection/something is still open.

    The unique_id of the experiment is added to the file as a filetag, as well
    as the unique_id of the run. Furthermore, all run parameter values are
    added as file tags, to make it easier to filter certain runs to
    manually inspect the results.

    Also exports the images of the graph behaviour.

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
            print(f"Exporting image:{filename}\n")

            output_stage_json(
                experiment_config,
                graphs_stage_2,
                filename,
                run_config,
                stage_index,
            )

        # TODO: Check if plots are already generated and if they must be
        # overwritten.
        # TODO: Distinguish between showing snns and outputting snns.
        if run_config["export_snns"] and stage_index == 3:
            # Output graph behaviour for stage stage_index.
            plot_graph_behaviours(filename, graphs_stage_2, run_config)

    elif run_config["simulator"] == "lava":
        # TODO: terminate simulation.
        # TODO: write simulated lava graphs to pickle.
        raise Exception("Error, lava export method not yet implemented.")
    else:
        raise Exception("Simulator not supported.")
    # TODO: write merged dict to file.

    # TODO: append tags to output file(s).


# def output_files_stage_3(experiment_config, run_config, graphs_stage_3):
# This only outputs the visualisation of the desired graphs.

# If the graphs are simulated for 50 timesteps, 50 pictures per graph
# will be outputted. For naming scheme and taging, see documentation
# of function output_files_stage_1 or output_files_stage_2.

# :param experiment_config: param run_config:
# :param graphs_stage_3:
# :param run_config:
# """
# run_config_to_filename(run_config)
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
        self, experiment_config: dict, stage_1_graphs: dict, run_config: dict
    ) -> None:
        self.experiment_config = experiment_config
        self.run_config = run_config
        self.stage_1_graphs: dict = stage_1_graphs
        verify_stage_1_graphs(
            experiment_config, run_config, self.stage_1_graphs
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


def merge_stage_1_graphs(graphs):
    """Puts all the graphs of stage 1 into a single graph."""
    graphs_dict_stage_1 = {}
    for graph_name, graph_container in graphs.items():
        print(f"graph_name={graph_name}")

        if not isinstance(graph_container, (nx.DiGraph, nx.Graph)):
            raise Exception(
                "stage_index=1, Error, for graph:"
                + f"{graph_name}, the graph is not a"
                + f"nx.Digraph(). Instead, it is:{type(graph_container)}"
            )
        graphs_dict_stage_1[graph_name] = digraph_to_json(graph_container)
    if not graphs_dict_stage_1:  # checks if dict not empty like: {}
        raise Exception(
            f"Error, len(graphs)={len(graphs)} stage=1, graphs_dict_stage_1"
            + " is empty."
        )
    return graphs_dict_stage_1


def merge_stage_2_graphs(graphs):
    """Puts all the graphs of stage 2 into a single graph."""
    graphs_dict_stage_2 = {}
    for graph_name, graph_container in graphs.items():
        print(f"graph_name={graph_name}")
        graphs_per_type = []
        if isinstance(graph_container, (nx.DiGraph, nx.Graph)):
            graphs_per_type.append(digraph_to_json(graph_container))
        elif isinstance(graph_container, List):
            for graph in graph_container:
                graphs_per_type.append(digraph_to_json(graph))
        else:
            raise Exception(f"Error, unsupported type:{type(graph_container)}")
        graphs_dict_stage_2[graph_name] = graphs_per_type
    if not graphs_dict_stage_2:  # checks if dict not empty like: {}
        raise Exception(
            f"Error, len(graphs)={len(graphs)} stage=2, graphs_dict_stage_2"
            + " is empty."
        )
    return graphs_dict_stage_2


def merge_experiment_and_run_config_with_graphs(
    experiment_config: dict, run_config: dict, graphs: dict, stage_index: int
) -> dict:
    """Adds the networkx graphs of the graphs dictionary into the run config
    dictionary."""

    # Load existing graph dict if it already exists, and if overwrite is off.
    graphs_dict = load_pre_existing_graph_dict(run_config, stage_index)
    # Convert incoming graphs to dictionary.

    if stage_index == 1:
        graphs_dict["stage_1"] = merge_stage_1_graphs(graphs)
    elif stage_index == 2:
        graphs_dict["stage_2"] = merge_stage_2_graphs(graphs)
    if stage_index == 3:
        pass
    # Convert into single output dict.
    output_dict = {
        "experiment_config": experiment_config,
        "run_config": run_config,
        "graphs_dict": graphs_dict,
    }
    return output_dict


def load_pre_existing_graph_dict(run_config, stage_index):
    """Returns the pre-existing graphs that were generated during earlier
    stages of the experiment.

    TODO: write tests to verify it returns the
    correct data.
    """
    # If stage index ==1  you should always return an empty dict.
    if stage_index == 2:
        if not run_config["overwrite_sim_results"]:
            # Load graphs stages 1, 2, 3, 4
            return load_graphs_from_json(run_config, [1, 2, 3, 4])
        return load_graphs_from_json(run_config, [1])
    if stage_index == 3:
        if not run_config["overwrite_visualisation"]:
            return load_graphs_from_json(run_config, [1, 2, 3, 4])
        return load_graphs_from_json(run_config, [1, 2])
    if stage_index == 4:
        return load_graphs_from_json(run_config, [1, 2, 3, 4])
    return {}


def load_graphs_from_json(run_config, stages) -> dict:
    """Loads the json dict and returns the graphs of the relevant stages."""
    restored_graphs_dict = {}

    filename: str = run_config_to_filename(run_config)
    json_filepath = f"results/{filename}.json"

    # Read output JSON file into dict.
    with open(json_filepath, encoding="utf-8") as json_file:
        output_dict = json.load(json_file)

    if "experiment_config" not in output_dict:
        raise Exception("Error, key: experiment_config not in output_dict.")
    if "run_config" not in output_dict:
        raise Exception("Error, key: run_config not in output_dict.")
    if "graphs_dict" not in output_dict:
        raise Exception("Error, key: graphs_dict not in output_dict.")

    for stage in stages:
        if f"stage_{stage}" in output_dict["graphs_dict"]:
            restored_graphs_dict[f"stage_{stage}"] = output_dict[
                "graphs_dict"
            ][f"stage_{stage}"]
    return restored_graphs_dict


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

    if graphs_of_stage == {}:
        raise Exception(
            "Error, the graphs_of_stage of stage_index="
            + f"{stage_index} was an empty dict."
        )

    output_dict = merge_experiment_and_run_config_with_graphs(
        experiment_config, run_config, graphs_of_stage, stage_index
    )

    # TODO: Optional: ensure output files exists.
    output_filepath = f"results/{filename}.json"
    write_dict_to_json(output_filepath, jsons.dump(output_dict))

    # Ensure output file exists.
    if not pathlib.Path(pathlib.Path(output_filepath)).resolve().is_file():
        raise Exception(f"Error:{output_filepath} does not exist.")
    # TODO: Verify the correct graphs is passed by checking the graph tag.

    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write experiment_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.


def plot_graph_behaviours(filepath: str, graphs: dict, run_config: dict):
    """Exports the plots of the graphs per time step of the run
    configuration."""

    # TODO: get this from the experiment settings/run configuration.
    desired_props = get_desired_properties_for_graph_printing()

    # Loop over the graph types
    for graph_name, graph in graphs.items():
        print(f"graph_name={graph_name}")
        # if not isinstance(graph, nx.DiGraph):
        #    raise Exception(
        #        "Error, expected single DiGraph, yet found:" f"{type(graph)}"
        #    )
        # TODO: change to loop over neurons per timestep, instead of
        # over graphs.

        sim_duration = get_sim_duration(
            graph,
            run_config,
        )
        print(f"sim_duration={sim_duration}")
        for t in range(
            0,
            sim_duration,
        ):
            # TODO: include circular input graph output.
            if graph_name != "input_graph":

                # pylint: disable=R0913
                # TODO: reduce the amount of arguments from 6/5 to at most 5/5.
                plot_coordinated_graph(
                    graph,
                    desired_props,
                    t,
                    False,
                    f"{graph_name}_{filepath}_{t}",
                    title=create_custom_plot_titles(
                        graph_name, t, run_config["seed"]
                    ),
                )


def print_dead_neuron_names(some_graph: nx.DiGraph):
    """Prints the dead neuron names."""
    print("Dead neuron names:")
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
