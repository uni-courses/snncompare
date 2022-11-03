"""Contains the output of an experiment run at 4 different stages.

Input: Experiment configuration.
SubInput: Run configuration within an experiment.
    Stage 1: The networkx graphs that will be propagated.
    Stage 2: The propagated networkx graphs (at least one per timestep).
    Stage 3: Visaualisation of the networkx graphs over time.
    Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism.
"""
import pathlib
from typing import List, Optional

import jsons

from src.snncompare.export_results.export_json_results import (
    write_dict_to_json,
)
from src.snncompare.export_results.export_nx_graph_to_json import (
    convert_digraphs_to_json,
)
from src.snncompare.export_results.helper import run_config_to_filename
from src.snncompare.export_results.load_pickles_get_results import (
    get_desired_properties_for_graph_printing,
)
from src.snncompare.export_results.verify_nx_graphs import (
    verify_results_nx_graphs,
    verify_results_nx_graphs_contain_expected_stages,
)
from src.snncompare.export_results.verify_stage_1_graphs import (
    verify_stage_1_graphs,
)
from src.snncompare.export_results.verify_stage_2_graphs import (
    verify_stage_2_graphs,
)
from src.snncompare.export_results.verify_stage_3_graphs import (
    verify_stage_3_graphs,
)
from src.snncompare.export_results.verify_stage_4_graphs import (
    verify_stage_4_graphs,
)
from src.snncompare.graph_generation.helper_network_structure import (
    plot_coordinated_graph,
)
from src.snncompare.helper import get_sim_duration

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


# def output_files_stage_3(experiment_config, run_config, graphs_stage_3):
# This only outputs the visualisation of the desired graphs.

# If the graphs are simulated for 50 timesteps, 50 pictures per graph
# will be outputted. For naming scheme and taging, see documentation
# of function output_files_stage_1_and_2 or output_files_stage_2.

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
) -> None:
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


def output_stage_json(
    results_nx_graphs: dict, filename: str, stage_index: int, to_run: dict
) -> None:
    """Exports results dict to a json file."""

    if results_nx_graphs["graphs_dict"] == {}:
        raise Exception(
            "Error, the graphs_of_stage of stage_index="
            + f"{stage_index} was an empty dict."
        )
    verify_results_nx_graphs(
        results_nx_graphs, results_nx_graphs["run_config"]
    )
    verify_results_nx_graphs_contain_expected_stages(
        results_nx_graphs, stage_index, to_run
    )

    results_json_graphs = convert_digraphs_to_json(
        results_nx_graphs, stage_index
    )

    # TODO: Optional: ensure output files exists.
    output_filepath = f"results/{filename}.json"
    write_dict_to_json(output_filepath, jsons.dump(results_json_graphs))

    # Ensure output file exists.
    if not pathlib.Path(pathlib.Path(output_filepath)).resolve().is_file():
        raise Exception(f"Error:{output_filepath} does not exist.")
    # TODO: Verify the correct graphs is passed by checking the graph tag.
    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write experiment_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.


def plot_graph_behaviours(
    filepath: str, stage_2_graphs: dict, run_config: dict
) -> None:
    """Exports the plots of the graphs per time step of the run
    configuration."""

    # TODO: get this from the experiment settings/run configuration.
    desired_props = get_desired_properties_for_graph_printing()

    # Loop over the graph types
    for graph_name, graph in stage_2_graphs.items():
        sim_duration = get_sim_duration(
            stage_2_graphs["input_graph"],
            run_config,
        )
        for t in range(
            0,
            sim_duration,
        ):
            print(f"Plotting graph:{graph_name}, t={t}")
            # TODO: include circular input graph output.
            if graph_name != "input_graph":

                # TODO: Include verify that graph len remains unmodified.
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


# pylint: disable=R0912
# pylint: disable=R0915
def create_custom_plot_titles(
    graph_name: str, t: int, seed: int
) -> Optional[str]:
    """Creates custom titles for the SNN graphs for seed = 42."""
    # TODO: update to specify specific run_config instead of seed, to ensure
    #  the description is accurate/consistent with the SNN propagation.
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
