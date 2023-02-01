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
from typing import Dict, List, Optional

import jsons
from snnbackends.verify_nx_graphs import (
    verify_results_nx_graphs,
    verify_results_nx_graphs_contain_expected_stages,
)
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.exp_config.run_config.Run_config import Run_config
from snncompare.export_plots.create_png_plot import plot_coordinated_graph
from snncompare.export_plots.plot_graphs import plot_uncoordinated_graph

from .export_json_results import write_dict_to_json
from .export_nx_graph_to_json import convert_digraphs_to_json
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
    "overwrite_images_only": True,
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
# run_config_to_filename(run_config)
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
        self.exp_config = exp_config
        self.run_config = run_config
        self.graphs_stage_4 = graphs_stage_4
        verify_stage_4_graphs(
            exp_config=exp_config,
            run_config=run_config,
            graphs_stage_4=self.graphs_stage_4,
        )


@typechecked
def output_stage_json(
    *, results_nx_graphs: Dict, filename: str, stage_index: int
) -> None:
    """Exports results dict to a json file."""

    if results_nx_graphs["graphs_dict"] == {}:
        raise Exception(
            "Error, the graphs_of_stage of stage_index="
            + f"{stage_index} was an empty dict."
        )
    verify_results_nx_graphs(
        results_nx_graphs=results_nx_graphs,
        run_config=results_nx_graphs["run_config"],
    )
    print(f"stage_index={stage_index}")
    verify_results_nx_graphs_contain_expected_stages(
        results_nx_graphs=results_nx_graphs, stage_index=stage_index
    )

    results_json_graphs = convert_digraphs_to_json(
        results_nx_graphs=results_nx_graphs, stage_index=stage_index
    )

    # TODO: Optional: ensure output files exists.
    output_filepath = f"results/{filename}.json"
    write_dict_to_json(
        output_filepath=output_filepath,
        some_dict=jsons.dump(results_json_graphs),
    )

    # Ensure output file exists.
    if not pathlib.Path(pathlib.Path(output_filepath)).resolve().is_file():
        raise Exception(f"Error:{output_filepath} does not exist.")
    # TODO: Verify the correct graphs is passed by checking the graph tag.
    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write exp_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.


@typechecked
def plot_graph_behaviours(
    *,
    filepath: str,
    stage_2_graphs: Dict,
    run_config: Run_config,
) -> None:
    """Exports the plots of the graphs per time step of the run
    configuration."""
    # TODO: get this from the experiment settings/run configuration.
    desired_props = get_desired_properties_for_graph_printing()

    # Loop over the graph types
    for graph_name, snn_graph in stage_2_graphs.items():
        if graph_name != "input_graph":
            sim_duration = snn_graph.graph["sim_duration"]
            for t in range(
                0,
                sim_duration,
            ):
                print(f"Plotting:{graph_name}, t={t}/{sim_duration}")
                # TODO: Include verify that graph len remains unmodified.
                # pylint: disable=R0913
                # TODO: reduce the amount of arguments from 6/5 to at most 5/5.
                # TODO: make plot dimensions etc. function of algorithm.
                plot_coordinated_graph(
                    extensions=run_config.export_types,
                    desired_properties=desired_props,
                    G=snn_graph,
                    height=(len(stage_2_graphs["input_graph"]) - 1) ** 2,
                    t=t,
                    filename=f"{graph_name}_{filepath}_{t}",
                    show=False,
                    title=None,
                    width=(run_config.algorithm["MDSA"]["m_val"] + 1) * 2.5,
                    zoom=run_config.zoom,
                    # title=create_custom_plot_titles(
                    #    graph_name, t, run_config.seed
                    # ),
                )
        else:
            plot_uncoordinated_graph(
                extensions=run_config.export_types,
                G=snn_graph,
                filename=f"{graph_name}_{filepath}.png",
                show=False,
            )


# pylint: disable=R0912
# pylint: disable=R0915
@typechecked
def create_custom_plot_titles(
    *, graph_name: str, t: int, seed: int
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


@typechecked
def get_desired_properties_for_graph_printing() -> List[str]:
    """Returns the properties that are to be printed to CLI."""
    desired_properties = [
        "bias",
        # "du",
        # "dv",
        "u",
        "v",
        "vth",
        "a_in_next",
    ]
    return desired_properties
