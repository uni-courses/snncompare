"""Contains the output of an experiment run at 4 different stages.

Input: Experiment configuration.
SubInput: Run configuration within an experiment.
    Stage 1: The networkx graphs that will be propagated.
    Stage 2: The propagated networkx graphs (at least one per timestep).
    Stage 3: Visaualisation of the networkx graphs over time.
    Stage 4: Post-processed performance data of algorithm and adaptation
    mechanism.
"""
import copy
import pathlib
from typing import Dict

import jsons
import networkx as nx
from networkx.readwrite import json_graph
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.export_plots.plot_graphs import plot_uncoordinated_graph
from snncompare.export_results.export_nx_graph_to_json import digraph_to_json
from snncompare.export_results.prepare_output_stage_1 import (
    prepare_stage_1_and_2_nx_lif_output,
    prepare_stage_1_and_2_simsnn_output,
)
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config

from .export_json_results import write_dict_to_json
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
# run_config_to_filename(run_config_dict)
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
    *, results_nx_graphs: Dict, run_config_filename: str, stage_index: int
) -> None:
    """Exports results dict to a json file."""

    if results_nx_graphs["graphs_dict"] == {}:
        raise ValueError(
            "Error, the graphs_of_stage of stage_index="
            + f"{stage_index} was an empty dict."
        )

    input_graph: nx.Graph = results_nx_graphs["graphs_dict"]["input_graph"]
    if results_nx_graphs["run_config"].simulator == "nx":
        results_nx_graphs = prepare_stage_1_and_2_nx_lif_output(
            results_nx_graphs=results_nx_graphs,
            stage_index=stage_index,
        )

    elif results_nx_graphs["run_config"].simulator == "simsnn":
        results_nx_graphs["graphs_dict"] = prepare_stage_1_and_2_simsnn_output(
            graphs_dict=results_nx_graphs["graphs_dict"],
        )
        results_nx_graphs["graphs_dict"]["input_graph"] = digraph_to_json(
            G=input_graph
        )

    else:
        raise NotImplementedError(
            "Error, did not yet implement simsnn to nx_lif converter."
        )

    # Convert Run_config and Exp_config into dicts before jsons.dump. Done to
    # elope warning on Exp_config.adaptations = None (optional argument), which
    # cannot be dumped into dict.
    exported_dict = copy.deepcopy(results_nx_graphs)
    for key, val in exported_dict.items():
        if not isinstance(val, Dict):
            exported_dict[key] = val.__dict__

    output_filepath = f"results/{run_config_filename}.json"
    write_dict_to_json(
        output_filepath=output_filepath,
        some_dict=jsons.dump(exported_dict),
    )

    # Ensure output file exists.
    if not pathlib.Path(pathlib.Path(output_filepath)).resolve().is_file():
        raise FileNotFoundError(f"Error:{output_filepath} does not exist.")
    # TODO: Verify the correct graphs is passed by checking the graph tag.
    # TODO: merge experiment config, run_config and graphs into single dict.
    # TODO: Write exp_config to file (pprint(dict), or json)
    # TODO: Write run_config to file (pprint(dict), or json)
    # TODO: Write graphs to file (pprint(dict), or json)
    # TODO: append tags to output file.

    # Revert input graph back from json to dict.
    results_nx_graphs["graphs_dict"][
        "input_graph"
    ] = json_graph.node_link_graph(
        results_nx_graphs["graphs_dict"]["input_graph"]
    )
    if not isinstance(
        results_nx_graphs["graphs_dict"]["input_graph"], nx.Graph
    ):
        raise TypeError(
            "Error, input graph was not of type nx.DiGraph:"
            + f'{type(results_nx_graphs["graphs_dict"]["input_graph"])}'
        )


@typechecked
def plot_graph_behaviours(
    *,
    run_config_filename: str,
    output_config: Output_config,
    stage_2_graphs: Dict,
) -> None:
    """Exports the plots of the graphs per time step of the run
    configuration."""

    # Loop over the graph types
    for graph_name, snn_graph in stage_2_graphs.items():
        if graph_name == "input_graph":
            plot_uncoordinated_graph(
                extensions=output_config.export_types,
                G=snn_graph,
                filename=f"{graph_name}_{run_config_filename}.png",
                show=False,
            )
