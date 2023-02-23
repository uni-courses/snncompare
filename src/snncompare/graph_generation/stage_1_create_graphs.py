"""Gets the input graphs that may be adapted in later stages.

Takes run config of an experiment config as input, and returns a
networkx Graph.
"""

import copy
from typing import Dict, List

import networkx as nx
from snnadaptation.redundancy.redundancy import apply_redundancy
from snnadaptation.redundancy.verify_redundancy_settings import (
    verify_redundancy_settings_for_run_config,
)
from snnalgorithms.sparse.MDSA.create_MDSA_snn_neurons import (
    get_new_mdsa_graph,
)
from snnalgorithms.sparse.MDSA.SNN_initialisation_properties import (
    SNN_initialisation_properties,
)
from snnalgorithms.Used_graphs import Used_graphs
from snnradiation.Radiation_damage import (
    Radiation_damage,
    verify_radiation_is_applied,
)
from typeguard import typechecked

from snncompare.export_plots.Plot_config import Plot_config
from snncompare.run_config.Run_config import Run_config

from ..helper import add_stage_completion_to_graph


@typechecked
def get_used_graphs(
    *,
    plot_config: Plot_config,
    run_config: Run_config,
) -> Dict:
    """First gets the input graph.

    Then generates a graph with adaptation if it is required. Then
    generates a graph with radiation if it is required. Then returns
    this list of graphs.
    """
    # TODO: move to central place in MDSA algo spec.
    graphs = {}
    graphs["input_graph"] = get_input_graph_of_run_config(
        run_config=run_config
    )

    graphs["snn_algo_graph"] = get_new_mdsa_graph(
        run_config=run_config, input_graph=graphs["input_graph"]
    )

    # TODO: write test to verify the algorithm yields valid results on default
    # MDSA SNN.
    if has_adaptation(run_config=run_config):
        graphs["adapted_snn_graph"] = get_adapted_graph(
            snn_algo_graph=graphs["snn_algo_graph"],
            plot_config=plot_config,
            run_config=run_config,
        )

    if has_radiation(run_config=run_config):
        graphs["rad_snn_algo_graph"] = get_radiation_graph(
            snn_graph=graphs["snn_algo_graph"],
            run_config=run_config,
            seed=run_config.seed,
        )

        if has_adaptation(run_config=run_config):
            graphs["rad_adapted_snn_graph"] = get_radiation_graph(
                snn_graph=graphs["adapted_snn_graph"],
                run_config=run_config,
                seed=run_config.seed,
            )
    # TODO: move this into a separate location/function.
    # Indicate the graphs have completed stage 1.
    for graph in graphs.values():
        add_stage_completion_to_graph(input_graph=graph, stage_index=1)
    return graphs


@typechecked
def get_input_graph_of_run_config(
    *,
    run_config: Run_config,
) -> nx.Graph:
    """TODO: support retrieving graph sizes larger than size 5.
    TODO: ensure those graphs have valid properties, e.g. triangle-free and
    non-planar."""

    # Get the graph of the right size.
    # TODO: Pass random seed.
    input_graph: nx.Graph = get_input_graphs(run_config=run_config)[
        run_config.graph_nr
    ]

    # TODO: Verify the graphs are valid (triangle free and planar for MDSA).
    return input_graph


@typechecked
def get_input_graphs(
    *,
    run_config: Run_config,
) -> List[nx.Graph]:
    """Removes graphs that are not used, because of a maximum nr of graphs that
    is to be evaluated."""
    used_graphs = Used_graphs()
    input_graphs: List[nx.DiGraph] = used_graphs.get_graphs(
        run_config.graph_size
    )
    if len(input_graphs) > run_config.graph_nr:
        for input_graph in input_graphs:
            # TODO: set alg_props:
            if "alg_props" not in input_graph.graph.keys():
                input_graph.graph["alg_props"] = SNN_initialisation_properties(
                    input_graph, run_config.seed
                ).__dict__

            if not isinstance(input_graph, nx.Graph):
                raise Exception(
                    "Error, the input graph is not a networkx graph:"
                    + f"{type(input_graph)}"
                )

        return input_graphs
    raise Exception(
        f"For input_graph of size:{run_config.graph_size}, I found:"
        + f"{len(input_graphs)} graphs, yet expected graph_nr:"
        + f"{run_config.graph_nr}. Please lower the max_graphs setting in:"
        + "size_and_max_graphs in the experiment configuration."
    )


@typechecked
def get_adapted_graph(
    *,
    snn_algo_graph: nx.DiGraph,
    plot_config: Plot_config,
    run_config: Run_config,
) -> nx.DiGraph:
    """Converts an input graph of stage 1 and applies a form of brain-inspired
    adaptation to it."""

    for adaptation_name, adaptation_setting in run_config.adaptation.items():

        if adaptation_name is None:
            raise Exception(
                "Error, if no adaptation is selected, this method should not"
                + " be reached."
            )
        if adaptation_name == "redundancy":
            verify_redundancy_settings_for_run_config(
                adaptation=run_config.adaptation
            )
            adaptation_graph: nx.DiGraph = get_redundant_graph(
                snn_algo_graph=snn_algo_graph,
                plot_config=plot_config,
                red_lev=adaptation_setting,
            )
            return adaptation_graph
        raise Exception(
            f"Error, adaptation_name:{adaptation_name} is not" + " supported."
        )


@typechecked
def has_adaptation(
    *,
    run_config: Run_config,
) -> bool:
    """Checks if the adaptation contains a None setting.

    TODO: ensure the adaptation only consists of 1 setting per run.
    TODO: throw an error if the adaptation settings contain multiple
    settings, like "redundancy" and "None" simultaneously.
    """
    if run_config.adaptation is None:
        return False
    for adaptation_name in run_config.adaptation.keys():
        if adaptation_name is not None:
            return True
    return False


@typechecked
def has_radiation(
    *,
    run_config: Run_config,
) -> bool:
    """Checks if the radiation contains a None setting.

    TODO: ensure the radiation only consists of 1 setting per run.
    TODO: throw an error if the radiation settings contain multiple
    settings, like "redundancy" and "None" simultaneously.
    """
    if run_config.radiation is None:
        return False
    for radiation_name in run_config.radiation.keys():
        if radiation_name is not None:
            return True
    return False


@typechecked
def get_redundant_graph(
    *, snn_algo_graph: nx.DiGraph, plot_config: Plot_config, red_lev: int
) -> nx.DiGraph:
    """Returns a networkx graph that has a form of adaptation added."""
    adaptation_graph = copy.deepcopy(snn_algo_graph)
    apply_redundancy(
        adaptation_graph=adaptation_graph,
        plot_config=plot_config,
        redundancy=red_lev,
    )
    return adaptation_graph


@typechecked
def get_radiation_graph(
    *,
    snn_graph: nx.DiGraph,
    run_config: Run_config,
    seed: int,
) -> nx.Graph:
    """Makes a deep copy of the incoming graph and applies radiation to it.

    Then returns the graph with the radiation, as well as a list of
    neurons that are dead.
    """

    # TODO: determine on which graphs to apply the adaptation.

    # TODO: Verify incoming graph has valid SNN properties.

    # TODO: Check different radiation simulation times.

    # Apply radiation simulation.

    for radiation_name, radiation_setting in run_config.radiation.items():

        if radiation_name is None:
            raise Exception(
                "Error, if no radiation is selected, this method should not"
                + " be reached."
            )
        if radiation_name == "neuron_death":
            if not isinstance(radiation_setting, float):
                raise Exception(
                    f"Error, radiation_setting={radiation_setting},"
                    + "which is not an int."
                )

            rad_dam = Radiation_damage(probability=radiation_setting)
            radiation_graph: nx.DiGraph = copy.deepcopy(snn_graph)
            dead_neuron_names = rad_dam.inject_simulated_radiation(
                get_degree=radiation_graph,
                probability=rad_dam.neuron_death_probability,
                seed=seed,
            )
            # TODO: verify radiation is injected with V1000
            verify_radiation_is_applied(
                some_graph=radiation_graph,
                dead_neuron_names=dead_neuron_names,
                rad_type=radiation_name,
            )

            return radiation_graph
        raise Exception(
            f"Error, radiation_name:{radiation_name} is not supported."
        )
