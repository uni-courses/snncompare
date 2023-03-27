"""Gets the input graphs that may be adapted in later stages.

Takes run config of an experiment config as input, and returns a
networkx Graph.
"""

import copy
from math import inf
from typing import Dict, List, Union

import networkx as nx
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
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
from snnbackends.networkx.LIF_neuron import LIF_neuron
from snnbackends.simsnn.simsnn_to_nx_lif import simsnn_graph_to_nx_lif_graph
from snnradiation.Radiation_damage import (
    Radiation_damage,
    verify_radiation_is_applied,
)
from typeguard import typechecked

from snncompare.export_plots.Plot_config import Plot_config
from snncompare.run_config.Run_config import Run_config


@typechecked
def get_graphs_stage_1(
    *,
    plot_config: Plot_config,
    run_config: Run_config,
) -> Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]]:
    """Returns the initialised graphs for stage 1 for the different
    simulators."""
    stage_1_graphs: Dict[
        str, Union[nx.Graph, nx.DiGraph, Simulator]
    ] = get_nx_lif_graphs(
        plot_config=plot_config,
        run_config=run_config,
    )
    if run_config.simulator == "nx":
        return stage_1_graphs
    if run_config.simulator == "simsnn":
        return nx_lif_graphs_to_simsnn_graphs(
            stage_1_graphs=stage_1_graphs,
            reverse_conversion=False,
            run_config=run_config,
        )
    raise NotImplementedError(
        "Error, did not yet implement simsnn to nx_lif converter."
    )


@typechecked
def nx_lif_graphs_to_simsnn_graphs(
    *,
    stage_1_graphs: Dict,
    reverse_conversion: bool,
    run_config: Run_config,
) -> Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]]:
    """Converts nx_lif graphs to sim snn graphs."""
    new_graphs: Dict = {}
    new_graphs["input_graph"] = stage_1_graphs["input_graph"]
    if "alg_props" not in new_graphs["input_graph"].graph.keys():
        new_graphs["input_graph"].graph[
            "alg_props"
        ] = SNN_initialisation_properties(
            G=new_graphs["input_graph"], seed=run_config.seed
        ).__dict__

    for graph_name in stage_1_graphs.keys():
        # if graph_name in ["snn_algo_graph", "adapted_snn_graph",]:
        if graph_name in [
            "snn_algo_graph",
            "adapted_snn_graph",
        ]:
            if reverse_conversion:
                new_graphs[graph_name] = simsnn_graph_to_nx_lif_graph(
                    simsnn=stage_1_graphs[graph_name]
                )
            else:
                new_graphs[graph_name] = nx_lif_graph_to_simsnn_graph(
                    snn_graph=stage_1_graphs[graph_name],
                    add_to_multimeter=True,
                    add_to_raster=True,
                )

    return new_graphs


@typechecked
def nx_lif_graph_to_simsnn_graph(
    *,
    snn_graph: nx.DiGraph,
    add_to_multimeter: bool,
    add_to_raster: bool,
) -> Simulator:
    """Converts an snn graph of type nx_LIF to sim snn graph."""
    net = Network()
    sim = Simulator(net, monitor_I=True)

    simsnn: Dict[str, LIF] = {}
    for node_name in snn_graph.nodes:
        nx_lif: LIF_neuron = snn_graph.nodes[node_name]["nx_lif"][0]
        simsnn[node_name] = net.createLIF(
            m=1,
            bias=nx_lif.bias.get(),
            V_init=0,
            V_reset=0,
            V_min=-inf,
            thr=nx_lif.vth.get(),
            amplitude=1,
            I_e=0,
            noise=0,
            rng=0,
            ID=0,
            name=node_name,
            increment_count=False,
            du=nx_lif.du.get(),
            pos=nx_lif.pos,
            spike_only_if_thr_exceeded=True,
        )
    for edge in snn_graph.edges():
        synapse = snn_graph.edges[edge]["synapse"]
        # pylint: disable=R0801
        net.createSynapse(
            pre=simsnn[edge[0]],
            post=simsnn[edge[1]],
            ID=edge,
            w=synapse.weight,
            d=1,
        )

    if add_to_raster:
        # Add all neurons to the raster.
        sim.raster.addTarget(net.nodes)
    if add_to_multimeter:
        # Add all neurons to the multimeter.
        sim.multimeter.addTarget(net.nodes)

    # Add (redundant) graph properties.
    sim.network.graph.graph = snn_graph.graph
    return sim


@typechecked
def get_nx_lif_graphs(
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
                raise TypeError(
                    "Error, the input graph is not a networkx graph:"
                    + f"{type(input_graph)}"
                )

        return input_graphs
    raise ValueError(
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
            raise ValueError(
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
        raise NotImplementedError(
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
            raise SystemError(
                "Error, if no radiation is selected, this method should not"
                + " be reached."
            )
        if radiation_name == "neuron_death":
            if not isinstance(radiation_setting, float):
                raise TypeError(
                    f"Error, radiation_setting={radiation_setting},"
                    + "which is not an int."
                )

            rad_dam = Radiation_damage(probability=radiation_setting)
            radiation_graph: nx.DiGraph = copy.deepcopy(snn_graph)
            dead_neuron_names = rad_dam.inject_simulated_radiation(
                snn_graph=radiation_graph,
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
        raise NotImplementedError(
            f"Error, radiation_name:{radiation_name} is not supported."
        )
