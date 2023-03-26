"""Checks whether stage 2 has been outputted."""

from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    get_radiation_names_and_hash,
)
from snncompare.progress_report.has_completed_stage1 import (
    has_outputted_snn_graph,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def has_outputted_stage_2(
    *,
    input_graph: nx.Graph,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
) -> bool:
    """Returns True if the I,V,Spike values are outputted as lists for:

    - snn_algo_graph
    stage2/algorithm_name+setting/no_adaptation/isomorphichash+rand_hash
    - adapted_snn_algo_graph
    stage2/algorithm_name+setting/adaptation_type/isomorphichash+rand_hash
    - rad_snn_algo_graph
    stage2/algorithm_name+setting/no_adaptation/isomorphichash+rand_hash
    - rad_adapted_snn_algo_graph
    stage2/algorithm_name+setting/adaptation_type/isomorphichash+rand_hash+rad_affected_neurons_hash
    under filenames:
    """
    for with_radiation in [False, True]:
        print(f"with_radiation={with_radiation}")
        radiation_parameter: float = 5000000.0  # TODO: change this.
        for with_adaptation in [False, True]:
            if with_adaptation:
                snn_graph = graphs_dict["adapted_snn_graph"]
            else:
                snn_graph = graphs_dict["snn_algo_graph"]
            (
                affected_neurons,
                rad_affected_neurons_hash,
            ) = get_radiation_names_and_hash(
                snn_graph=snn_graph,
                radiation_parameter=radiation_parameter,
                run_config=run_config,
            )
            print(f"affected_neurons={affected_neurons}")
            if not has_outputted_snn_graph(
                input_graph=input_graph,
                run_config=run_config,
                with_adaptation=with_adaptation,
                stage_index=2,
                rad_affected_neurons_hash=rad_affected_neurons_hash,
            ):
                return False
    return True
