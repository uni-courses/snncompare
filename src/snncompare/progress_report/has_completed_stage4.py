"""Checks whether stage 2 has been outputted."""

from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.progress_report.has_completed_stage2 import (
    has_outputted_non_radiation_json,
    has_outputted_radiation_json,
)
from snncompare.run_config.Run_config import Run_config


@typechecked
def has_outputted_stage_4(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
) -> bool:
    """Returns True if the results dict is outputted for:

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
    return has_outputted_non_radiation_json(
        graphs_dict=graphs_dict,
        run_config=run_config,
        stage_index=4,
    ) and has_outputted_radiation_json(
        graphs_dict=graphs_dict,
        run_config=run_config,
        stage_index=4,
    )


@typechecked
def assert_has_outputted_stage_4(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
) -> None:
    """Raises exception if results have not been outputted."""
    if not has_outputted_stage_4(
        graphs_dict=graphs_dict,
        run_config=run_config,
    ):
        raise FileNotFoundError("Error, stage 4 results were not outputted.")
