"""Exports the following structure to an output file for simsnn:

/stage_2/
    snn_algo_graph: spikes, du, dv.
    adapted_snn_algo_graph: spikes, du, dv.
    rad_snn_algo_graph: spikes, du, dv.
    rad_adapted_snn_algo_graph: spikes, du, dv.
"""
from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.run_config.Run_config import Run_config


@typechecked
def output_stage_2_snns(
    *,
    exp_config: Exp_config,
    run_config: Run_config,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
) -> None:
    """Exports results dict to a json file."""
    raise NotImplementedError
