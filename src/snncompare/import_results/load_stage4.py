"""Loads the results for the 4 graph types:

no adaptation no radiation, adaptation no radiation, no adaptation with
radiation, adaptation with radiation,
"""
from typing import Dict, Optional

from typeguard import typechecked

from snncompare.graph_generation.stage_1_create_graphs import (
    load_input_graph_from_file_with_init_props,
)
from snncompare.helper import get_snn_graph_name
from snncompare.import_results.load_stage_1_and_2 import load_simsnn_graphs
from snncompare.run_config.Run_config import Run_config


@typechecked
def load_stage4_results(
    *,
    run_config: Run_config,
    stage_4_results_dict: Optional[Dict] = None,
) -> Dict:
    """Loads stage1 simsnn graphs and input graph."""
    if stage_4_results_dict is None:
        stage_4_results_dict = {}
        stage_4_results_dict[
            "input_graph"
        ] = load_input_graph_from_file_with_init_props(run_config=run_config)

    for with_adaptation in [False, True]:
        for with_radiation in [False, True]:
            graph_name: str = get_snn_graph_name(
                with_adaptation=with_adaptation, with_radiation=with_radiation
            )
            stage_4_results_dict[graph_name] = load_simsnn_graphs(
                run_config=run_config,
                input_graph=stage_4_results_dict["input_graph"],
                with_adaptation=with_adaptation,
                with_radiation=with_radiation,
                stage_index=4,
            )

    return stage_4_results_dict
