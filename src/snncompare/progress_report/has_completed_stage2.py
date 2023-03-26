"""Checks whether stage 2 has been outputted."""

from typing import Dict, Union

import networkx as nx
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snncompare.export_results.output_stage1_configs_and_input_graph import (
    Radiation_output_data,
    get_radiation_names_filepath_and_exists,
    get_rand_nrs_and_hash,
)
from snncompare.import_results.helper import simsnn_files_exists_and_get_path
from snncompare.run_config.Run_config import Run_config


@typechecked
def has_outputted_stage_2(
    *,
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
    return has_outputted_stage_2_non_radiation_snns(
        graphs_dict=graphs_dict,
        run_config=run_config,
    ) and has_outputted_stage_2_radiation_snns(
        graphs_dict=graphs_dict,
        run_config=run_config,
    )


def has_outputted_stage_2_non_radiation_snns(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
) -> bool:
    """Returns False if not both rad_snn_algo_graph and
    rad_adapted_snn_algo_graph files exist."""
    for with_adaptation in [False, True]:
        _, rand_nrs_hash = get_rand_nrs_and_hash(
            input_graph=graphs_dict["input_graph"]
        )
        simsnn_exists, _ = simsnn_files_exists_and_get_path(
            output_category="snns",
            input_graph=graphs_dict["input_graph"],
            run_config=run_config,
            with_adaptation=with_adaptation,
            stage_index=2,
            rand_nrs_hash=rand_nrs_hash,
        )
        if not simsnn_exists:
            return False
    return True


def has_outputted_stage_2_radiation_snns(
    *,
    graphs_dict: Dict[str, Union[nx.Graph, nx.DiGraph, Simulator]],
    run_config: Run_config,
) -> bool:
    """Returns False if not both rad_snn_algo_graph and
    rad_adapted_snn_algo_graph files exist."""
    for with_adaptation in [False, True]:
        radiation_output_data: Radiation_output_data = (
            get_radiation_names_filepath_and_exists(
                graphs_dict=graphs_dict,
                run_config=run_config,
                stage_index=2,
                with_adaptation=with_adaptation,
            )
        )
        if not radiation_output_data.radiation_file_exists:
            return False
    return True
