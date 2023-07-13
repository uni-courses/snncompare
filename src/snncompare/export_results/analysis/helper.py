"""Creates the experiment results in the form of plots with lines indicating
the performance of the SNNs."""


# Take in exp_config or run_configs
# If exp_config, get run_configs
from typing import Dict, List, Set, Tuple

import networkx as nx
from snnradiation.Rad_damage import Rad_damage
from typeguard import typechecked

from snncompare.exp_config import Exp_config
from snncompare.graph_generation.stage_1_create_graphs import (
    load_input_graph_from_file_with_init_props,
)
from snncompare.import_results.load_stage_1_and_2 import (
    has_outputted_stage_1,
    load_stage1_simsnn_graphs,
)
from snncompare.progress_report.has_completed_stage2_or_4 import (
    has_outputted_stage_2_or_4,
)
from snncompare.run_config.Run_config import Run_config


# pylint: disable=R0912
# pylint: disable=R0914
@typechecked
def get_boxplot_title(*, img_index: int, exp_config: Exp_config) -> str:
    """Removes the non-adaptation data from the y-series."""
    if len(list(exp_config.algorithms.keys())) > 1:
        raise ValueError("Error, multiple algorithms not yet supported.")
    algorithm_names: Set = set()
    for algorithm_name, algo_specs in exp_config.algorithms.items():
        algorithm_names.add(algorithm_name)
        if len(list(algorithm_names)) > 1:
            raise ValueError("Error, multiple algorithms not yet supported.")
        parameter_names: Set = set()
        parameter_values: Set = set()
        # TODO: concatenate values of dict into list.
        for algo_spec in algo_specs:
            if len(list(algo_spec.keys())) > 1:
                raise ValueError(
                    "Error, multiple parameter types not yet supported."
                )
            if len(list(algo_spec.values())) > 1:
                raise ValueError(
                    "Error, multiple parameter values not yet supported."
                )
            for param_name in algo_spec.keys():
                parameter_names.add(param_name)
            for param_value in algo_spec.values():
                parameter_values.add(param_value)
    rad_types: Set[str] = set()
    rad_probabilities: List[float] = []
    for run_config_radiation in exp_config.radiations:
        rad_types.add(run_config_radiation.effect_type)
        rad_probabilities.append(run_config_radiation.probability_per_t)
    if len(rad_types) > 1:
        raise ValueError("Error, multiple radiation types not yet supported.")

    # Drop the set accolades {}.
    str_parameter_names: str = "".join(
        str(element) for element in parameter_names
    )
    str_parameter_values: str = ",".join(
        str(element) for element in parameter_values
    )
    str_algorithm_names: str = ",".join(
        str(element) for element in algorithm_names
    )
    title: str = (
        f"{str_algorithm_names} SNN "
        + f"adaptations -\n {str_parameter_names}:{str_parameter_values}, "
        + f"radiation type:{exp_config.radiations[img_index].effect_type}"
        # + "probability: "
        # + f"{exp_config.radiations[img_index].probability_per_t*100}"
        # + f"{units}"
    )
    return title


@typechecked
def delete_non_radiation_data(*, y_series: Dict[str, List[float]]) -> None:
    """Removes the non-adaptation data from the y-series."""
    graph_names = list(y_series.keys())
    for graph_name in graph_names:
        if graph_name in ["snn_algo_graph", "adapted_snn_graph"]:
            y_series.pop(graph_name)


@typechecked
def get_image_name(*, count: int, rad_setts: Rad_damage) -> str:
    """Returns the filename for a radiation setting. Uses output structure:
    <raditation type><excitation type><probability><amplitude> Where the
    excitation types are:

     - excitatory
     - inhibitory
     - both
    All adaptation methods are included in a single box plot.
    """
    if rad_setts.excitatory and rad_setts.inhibitory:
        excitation_type: str = "both"
    elif rad_setts.excitatory:
        excitation_type = "excitatory"
    elif rad_setts.inhibitory:
        excitation_type = "inhibitory"

    return (
        f"{count}_{rad_setts.effect_type}_{excitation_type}_"
        + f"{rad_setts.probability_per_t}_amplitude_{rad_setts.amplitude}"
    )


@typechecked
def get_completed_and_missing_run_configs(
    *,
    run_configs: List[Run_config],
) -> Tuple[List[Run_config], List[Run_config]]:
    """Returns the run configs that still need to be ran."""
    missing_run_configs: List[Run_config] = []
    completed_run_configs: List[Run_config] = []

    for run_config in run_configs:
        input_graph: nx.Graph = load_input_graph_from_file_with_init_props(
            run_config=run_config
        )
        if has_outputted_stage_1(
            input_graph=input_graph,
            run_config=run_config,
        ):
            graphs_dict: Dict = load_stage1_simsnn_graphs(
                run_config=run_config,
            )
            if has_outputted_stage_2_or_4(
                graphs_dict=graphs_dict,
                run_config=run_config,
                stage_index=4,
            ):
                completed_run_configs.append(run_config)
            else:
                missing_run_configs.append(run_config)
        else:
            missing_run_configs.append(run_config)
    if len(missing_run_configs) > 0:
        print(f"Want:{len(run_configs)}, missing:{len(missing_run_configs)}")
    return completed_run_configs, missing_run_configs
