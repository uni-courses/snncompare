"""Determines which algorithm(s) are ran, then determines for which graphs the
results are to be computed, and then computes the results per graph, against
the expected results. A different algorithm may return a different dictionary.

The MDSA algorithm results will consist of a list of nodes per used
graph that have been selected according to Alipour, and according to the
respective SNN graph.
"""
from typing import Dict

from simsnn.core.simulators import Simulator
from snnalgorithms.sparse.MDSA.apply_results_to_graphs import (
    set_mdsa_snn_results,
)
from snnbackends.verify_nx_graphs import verify_snn_contains_correct_stages
from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.optional_config.Output_config import Output_config
from snncompare.run_config.Run_config import Run_config
from snncompare.simulation.stage2_sim import stage_2_or_4_graph_exists_already

from ..helper import (
    add_stage_completion_to_graph,
    get_expected_stages,
    get_with_adaptation_bool,
    get_with_radiation_bool,
)
from ..import_results.check_completed_stages import (
    nx_graphs_have_completed_stage,
)


@typechecked
def set_results(
    *,
    exp_config: Exp_config,
    output_config: Output_config,
    run_config: Run_config,
    stage_2_graphs: Dict,
) -> bool:
    """Gets the results for the algorithms that have been ran."""
    for algo_name, algo_settings in run_config.algorithm.items():
        if algo_name == "MDSA":
            if isinstance(algo_settings["m_val"], int):
                return perform_mdsa_results_computation_if_needed(
                    exp_config=exp_config,
                    m_val=algo_settings["m_val"],
                    output_config=output_config,
                    run_config=run_config,
                    stage_2_graphs=stage_2_graphs,
                )
            # pylint: disable=R0801
            raise TypeError(
                "Error, m_val setting is not of type int:"
                f'{type(algo_settings["m_val"])}'
                f'm_val={algo_settings["m_val"]}'
            )
        raise NotImplementedError(
            f"Error, algo_name:{algo_name} is not (yet) supported."
        )
    return False


@typechecked
def perform_mdsa_results_computation_if_needed(
    *,
    exp_config: Exp_config,
    m_val: int,
    output_config: Output_config,
    run_config: Run_config,
    stage_2_graphs: Dict,
) -> bool:
    """Performs result computation if the results are not in the graph yet.

    TODO: separate check whether computation is needed, with setting results.
    """
    set_new_results: bool = False

    for graph_name, snn in stage_2_graphs.items():
        if isinstance(snn, Simulator):
            graph = snn.network.graph
        else:
            graph = snn
        if graph_name != "input_graph":
            with_adaptation: bool = get_with_adaptation_bool(
                graph_name=graph_name
            )
            with_radiation: bool = get_with_radiation_bool(
                graph_name=graph_name
            )

            if not stage_2_or_4_graph_exists_already(
                input_graph=stage_2_graphs["input_graph"],
                stage_1_graphs=stage_2_graphs,
                run_config=run_config,
                with_adaptation=with_adaptation,
                with_radiation=with_radiation,
                stage_index=4,
            ):
                if (
                    4 not in graph.graph["completed_stages"]
                    or 4 in output_config.recreate_stages
                ):
                    set_new_results = True
                    set_mdsa_snn_results(
                        exp_config=exp_config,
                        m_val=m_val,
                        output_config=output_config,
                        run_config=run_config,
                        stage_2_graphs=stage_2_graphs,
                    )

        # Indicate the graphs have completed stage 1.
        add_stage_completion_to_graph(snn=graph, stage_index=4)
    return set_new_results


@typechecked
def verify_stage_completion(
    *,
    results_nx_graphs: Dict,
    simulator: str,
    stage_index: int,
) -> None:
    """Integrates the results per graph type into the graph, then export the
    results dictionary (again) into the stage 4 folder."""

    # Verify the results_nx_graphs are valid.
    if simulator == "nx":
        nx_graphs_have_completed_stage(
            run_config=results_nx_graphs["run_config"],
            results_nx_graphs=results_nx_graphs,
            stage_index=4,
        )

    # Export graphs with embedded results to json.
    # TODO: move export into separate function.
    for graph_name, nx_graph in results_nx_graphs["graphs_dict"].items():
        verify_snn_contains_correct_stages(
            graph_name=graph_name,
            snn=nx_graph,
            expected_stages=get_expected_stages(
                stage_index=stage_index,
            ),
        )
