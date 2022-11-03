"""Verifies the graphs that are provided for the output of the first stage of
the experiment.

Input: Experiment configuration.
    SubInput: Run configuration within an experiment.
        Stage 1: The networkx graphs that will be propagated.
        Stage 2: The propagated networkx graphs (at least one per timestep).
        Stage 3: Visaualisation of the networkx graphs over time.
        Stage 4: Post-processed performance data of algorithm and adaptation
        mechanism.
"""


# pylint: disable=W0613
from typing import List

from src.snncompare.graph_generation.stage_1_get_input_graphs import (
    has_adaptation,
    has_radiation,
)


def verify_stage_1_graphs(
    experiment_config: dict, run_config: dict, graphs: dict
) -> None:
    """Verifies the generated graphs are compliant and complete for the
    specified run configuration.

    An experiment may consist of multiple runs.
    """
    # TODO: Verify run_config is valid "subset" of experiment config.

    # Verify the graphs that are required for the run_config are generated.
    assert_graphs_are_in_dict(run_config, graphs, 1)

    # TODO: verify the properties required by the run config are in the graphs.


def get_expected_stage_1_graph_names(run_config: dict) -> List[str]:
    """Parses the run config and returns a list with the graph names that are
    expected at the end of stage 1."""

    # TODO: make into hash
    expected_graph_names = ["input_graph", "snn_algo_graph"]
    if has_adaptation(run_config):
        expected_graph_names.append("adapted_snn_graph")

    if has_radiation(run_config):
        expected_graph_names.append("rad_snn_algo_graph")
        expected_graph_names.append("rad_adapted_snn_graph")
    return expected_graph_names


def expected_graphs_are_in_dict(
    run_config: dict, graphs: dict, stage: int
) -> bool:
    """Gets the graphs that are expected in the dict, and returns True if they
    are found in the list of graphs."""

    if stage == 1:
        # Compute which graphs are expected, based on run config.
        expected_graphs = get_expected_stage_1_graph_names(run_config)
    else:
        # TODO: implement.
        raise Exception(f"Stage {stage} not yet implemented.")

    for expected_graph_name in expected_graphs:
        if expected_graph_name not in graphs:
            return False
    return True


def assert_graphs_are_in_dict(
    run_config: dict, graphs: dict, stage: int
) -> None:
    """Throws error if the not all the expected graphs are in the list of
    graphs."""
    if not expected_graphs_are_in_dict(run_config, graphs, stage):
        raise Exception(f"Error, graph is missing:{graphs},stage:{stage}")
