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
from typing import Dict

from typeguard import typechecked

from snncompare.exp_config.Exp_config import Exp_config
from snncompare.run_config.Run_config import Run_config


# pylint: disable=W0613
@typechecked
def verify_stage_3_graphs(
    *,
    exp_config: Exp_config,
    run_config: Run_config,
    graphs_stage_3: Dict,
) -> None:
    """Verifies the generated graphs are compliant and complete for the
    specified run configuration.

    Args:
    :exp_config: (Exp_config), The experiment configuration for which the
    graphs are generated.
    :run_config: (Run_config), The run configuration for which the graphs
    are generated.
    :graphs_stage_3: (Dict), A dictionary containing the generated graphs
    for stage 3 of the experiment.
    Returns:
    This function does not return any value, it verifies the compliance and
    completeness of generated graphs for the specified configurations.
    """
    # TODO: Verify run_config is valid "subset" of experiment config.# Compute which graphs are expected, based on run config.# Verify the graphs that are required for the run_config are generated.

    # TODO: verify the properties required by the run config are in the graphs.
