"""Used to verify stages are completed."""

from typing import Dict, List

from typeguard import typechecked

from snncompare.exp_config.run_config.Run_config import Run_config

from ...import_results.check_completed_stages import (
    has_outputted_images,
    has_outputted_stage_jsons,
)


@typechecked
def assert_stage_is_completed(
    *,
    expected_stages: List[int],
    run_config: Run_config,
    stage_index: int,
) -> None:
    """Checks  if stage is completed, throws error if not."""
    if not has_outputted_stage_jsons(
        expected_stages=expected_stages,
        run_config=run_config,
        stage_index=stage_index,
    ):
        raise Exception(f"Error, stage {stage_index} was not completed.")


@typechecked
def assert_stage_3_is_completed(
    *,
    results_nx_graphs: Dict,
    run_config: Run_config,
) -> None:
    """Checks  if stage 3 is completed, throws error if not.

    # TODO: assert gif file exists.
    """
    assert_stage_is_completed(
        expected_stages=[1, 2],
        run_config=run_config,
        stage_index=2,
    )

    if not has_outputted_images(
        results_nx_graphs=results_nx_graphs,
        run_config=run_config,
    ):
        raise Exception("Error, stage 3 images were not outputted.")
