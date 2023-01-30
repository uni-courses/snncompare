"""Used to verify stages are completed."""

from typeguard import typechecked

from snncompare.exp_config.run_config.Run_config import Run_config

from ...import_results.check_completed_stages import has_outputted_stage


@typechecked
def assert_stage_is_completed(
    *,
    run_config: Run_config,
    stage_index: int,
) -> None:
    """Checks  if stage is completed, throws error if not."""
    if not has_outputted_stage(
        run_config=run_config,
        stage_index=stage_index,
    ):
        raise Exception(f"Error, stage {stage_index} was not completed.")
