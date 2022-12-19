"""Used to verify stages are completed."""

from typeguard import typechecked

from snncompare.exp_setts.run_config.Run_config import Run_config

from ...import_results.check_completed_stages import has_outputted_stage


@typechecked
def assert_stage_is_completed(
    run_config: Run_config,
    stage_index: int,
    to_run: dict,
    verbose: bool = False,
) -> None:
    """Checks  if stage is completed, throws error if not."""
    if not has_outputted_stage(
        run_config, stage_index, to_run, verbose=verbose
    ):
        raise Exception(f"Error, stage {stage_index} was not completed.")
