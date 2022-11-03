"""Used to verify stages are completed."""
from src.snncompare.import_results.check_completed_stages import (
    has_outputted_stage,
)


def assert_stage_is_completed(
    run_config: dict, stage_index: int, to_run: dict
) -> None:
    """Checks  if stage is completed, throws error if not."""
    if not has_outputted_stage(run_config, stage_index, to_run):
        raise Exception(f"Error, stage {stage_index} was not completed.")
