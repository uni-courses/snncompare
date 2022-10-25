"""Used to verify stages are completed."""
from src.import_results.check_completed_stages import has_outputted_stage


def assert_stage_is_completed(run_config: dict, stage_index: int):
    """Checks  if stage is completed, throws error if not."""
    # TODO: Verify stage 1 is completed.
    if not has_outputted_stage(run_config, stage_index):
        raise Exception(f"Error, stage {stage_index} was not completed.")
