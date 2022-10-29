"""Performs generic verification tasks."""
from typing import List


def verify_completed_stages_list(completed_stages: List) -> None:
    """Verifies the completed stages list is a list of consecutive positive
    integers.

    TODO: test this function.
    """
    start_stage = completed_stages[0]
    for next_stage in completed_stages[1:]:
        if start_stage >= next_stage:
            raise Exception(
                f"Stage indices are not consecutive:{completed_stages}."
            )
        start_stage = next_stage
    for stage in completed_stages:
        if stage < 1:
            raise Exception(
                "completed_stages contained non positive integer:"
                + f"{completed_stages}"
            )
