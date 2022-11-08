"""Exports the test results to a json file."""
import json
from pathlib import Path

from typeguard import typechecked


@typechecked
def write_dict_to_json(output_filepath: str, some_dict: dict) -> None:
    """Writes a dict file to a .json file."""
    with open(output_filepath, "w", encoding="utf-8") as fp:
        json.dump(some_dict, fp, indent=4, sort_keys=True)

    # Verify the file exists.
    if not Path(output_filepath).is_file():
        raise Exception(f"Error, filepath:{output_filepath} was not created.")
    # TODO: verify the file content is valid.
