"""Gets the run_config_filepath that contains a unique_id."""

import os
from typing import List

from typeguard import typechecked

from snncompare.import_results.helper import file_contains_line


@typechecked
def get_all_filepaths_in_dir(*, root_dir: str) -> List[str]:
    """Returns a list of file paths for all Python files in root_dir and its
    subdirectories.

    Args:
    :root_dir: (str), The root directory to search for Python files.
    """
    filepaths = []
    for dir_name, _, file_list in os.walk(root_dir):
        for file_name in file_list:
            # if file_name.endswith(".py"):
            full_path = os.path.join(dir_name, file_name)
            if "build/lib" not in full_path:
                filepaths.append(full_path)
    return list(set(filepaths))


@typechecked
def get_run_config_filepath(*, run_config_unique_id: str) -> str:
    """Loops over the stage 1 directory that contains the run_config
    dictionaries, checks whether any of them contains the desired run_config
    unique_id, and if yes, returns the filepath of that file.

    Args:
    :run_config_unique_id: (str), The unique ID to search for in the
    run_config dictionaries.
    Returns:
    The filepath of the run_config file containing the desired unique ID.
    """

    relative_dir: str = f"results/stage{1}/run_configs"
    if not os.path.exists(relative_dir):
        raise FileNotFoundError(f"Error, dir:{relative_dir} does not exist.")

    run_config_filepaths = sorted(
        get_all_filepaths_in_dir(root_dir=f"{relative_dir}/")
    )

    found_run_config_filepaths: List[str] = []
    for run_config_filepath in run_config_filepaths:
        # If file
        if file_contains_line(
            filepath=run_config_filepath, expected_line=run_config_unique_id
        ):
            found_run_config_filepaths.append(run_config_filepath)

    if len(found_run_config_filepaths) == 0:
        raise FileNotFoundError(
            "Error, did not find the run_config file with Unique id:"
            f"{found_run_config_filepaths}"
        )
    if len(found_run_config_filepaths) > 1:
        raise FileNotFoundError(
            "Error, Hash collision, more than 1 file found with the same "
            + f"unique run_config id: {found_run_config_filepaths}"
        )
    filepath_without_extension: str = found_run_config_filepaths[0][:-5]

    return filepath_without_extension
