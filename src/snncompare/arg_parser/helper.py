"""Assists the process args."""

from typing import List, Union


def convert_csv_list_arg_to_list(
    arg_name: str, arg_val: Union[List[str], str]
) -> List[str]:
    """Splits a csv list into a list of strings."""
    return_list: List[str] = []
    if arg_val is not None:
        if isinstance(arg_val, List):
            for elem in arg_val:
                return_list.append(str(elem))
        elif isinstance(arg_val, str):
            return_list = arg_val.split(",")
        else:
            raise TypeError(
                f"Error, {arg_name} should be csv list, got:{type(arg_val)}"
            )
    return return_list
