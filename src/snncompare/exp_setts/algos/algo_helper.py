"""Contains functions that the algorithm specification files use."""
from typing import List

from typeguard import typechecked


@typechecked
def assert_parameter_is_list(parameter: List) -> None:
    """Asserts the incoming parameter is of type list.

    Throws error if it is of another type.
    """
    # Verify type of parameters.
    if not isinstance(parameter, List):
        raise TypeError(
            "some_vals is not of type:List[int]. Instead it is of "
            + f"type:{type(parameter)}"
        )
