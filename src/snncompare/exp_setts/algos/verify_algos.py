"""Used to verify the algorithm specifications in an experiment
configuration."""


from typing import Any, List, Union

from src.snncompare.exp_setts.algos.algo_helper import assert_parameter_is_list
from src.snncompare.exp_setts.algos.get_alg_configs import verify_algo_configs


def verify_algos_in_experiment_config(exp_setts: dict) -> None:
    """Verifies an algorithm specification is valid."""
    for algo_name, algo_spec in exp_setts["algorithms"].items():
        if algo_name == "MDSA":
            verify_algo_configs("MDSA", algo_spec)
        else:
            raise NameError(
                f"Error, algo_name:{algo_name} is not yet supported."
            )


def verify_list_with_numbers(
    elem_type: type,
    min_val: Union[float, int],
    max_val: Union[float, int],
    some_vals: List[Any],
    var_name: str,
) -> None:
    """Verifies the some_vals parameter setting of the algorithm."""
    assert_parameter_is_list(some_vals)
    if not isinstance(elem_type, type):
        raise Exception(
            "Error, the elem_type is not of type type. It is "
            + f"of type:{type(elem_type)}"
        )

    # Verify values of parameters.
    for some_val in some_vals:
        if isinstance(some_val, List):
            for elem in some_val:
                verify_val_bound_and_type(
                    elem_type, min_val, max_val, elem, var_name
                )
        elif isinstance(some_val, (float, int)):
            verify_val_bound_and_type(
                elem_type, min_val, max_val, some_val, var_name
            )


def verify_val_bound_and_type(
    elem_type: type,
    min_val: Union[float, int],
    max_val: Union[float, int],
    some_val: Union[int, float],
    var_name: str,
) -> None:
    """Verifies an incoming number is of expected type, and that its value is
    bounded."""
    # Verify type of parameters
    if not isinstance(some_val, elem_type):
        raise TypeError(
            f"{var_name} is not of type:{elem_type}. Instead it is of "
            + f"type:{type(some_val)}"
        )
    if some_val < min_val:
        raise ValueError(
            f"Error, the minimum supported value for {var_name} is:"
            + f"{min_val}, yet we found:{some_val}"
        )
    if some_val > max_val:
        raise ValueError(
            f"Error, the maximum supported value for {var_name} is:"
            + f"{max_val}, yet we found:{some_val}"
        )
