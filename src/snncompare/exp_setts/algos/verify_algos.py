"""Used to verify the algorithm specifications in an experiment
configuration."""

from typing import List

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
