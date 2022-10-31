"""Used to verify the algorithm specifications in an experiment
configuration."""

from src.snncompare.exp_setts.algos.get_alg_configs import verify_mdsa_configs


def verify_algos_in_experiment_config(exp_setts: dict) -> None:
    """Verifies an algorithm specification is valid."""
    for algo_name, algo_spec in exp_setts["algorithms"].items():
        if algo_name == "MDSA":
            verify_mdsa_configs("MDSA", algo_spec)
        else:
            raise NameError(
                f"Error, algo_name:{algo_name} is not yet supported."
            )
