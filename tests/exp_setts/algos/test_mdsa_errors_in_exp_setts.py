"""Tests whether MDSA algorithm specification detects invalid
specifications."""
# pylint: disable=R0801
import copy
import unittest

from src.snncompare.exp_setts.algos.get_alg_configs import (
    get_algo_configs,
    verify_algo_configs,
)
from src.snncompare.exp_setts.algos.MDSA import MDSA
from src.snncompare.exp_setts.verify_experiment_settings import (
    verify_experiment_config,
)
from tests.exp_setts.exp_setts.test_generic_experiment_settings import (
    adap_sets,
    rad_sets,
    supp_exp_setts,
    with_adaptation_with_radiation,
)


class Test_mdsa(unittest.TestCase):
    """Tests whether MDSA algorithm specification detects invalid
    specifications."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mdsa = MDSA(list(range(0, 4, 1)))
        self.mdsa_configs = get_algo_configs(self.mdsa.__dict__)
        verify_algo_configs("MDSA", self.mdsa_configs)

        # Create experiment settings.
        self.supp_exp_setts = supp_exp_setts
        self.adap_sets = adap_sets
        self.rad_sets = rad_sets
        self.with_adaptation_with_radiation = with_adaptation_with_radiation
        self.valid_iterations = self.supp_exp_setts.iterations

    def test_error_is_thrown_if_m_val_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the m_val key is missing from
        (one of the) the mdsa_configs."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)

        # First verify the mdsa_configs are valid.
        verify_experiment_config(
            self.supp_exp_setts,
            experiment_config,
            has_unique_id=False,
            strict=True,
        )

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        self.mdsa_configs[1].pop("m_val")
        experiment_config["algorithms"]["MDSA"][1].pop("m_val")
        with self.assertRaises(KeyError) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "'m_val'",
            str(context.exception),
        )

    def test_error_is_thrown_if_m_val_has_invalid_type(self) -> None:
        """Verifies an exception is thrown if the m_vals key is missing from
        the mdsa configs."""

        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)

        # First verify the mdsa_configs are valid.
        verify_experiment_config(
            self.supp_exp_setts,
            experiment_config,
            has_unique_id=False,
            strict=True,
        )

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        experiment_config["algorithms"]["MDSA"][1]["m_val"] = "somestring"
        with self.assertRaises(TypeError) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            "m_val is not of type:int. Instead it is of " + f"type:{str}",
            str(context.exception),
        )

    def test_error_is_thrown_if_m_val_is_too_large(self) -> None:
        """Verifies an exception is thrown if the m_vals key is too large in
        the mdsa configs."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)

        # First verify the mdsa_configs are valid.
        verify_experiment_config(
            self.supp_exp_setts,
            experiment_config,
            has_unique_id=False,
            strict=True,
        )

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        # self.mdsa_configs[2]["m_val"] = self.mdsa.max_m_vals + 1
        experiment_config["algorithms"]["MDSA"][2]["m_val"] = (
            self.mdsa.max_m_vals + 1
        )
        with self.assertRaises(ValueError) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            (
                "Error, the maximum supported value for m_vals is:"
                + f"{self.mdsa.min_m_vals}, yet we found:"
                + f'{[experiment_config["algorithms"]["MDSA"][2]["m_val"]]}'
            ),
            str(context.exception),
        )

    def test_error_is_thrown_if_m_val_is_too_low(self) -> None:
        """Verifies an exception is thrown if the m_vals key is too low in the
        mdsa configs."""
        # Create deepcopy of configuration settings.
        experiment_config = copy.deepcopy(self.with_adaptation_with_radiation)

        # First verify the mdsa_configs are valid.
        verify_experiment_config(
            self.supp_exp_setts,
            experiment_config,
            has_unique_id=False,
            strict=True,
        )

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        # self.mdsa_configs[2]["m_val"] = self.mdsa.min_m_vals - 1
        experiment_config["algorithms"]["MDSA"][2]["m_val"] = (
            self.mdsa.min_m_vals - 1
        )
        with self.assertRaises(ValueError) as context:
            verify_experiment_config(
                self.supp_exp_setts,
                experiment_config,
                has_unique_id=False,
                strict=True,
            )

        self.assertEqual(
            (
                "Error, the minimum supported value for m_vals is:"
                + f"{self.mdsa.min_m_vals}, yet we found:"
                + f'{[experiment_config["algorithms"]["MDSA"][2]["m_val"]]}'
            ),
            str(context.exception),
        )