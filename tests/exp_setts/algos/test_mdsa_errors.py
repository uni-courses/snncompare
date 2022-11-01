"""Tests whether MDSA algorithm specification detects invalid
specifications."""
# pylint: disable=R0801
import unittest

from src.snncompare.exp_setts.algos.get_alg_configs import (
    get_algo_configs,
    verify_algo_configs,
)
from src.snncompare.exp_setts.algos.MDSA import MDSA


class Test_mdsa(unittest.TestCase):
    """Tests whether MDSA algorithm specification detects invalid
    specifications."""

    # Initialize test object
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mdsa = MDSA(list(range(0, 4, 1)))
        self.mdsa_configs = get_algo_configs(self.mdsa.__dict__)
        verify_algo_configs("MDSA", self.mdsa_configs)

    def test_error_is_thrown_if_m_val_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the m_val key is missing from
        (one of the) the mdsa_configs."""
        # First verify the mdsa_configs are valid.
        verify_algo_configs("MDSA", self.mdsa_configs)

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        self.mdsa_configs[1].pop("m_val")
        with self.assertRaises(KeyError) as context:
            verify_algo_configs("MDSA", self.mdsa_configs)

        self.assertEqual(
            "'m_val'",
            str(context.exception),
        )

    def test_error_is_thrown_if_m_val_has_invalid_type(self) -> None:
        """Verifies an exception is thrown if the m_vals key is missing from
        the mdsa configs."""
        # First verify the mdsa_configs are valid.
        verify_algo_configs("MDSA", self.mdsa_configs)

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        self.mdsa_configs[1]["m_val"] = "somestring"
        with self.assertRaises(TypeError) as context:
            verify_algo_configs("MDSA", self.mdsa_configs)

        self.assertEqual(
            "m_val is not of type:int. Instead it is of " + f"type:{str}",
            str(context.exception),
        )

    def test_error_is_thrown_if_m_val_is_too_large(self) -> None:
        """Verifies an exception is thrown if the m_vals key is too large in
        the mdsa configs."""
        # First verify the mdsa_configs are valid.
        verify_algo_configs("MDSA", self.mdsa_configs)

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        self.mdsa_configs[2]["m_val"] = self.mdsa.max_m_vals + 1
        with self.assertRaises(ValueError) as context:
            verify_algo_configs("MDSA", self.mdsa_configs)

        self.assertEqual(
            (
                "Error, the maximum supported value for m_vals is:"
                + f"{self.mdsa.min_m_vals}, yet we found:"
                + f"{[self.mdsa_configs[2]['m_val']]}"
            ),
            str(context.exception),
        )

    def test_error_is_thrown_if_m_val_is_too_low(self) -> None:
        """Verifies an exception is thrown if the m_vals key is too low in the
        mdsa configs."""
        # First verify the mdsa_configs are valid.
        verify_algo_configs("MDSA", self.mdsa_configs)

        # Then remove one m_val parameter from a congig and assert KeyError is
        # thrown.
        self.mdsa_configs[2]["m_val"] = self.mdsa.min_m_vals - 1
        with self.assertRaises(ValueError) as context:
            verify_algo_configs("MDSA", self.mdsa_configs)

        self.assertEqual(
            (
                "Error, the minimum supported value for m_vals is:"
                + f"{self.mdsa.min_m_vals}, yet we found:"
                + f"{[self.mdsa_configs[2]['m_val']]}"
            ),
            str(context.exception),
        )
