"""Tests whether DUMMY algorithm specification detects invalid
specifications."""
# pylint: disable=R0801
import unittest

from snnalgorithms.get_alg_configs import get_algo_configs, verify_algo_configs
from snnalgorithms.population.DUMMY import DUMMY
from typeguard import typechecked


class Test_dummy(unittest.TestCase):
    """Tests whether DUMMY algorithm specification detects invalid
    specifications."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.dummy = DUMMY(
            some_vals=list(range(4, 8, 1)),
            other_vals=["onestring", "anotherstring"],
        )
        self.dummy_configs = get_algo_configs(self.dummy.__dict__)
        verify_algo_configs("DUMMY", self.dummy_configs)

    @typechecked
    def test_error_is_thrown_if_some_val_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the some_val key is missing from
        (one of the) the dummy_configs."""
        # First verify the dummy_configs are valid.
        verify_algo_configs("DUMMY", self.dummy_configs)

        # Then remove one some_val parameter from a config and assert KeyError
        # is thrown.
        self.dummy_configs[1].pop("some_val")
        with self.assertRaises(KeyError) as context:
            verify_algo_configs("DUMMY", self.dummy_configs)

        self.assertEqual(
            "'some_val'",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_if_other_val_key_is_missing(self) -> None:
        """Verifies an exception is thrown if the other_val key is missing from
        (one of the) the dummy_configs."""
        # First verify the dummy_configs are valid.
        verify_algo_configs("DUMMY", self.dummy_configs)

        # Then remove one other_val parameter from a config and assert KeyError
        # is thrown.
        self.dummy_configs[1].pop("other_val")
        with self.assertRaises(KeyError) as context:
            verify_algo_configs("DUMMY", self.dummy_configs)

        self.assertEqual(
            "'other_val'",
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_if_some_val_has_invalid_type(self) -> None:
        """Verifies an exception is thrown if the some_vals key is missing from
        the dummy configs."""
        # First verify the dummy_configs are valid.
        verify_algo_configs("DUMMY", self.dummy_configs)

        # Then remove one some_val parameter from a congig and assert KeyError
        # is thrown.
        self.dummy_configs[1]["some_val"] = "somestring"
        with self.assertRaises(TypeError) as context:
            verify_algo_configs("DUMMY", self.dummy_configs)

        self.assertEqual(
            'type of argument "some_vals"[0] must be int; got str instead',
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_if_other_val_has_invalid_type(self) -> None:
        """Verifies an exception is thrown if the other_vals key is missing
        from the dummy configs."""
        # First verify the dummy_configs are valid.
        verify_algo_configs("DUMMY", self.dummy_configs)

        # Then remove one other_val parameter from a congig and assert KeyError
        # is thrown.
        self.dummy_configs[1]["other_val"] = 42
        with self.assertRaises(TypeError) as context:
            verify_algo_configs("DUMMY", self.dummy_configs)

        self.assertEqual(
            'type of argument "other_vals"[0] must be str; got int instead',
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_if_some_val_is_too_large(self) -> None:
        """Verifies an exception is thrown if the some_vals key is too large in
        the dummy configs."""
        # First verify the dummy_configs are valid.
        verify_algo_configs("DUMMY", self.dummy_configs)

        # Then remove one some_val parameter from a congig and assert KeyError
        # is thrown.
        self.dummy_configs[2]["some_val"] = self.dummy.max_some_vals + 1
        with self.assertRaises(ValueError) as context:
            verify_algo_configs("DUMMY", self.dummy_configs)

        self.assertEqual(
            (
                "Error, the maximum supported value for some_vals is:"
                + f"{self.dummy.min_some_vals}, yet we found:"
                + f"{[self.dummy_configs[2]['some_val']]}"
            ),
            str(context.exception),
        )

    @typechecked
    def test_error_is_thrown_if_some_val_is_too_low(self) -> None:
        """Verifies an exception is thrown if the some_vals key is too low in
        the dummy configs."""
        # First verify the dummy_configs are valid.
        verify_algo_configs("DUMMY", self.dummy_configs)

        # Then remove one some_val parameter from a congig and assert KeyError
        # is thrown.
        self.dummy_configs[2]["some_val"] = self.dummy.min_some_vals - 1
        with self.assertRaises(ValueError) as context:
            verify_algo_configs("DUMMY", self.dummy_configs)

        self.assertEqual(
            (
                "Error, the minimum supported value for some_vals is:"
                + f"{self.dummy.min_some_vals}, yet we found:"
                + f"{[self.dummy_configs[2]['some_val']]}"
            ),
            str(context.exception),
        )
