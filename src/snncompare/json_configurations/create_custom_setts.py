"""Used to create custom experiment and run configuration files."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def create_basic_test_config(*, custom_config_path: str) -> None:
    """Creates and exports an experiment setup that can be used to quickly test
    the MDSA algorithm."""
    print(f"TODO: remove{custom_config_path}")
