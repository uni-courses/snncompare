"""Contains functions used to help the tests."""
import pathlib


def assertIsFile(path):
    """Asserts a file exists.

    Throws error if a file does not exist.
    """
    if not pathlib.Path(path).resolve().is_file():
        # pylint: disable=C0209
        raise AssertionError("File does not exist: %s" % str(path))


def assertIsNotFile(path):
    """Asserts a file does not exists.

    Throws error if the file does exist.
    """
    if pathlib.Path(path).resolve().is_file():
        # pylint: disable=C0209
        raise AssertionError("File exist: %s" % str(path))
