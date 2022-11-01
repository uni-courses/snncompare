"""Configuration for pytest to automatically collect types.

Thanks to Guilherme Salgado.
"""

import pytest
from pyannotate_runtime import collect_types


# pylint: disable=W0613
def pytest_collection_finish(session):
    """Handle the pytest collection finish hook: configure pyannotate.

    Explicitly delay importing `collect_types` until all tests have been
    collected.  This gives gevent a chance to monkey patch the world
    before importing pyannotate.
    """

    collect_types.init_types_collection()


# pylint: disable=W0613
@pytest.fixture(autouse=True)
def collect_types_fixture():
    """Performs unknown activity."""

    collect_types.start()
    yield
    collect_types.stop()


# pylint: disable=W0613
def pytest_sessionfinish(session, exitstatus):
    """Performs unknown activity."""
    collect_types.dump_stats("type_info.json")
