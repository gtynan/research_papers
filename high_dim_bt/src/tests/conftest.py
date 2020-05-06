import pytest

from kedro.context import load_context
from pathlib import Path


# Kedro context to access params
@pytest.fixture(scope='session')
def context():
    return load_context(Path.cwd())
