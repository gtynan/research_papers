import pytest
from pathlib import Path
from dynamic_bt.run import ProjectContext


@pytest.fixture(scope='session')
def context():
    return ProjectContext(Path.cwd())
