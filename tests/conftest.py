import pytest
from pathlib import Path
import shutil
import tempfile

@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)
