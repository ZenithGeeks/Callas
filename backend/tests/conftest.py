import os

os.environ["CALLAS_FAKE"] = "1"
os.environ["CALLAS_FAKE_DELAY"] = "0"
os.environ["CALLAS_LIVE_SPEED"] = "200"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import web  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _private_upload_dir(tmp_path, monkeypatch):
    """Tests get their own upload folder, so a test run never empties a running demo's."""
    monkeypatch.setattr(web, "UPLOAD_DIR", tmp_path / "uploads")


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
