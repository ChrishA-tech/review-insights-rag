import pytest
from fastapi.testclient import TestClient

pytest.importorskip("rank_bm25")
from rvi.api import app  # noqa: E402

c = TestClient(app)


def test_health_and_search():
    assert c.get("/health").json()["reviews"] > 20000
    r = c.get("/search", params={"q": "runs small", "department": "Dresses", "k": 3}).json()
    assert len(r["reviews"]) == 3 and all(x["department"] == "Dresses" for x in r["reviews"])


def test_aspects_and_404():
    assert "aspect_rates" in c.get("/aspects", params={"department": "Tops"}).json()
    assert c.get("/aspects", params={"department": "Shoes"}).status_code == 404
