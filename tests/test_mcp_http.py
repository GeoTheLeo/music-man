"""
The MCP server over HTTP, as deployed: API-key gate, open health probe, and
the refusal to serve publicly without a key.
"""

from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient

from music_man import mcp_server

KEY = "test-key-123"
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def client():
    app = mcp_server.http_app(host="0.0.0.0", api_key=KEY, allowed_hosts=["testserver"])
    with TestClient(app) as c:
        yield c


def test_health_probe_needs_no_key(client):
    assert client.get("/healthz").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_mcp_without_the_right_key_is_rejected(client, headers):
    response = client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, **headers})
    assert response.status_code == 401


def test_mcp_with_the_key_completes_the_handshake(client):
    response = client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, "X-API-Key": KEY})
    assert response.status_code == 200
    assert "music-man" in response.text


def test_refuses_to_serve_publicly_without_a_key(monkeypatch):
    monkeypatch.delenv("MUSIC_MAN_MCP_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["mcp_server", "--http", "8765", "--host", "0.0.0.0"])
    with pytest.raises(SystemExit, match="MUSIC_MAN_MCP_API_KEY"):
        mcp_server.main()
