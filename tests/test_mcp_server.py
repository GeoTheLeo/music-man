"""
The MCP server, exercised through a real MCP client connected in-process:
the guardrails must hold for any client, not just our own agents.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import Client

from evals.fixtures import FRAUD_DEVICE, FRAUD_DEVICE_PERIOD, write_snapshot
from music_man import mcp_server
from music_man.agent import queue


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("MUSIC_MAN_QUEUE_DB", str(tmp_path / "queue.db"))
    monkeypatch.setenv("MUSIC_MAN_SNAPSHOT", str(write_snapshot(tmp_path / "snapshot.parquet")))


def _call(*calls: tuple[str, dict]):
    async def run():
        async with Client(mcp_server.server) as client:
            return [await client.call_tool(name, args) for name, args in calls]

    return asyncio.run(run())


CHECK = ("check_hold_policy", {"artist_id": FRAUD_DEVICE, "period": FRAUD_DEVICE_PERIOD})
PROPOSE = ("propose_hold", {"artist_id": FRAUD_DEVICE, "period": FRAUD_DEVICE_PERIOD, "rationale": "device concentration 0.92"})


def test_lists_four_tools_with_honest_annotations():
    async def run():
        async with Client(mcp_server.server) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(run())}
    assert set(tools) == {"list_flagged_artists", "get_artist_detail", "check_hold_policy", "propose_hold"}
    assert tools["get_artist_detail"].annotations.read_only_hint is True
    propose = tools["propose_hold"].annotations
    assert propose.read_only_hint is False and propose.destructive_hint is False and propose.idempotent_hint is True


def test_guardrail_refusal_reaches_the_client_with_its_reason():
    mcp_server.start_session(dry_run=False)
    [result] = _call(PROPOSE)
    assert result.is_error
    assert "check_hold_policy" in result.content[0].text
    assert queue.list_holds() == []


def test_check_then_propose_queues_once_even_when_retried():
    mcp_server.start_session(dry_run=False)
    _, first, retry = _call(CHECK, PROPOSE, PROPOSE)
    first, retry = json.loads(first.content[0].text), json.loads(retry.content[0].text)
    assert first["queued"] and retry["deduplicated"] and retry["hold_id"] == first["hold_id"]
    [hold] = queue.list_holds()
    assert hold.status == queue.PENDING and hold.run_id.startswith("mcp_")


def test_dry_run_session_writes_nothing(monkeypatch):
    monkeypatch.setenv("MUSIC_MAN_DRY_RUN", "1")
    mcp_server.start_session()
    _, result = _call(CHECK, PROPOSE)
    assert json.loads(result.content[0].text)["dry_run"] is True
    assert queue.list_holds() == []


def test_no_tool_can_approve_a_hold():
    async def run():
        async with Client(mcp_server.server) as client:
            return [t.name for t in (await client.list_tools()).tools]

    assert not any(word in name for name in asyncio.run(run()) for word in ("approve", "execute", "decide"))
