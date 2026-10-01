"""
Anthropic tool-runner wrappers for the fraud-investigation operations.

The operations and every guardrail live in agent/operations.py, shared with
the MCP server and the LangGraph agent; this module only exposes them in
the shape client.beta.messages.tool_runner expects.
"""

from __future__ import annotations

from anthropic import beta_tool

from music_man.agent import operations
from music_man.agent.operations import current_run, snapshot_path, start_run  # noqa: F401 - re-exported

list_flagged_artists = beta_tool(operations.list_flagged_artists)
get_artist_detail = beta_tool(operations.get_artist_detail)
check_hold_policy = beta_tool(operations.check_hold_policy)
propose_hold = beta_tool(operations.propose_hold)

TOOLS = [list_flagged_artists, get_artist_detail, check_hold_policy, propose_hold]
