"""Integration proofs for stdio binding capture at production call sites."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.mcp.session_pool import (
    get_session_pool,
    normalized_connection_fingerprint,
    reset_session_pool,
)
from deerflow.mcp.task_tool_caller import McpTaskToolCaller
from deerflow.mcp.tools import get_mcp_tools


class _Args(BaseModel):
    value: int


def _tool(name: str) -> StructuredTool:
    async def _call(value: int) -> int:
        return value

    return StructuredTool(
        name=name,
        description="test",
        args_schema=_Args,
        coroutine=_call,
    )


@pytest.mark.asyncio
async def test_discovery_captures_pool_and_binding_before_first_await():
    reset_session_pool()
    pool = get_session_pool()
    config = ExtensionsConfig.model_validate(
        {
            "mcpServers": {
                "A": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["demo"],
                }
            }
        }
    )
    servers = {
        "A": {
            "transport": "stdio",
            "command": "uvx",
            "args": ["demo"],
        }
    }
    entered = asyncio.Event()
    release = asyncio.Event()

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.callbacks = None
            self.tool_interceptors = []

        async def get_tools(self, *, server_name=None):
            assert server_name == "A"
            entered.set()
            await release.wait()
            return [_tool("A_echo")]

    with (
        patch("deerflow.mcp.tools.build_servers_config", return_value=servers),
        patch("deerflow.mcp.tools.get_initial_oauth_headers", new_callable=AsyncMock, return_value={}),
        patch("deerflow.mcp.tools.build_mcp_tool_interceptors", return_value=[]),
        patch("langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient),
        patch("deerflow.mcp.tools._make_session_pool_tool", side_effect=lambda tool, *_args, **_kwargs: tool) as wrap,
    ):
        task = asyncio.create_task(get_mcp_tools(config))
        await asyncio.wait_for(entered.wait(), timeout=1)

        captured = pool.active_binding("A")
        assert captured is not None
        assert captured.fingerprint == normalized_connection_fingerprint(servers["A"])

        # Supersede A while discovery is blocked. The wrapper must still receive
        # the old capability captured before discovery rather than re-reading
        # the replacement epoch afterwards.
        pool.reconcile_bindings({"A": "replacement"}, ())
        release.set()
        tools = await asyncio.wait_for(task, timeout=1)

    assert [tool.name for tool in tools] == ["A_echo"]
    assert wrap.call_count == 1
    assert wrap.call_args.kwargs["pool"] is pool
    assert wrap.call_args.kwargs["binding"] is captured
    reset_session_pool()


@pytest.mark.asyncio
async def test_task_caller_binds_base_connection_before_workspace_augmentation():
    config = ExtensionsConfig.model_validate(
        {
            "mcpServers": {
                "reports": {
                    "enabled": True,
                    "type": "stdio",
                    "command": "uvx",
                    "args": ["reports-mcp"],
                }
            }
        }
    )
    base_connection = {
        "transport": "stdio",
        "command": "uvx",
        "args": ["reports-mcp"],
        "env": {"TOKEN": "configured"},
    }
    prepared_connection = {
        **base_connection,
        "cwd": "/threads/u/t",
        "env": {
            **base_connection["env"],
            "TMPDIR": "/threads/u/t/.tmp",
            "TMP": "/threads/u/t/.tmp",
            "TEMP": "/threads/u/t/.tmp",
        },
    }

    pool = MagicMock()
    binding = object()
    pool.ensure_binding.return_value = binding
    session = AsyncMock()
    pool.get_session = AsyncMock(return_value=session)

    oauth = SimpleNamespace(
        has_oauth_servers=lambda: False,
        get_authorization_header=AsyncMock(return_value=None),
    )
    caller = McpTaskToolCaller(config, oauth_token_manager=oauth)
    caller._invoke = AsyncMock(return_value={"ok": True})

    with (
        patch("deerflow.mcp.task_tool_caller.get_session_pool", return_value=pool),
        patch("deerflow.mcp.task_tool_caller.build_server_params", return_value=base_connection),
        patch("deerflow.mcp.task_tool_caller._prepare_stdio_connection", return_value=prepared_connection),
    ):
        result = await caller._call_configured_tool(
            server_name="reports",
            tool_name="status",
            arguments={},
            user_id="u",
            thread_id="t",
            thread_incarnation="i",
            request_scoped_headers=False,
            connection_scope="deployment",
        )

    assert result == {"ok": True}
    pool.ensure_binding.assert_called_once_with(
        "reports",
        normalized_connection_fingerprint(base_connection),
        domain="deployment",
    )
    pool.get_session.assert_awaited_once()
    assert pool.get_session.await_args.args[2] is prepared_connection
    assert pool.get_session.await_args.kwargs["binding"] is binding
