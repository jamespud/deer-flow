"""Focused tests for stdio server binding fencing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deerflow.mcp.session_pool import (
    MCPSessionPool,
    StaleMCPBindingError,
    get_session_pool,
    normalized_connection_fingerprint,
    reset_session_pool,
)

_CONNECTION = {"transport": "stdio", "command": "x", "args": []}


class _GatedInitCm:
    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.initialize_started = asyncio.Event()
        self.closed = False
        self.enter_task = None
        self.exit_task = None
        self.session = MagicMock()
        self.session.initialize = self._initialize

    async def __aenter__(self):
        self.enter_task = asyncio.current_task()
        return self.session

    async def _initialize(self):
        self.initialize_started.set()
        if self.gate is not None:
            await self.gate.wait()

    async def __aexit__(self, *_args):
        self.exit_task = asyncio.current_task()
        self.closed = True
        return False


def test_fingerprint_is_secret_safe_and_tracks_process_identity():
    base = {
        "transport": "stdio",
        "command": "uvx",
        "args": ["demo"],
        "cwd": "/srv/a",
        "env": {"TOKEN": "resolved-secret"},
    }
    fingerprint = normalized_connection_fingerprint(base)
    assert "resolved-secret" not in fingerprint
    assert fingerprint == normalized_connection_fingerprint(dict(base))
    assert fingerprint != normalized_connection_fingerprint({**base, "cwd": "/srv/b"})
    assert fingerprint != normalized_connection_fingerprint({**base, "env": {"TOKEN": "other"}})


def test_same_fingerprint_is_idempotent_and_change_mints_new_binding():
    pool = MCPSessionPool()
    first = pool.ensure_binding("A", "fp-1")
    assert pool.ensure_binding("A", "fp-1") is first

    pool.reconcile_bindings({"A": "fp-2"}, ())
    second = pool.active_binding("A")
    assert second is not None
    assert second is not first
    assert second.epoch > first.epoch


@pytest.mark.asyncio
async def test_reconcile_detaches_only_changed_server_and_preserves_personal_domain():
    pool = MCPSessionPool()
    a = pool.ensure_binding("A", "a1")
    b = pool.ensure_binding("B", "b1")
    personal_a = pool.ensure_binding("A", "personal-a", domain="personal")
    cm_a, cm_b = _GatedInitCm(), _GatedInitCm()

    with patch("langchain_mcp_adapters.sessions.create_session", side_effect=[cm_a, cm_b]):
        session_a = await pool.get_session("A", "u:t", _CONNECTION, binding=a)
        session_b = await pool.get_session("B", "u:t", _CONNECTION, binding=b)

    prepared = pool.reconcile_bindings({"A": "a2", "B": "b1"}, (), domain="deployment")
    assert [session for session, *_ in prepared.entries] == [session_a]
    assert session_b not in [session for session, *_ in prepared.entries]
    assert pool.active_binding("B") is b
    assert pool.active_binding("A", domain="personal") is personal_a

    await pool.close_prepared_owners(prepared)
    assert cm_a.closed is True
    assert cm_b.closed is False
    await pool.close_all()


@pytest.mark.asyncio
async def test_stale_binding_fails_before_session_creation():
    pool = MCPSessionPool()
    old = pool.ensure_binding("A", "a1")
    pool.reconcile_bindings({"A": "a2"}, ())

    with patch("langchain_mcp_adapters.sessions.create_session") as create_session:
        with pytest.raises(StaleMCPBindingError):
            await pool.get_session("A", "u:t", _CONNECTION, binding=old)

    create_session.assert_not_called()
    assert not pool._entries
    assert not pool._inflight


@pytest.mark.asyncio
async def test_reconcile_while_initializing_fences_creator_commit():
    pool = MCPSessionPool()
    old = pool.ensure_binding("A", "a1")
    gate = asyncio.Event()
    cm = _GatedInitCm(gate)

    with patch("langchain_mcp_adapters.sessions.create_session", return_value=cm):
        creator = asyncio.create_task(pool.get_session("A", "u:t", _CONNECTION, binding=old))
        await asyncio.wait_for(cm.initialize_started.wait(), timeout=1)
        prepared = pool.reconcile_bindings({"A": "a2"}, ())
        gate.set()
        with pytest.raises(StaleMCPBindingError):
            await asyncio.wait_for(creator, timeout=1)

    await pool.close_prepared_owners(prepared)
    assert not pool._entries
    assert not pool._inflight
    assert cm.closed is True
    assert cm.exit_task is cm.enter_task


@pytest.mark.asyncio
async def test_reconcile_fences_creator_and_joiner_waiting_on_same_creation():
    pool = MCPSessionPool()
    old = pool.ensure_binding("A", "a1")
    gate = asyncio.Event()
    cm = _GatedInitCm(gate)

    with patch("langchain_mcp_adapters.sessions.create_session", return_value=cm):
        creator = asyncio.create_task(pool.get_session("A", "u:t", _CONNECTION, binding=old))
        await asyncio.wait_for(cm.initialize_started.wait(), timeout=1)

        joiner = asyncio.create_task(pool.get_session("A", "u:t", _CONNECTION, binding=old))
        # One scheduler handoff is sufficient: the inflight record already
        # exists, so the joiner deterministically parks on its ready Future.
        await asyncio.sleep(0)
        assert len(pool._inflight) == 1

        prepared = pool.reconcile_bindings({"A": "a2"}, ())
        gate.set()

        with pytest.raises(StaleMCPBindingError):
            await asyncio.wait_for(creator, timeout=1)
        with pytest.raises(StaleMCPBindingError):
            await asyncio.wait_for(joiner, timeout=1)

    await pool.close_prepared_owners(prepared)
    assert not pool._entries
    assert not pool._inflight


@pytest.mark.asyncio
async def test_remove_readd_same_fingerprint_does_not_reauthorize_old_binding():
    pool = MCPSessionPool()
    old = pool.ensure_binding("A", "same")
    pool.reconcile_bindings({}, {"A"})
    tombstone = pool.active_binding("A")
    assert tombstone is not None and tombstone.fingerprint is None

    pool.reconcile_bindings({"A": "same"}, ())
    current = pool.active_binding("A")
    assert current is not None
    assert current is not old
    assert current.epoch > tombstone.epoch > old.epoch

    with patch("langchain_mcp_adapters.sessions.create_session") as create_session:
        with pytest.raises(StaleMCPBindingError):
            await pool.get_session("A", "u:t", _CONNECTION, binding=old)
    create_session.assert_not_called()


@pytest.mark.asyncio
async def test_bindingless_access_fails_after_explicit_binding_lifecycle():
    pool = MCPSessionPool()
    pool.ensure_binding("A", normalized_connection_fingerprint(_CONNECTION))

    with pytest.raises(StaleMCPBindingError):
        await pool.get_session("A", "u:t", _CONNECTION)


@pytest.mark.asyncio
async def test_reset_fences_old_pool_even_when_new_pool_reuses_same_epoch():
    reset_session_pool()
    old_pool = get_session_pool()
    old_binding = old_pool.ensure_binding("A", "same")

    assert reset_session_pool() is old_pool
    new_pool = get_session_pool()
    new_binding = new_pool.ensure_binding("A", "same")
    assert new_binding.epoch == old_binding.epoch
    assert new_binding is not old_binding

    with patch("langchain_mcp_adapters.sessions.create_session") as create_session:
        with pytest.raises(StaleMCPBindingError):
            await old_pool.get_session("A", "u:t", _CONNECTION, binding=old_binding)
    create_session.assert_not_called()

    reset_session_pool()
