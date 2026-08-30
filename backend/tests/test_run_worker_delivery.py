"""Worker-level regression tests for the terminal run.delivery event (#4272 slice 1)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deerflow.config.app_config import AppConfig
from deerflow.config.paths import Paths
from deerflow.config.run_ownership_config import RunOwnershipConfig
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.tool_output_config import ToolOutputConfig
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.runs.manager import RunManager
from deerflow.runtime.runs.schemas import RunStatus
from deerflow.runtime.runs.store.memory import MemoryRunStore
from deerflow.runtime.runs.worker import RunContext, _delivery_content_with_outputs, run_agent
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.workspace_changes.types import WorkspaceSnapshot


def _make_bridge():
    return SimpleNamespace(publish=AsyncMock(), publish_end=AsyncMock(), cleanup=AsyncMock())


async def _delivery_events(store: MemoryRunEventStore, thread_id: str, run_id: str) -> list[dict]:
    events = await store.list_events(thread_id, run_id)
    return [e for e in events if e["event_type"] == "run.delivery"]


def test_delivery_verification_treats_presented_directory_as_covering_produced_files():
    content = {
        "presented": 1,
        "paths": ["/mnt/user-data/outputs/site"],
        "by_tool": {"present_files": ["/mnt/user-data/outputs/site"]},
    }

    delivery = _delivery_content_with_outputs(
        content,
        [
            "/mnt/user-data/outputs/site/index.html",
            "/mnt/user-data/outputs/site/assets/style.css",
        ],
    )

    assert delivery["matched_paths"] == [
        "/mnt/user-data/outputs/site/index.html",
        "/mnt/user-data/outputs/site/assets/style.css",
    ]
    assert delivery["satisfied"] is True


@pytest.mark.anyio
async def test_delivery_event_records_present_files_paths_on_success():
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            ai = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}])
            journal._remember_current_run_tool_calls(ai, caller="lead_agent")
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )
    await asyncio.sleep(0)

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert len(delivery) == 1
    assert delivery[0]["content"]["presented"] == 1
    assert delivery[0]["content"]["paths"] == ["/mnt/user-data/outputs/report.md"]
    assert delivery[0]["content"]["by_tool"] == {"present_files": ["/mnt/user-data/outputs/report.md"]}
    fetched = await run_manager.get(record.run_id)
    assert fetched.status == RunStatus.success


@pytest.mark.anyio
async def test_delivery_event_presented_zero_without_artifact_production():
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )
    await asyncio.sleep(0)

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert len(delivery) == 1
    assert delivery[0]["content"] == {"presented": 0, "paths": [], "by_tool": {}}
    fetched = await run_manager.get(record.run_id)
    assert fetched.status == RunStatus.success


@pytest.mark.anyio
async def test_ordered_finalization_does_not_overtake_hung_journal_write(monkeypatch):
    import deerflow.runtime.journal as journal_module
    import deerflow.runtime.runs.worker as worker_module
    from deerflow.runtime.cancellation import wait_for_task_until
    from deerflow.runtime.journal import RunJournal

    monkeypatch.setattr(journal_module, "_CANCELLATION_DRAIN_TIMEOUT_SECONDS", 0.01, raising=False)

    class LockedEventStore(MemoryRunEventStore):
        def __init__(self):
            super().__init__()
            self.lock = asyncio.Lock()
            self.journal_started = asyncio.Event()
            self.release_journal = asyncio.Event()
            self.receipt_attempted = asyncio.Event()

        async def put_batch(self, events):
            async with self.lock:
                self.journal_started.set()
                await self.release_journal.wait()
                return await super().put_batch(events)

        async def put_if_absent(self, **kwargs):
            self.receipt_attempted.set()
            async with self.lock:
                return await super().put_if_absent(**kwargs)

    store = LockedEventStore()
    journal = RunJournal("run-1", "thread-1", store, flush_threshold=1)
    journal._put(event_type="before.receipt", category="trace", content="first")
    pipeline = None
    try:
        await asyncio.wait_for(store.journal_started.wait(), timeout=0.2)
        pipeline = asyncio.create_task(
            worker_module._persist_journal_and_delivery_receipt(
                journal,
                store,
                thread_id="thread-1",
                run_id="run-1",
                content={"presented": 0, "paths": [], "by_tool": {}},
            )
        )
        run_manager = RunManager()
        run_manager.track_background_finalization(
            pipeline,
            action="persist journal and delivery receipt",
            run_id="run-1",
        )
        completed = await wait_for_task_until(
            pipeline,
            deadline=asyncio.get_running_loop().time() + 0.01,
        )

        assert completed is False
        assert store.receipt_attempted.is_set() is False
        assert len(run_manager._background_finalization_tasks) == 1

        store.release_journal.set()
        assert await asyncio.wait_for(asyncio.shield(pipeline), timeout=0.2) is True
        await asyncio.sleep(0)

        events = await store.list_events("thread-1", "run-1")
        assert [event["event_type"] for event in events] == ["before.receipt", "run.delivery"]
        assert not run_manager._background_finalization_tasks
    finally:
        store.release_journal.set()
        if pipeline is not None:
            await asyncio.gather(pipeline, return_exceptions=True)
        pending_writes = tuple(journal._pending_flush_tasks) + tuple(journal._detached_write_tasks)
        if pending_writes:
            await asyncio.gather(*pending_writes, return_exceptions=True)


@pytest.mark.anyio
async def test_changed_outputs_succeed_when_a_produced_output_is_presented(monkeypatch):
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    monkeypatch.setattr(
        "deerflow.runtime.runs.worker._produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            ai = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}])
            journal._remember_current_run_tool_calls(ai, caller="lead_agent")
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert delivery[0]["content"] == {
        "presented": 1,
        "paths": ["/mnt/user-data/outputs/report.md"],
        "by_tool": {"present_files": ["/mnt/user-data/outputs/report.md"]},
        "verification": {
            "source": "outputs_changed",
            "requirement": "present_files_matches_produced_output",
        },
        "produced_paths": ["/mnt/user-data/outputs/report.md"],
        "presented_paths": ["/mnt/user-data/outputs/report.md"],
        "matched_paths": ["/mnt/user-data/outputs/report.md"],
        "stage": "presented",
        "satisfied": True,
    }
    assert record.status == RunStatus.success


@pytest.mark.anyio
async def test_changed_outputs_fail_closed_when_not_presented(monkeypatch):
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    monkeypatch.setattr(
        "deerflow.runtime.runs.worker._produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    class ProseOnlyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            yield {"messages": [AIMessage(content="SESSION SUMMARY")]}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: ProseOnlyAgent(),
        graph_input={},
        config={},
    )

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert delivery[0]["content"] == {
        "presented": 0,
        "paths": [],
        "by_tool": {},
        "verification": {
            "source": "outputs_changed",
            "requirement": "present_files_matches_produced_output",
        },
        "produced_paths": ["/mnt/user-data/outputs/report.md"],
        "presented_paths": [],
        "matched_paths": [],
        "stage": "not_started",
        "satisfied": False,
    }
    assert record.status == RunStatus.error
    assert record.error == "Artifact delivery incomplete: no produced output artifact was presented"
    assert record.stop_reason is None


@pytest.mark.anyio
async def test_externalized_tool_results_do_not_trigger_delivery_verification(tmp_path, monkeypatch):
    """Oversized tool outputs externalized under outputs/.tool-results/ are
    process feedback for the model, not deliverables: a run that only produced
    those files must succeed without any present_files call."""
    paths = Paths(base_dir=tmp_path)
    monkeypatch.setattr("deerflow.workspace_changes.recorder.get_paths", lambda: paths)
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()

    class ExternalizingAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            # Simulates ToolOutputBudgetMiddleware persisting an oversized tool
            # output mid-run (default storage_subdir is ".tool-results").
            tool_results = paths.sandbox_outputs_dir("thread-1", user_id=get_effective_user_id()) / ".tool-results"
            tool_results.mkdir(parents=True, exist_ok=True)
            (tool_results / "bash-abcdef123456.log").write_text("x" * 20000, encoding="utf-8")
            yield {"messages": [AIMessage(content="Here is the answer.")]}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: ExternalizingAgent(),
        graph_input={},
        config={},
    )

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert len(delivery) == 1
    assert delivery[0]["content"] == {"presented": 0, "paths": [], "by_tool": {}}
    fetched = await run_manager.get(record.run_id)
    assert fetched.status == RunStatus.success


@pytest.mark.anyio
async def test_custom_tool_output_storage_subdir_does_not_trigger_delivery_verification(tmp_path, monkeypatch):
    """A custom tool_output.storage_subdir is honoured by the exclusion, not
    only the default .tool-results name."""
    paths = Paths(base_dir=tmp_path)
    monkeypatch.setattr("deerflow.workspace_changes.recorder.get_paths", lambda: paths)
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    app_config = AppConfig(sandbox=SandboxConfig(use="test"), tool_output=ToolOutputConfig(storage_subdir="tool-output-cache"))

    class ExternalizingAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            cache = paths.sandbox_outputs_dir("thread-1", user_id=get_effective_user_id()) / "tool-output-cache"
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "web_fetch-abcdef123456.log").write_text("y" * 20000, encoding="utf-8")
            yield {"messages": [AIMessage(content="Here is the answer.")]}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store, app_config=app_config),
        agent_factory=lambda *, config: ExternalizingAgent(),
        graph_input={},
        config={},
    )

    fetched = await run_manager.get(record.run_id)
    assert fetched.status == RunStatus.success


@pytest.mark.anyio
async def test_changed_outputs_succeed_when_one_of_multiple_outputs_is_presented(monkeypatch):
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    monkeypatch.setattr(
        "deerflow.runtime.runs.worker._produced_output_paths",
        AsyncMock(
            return_value=[
                "/mnt/user-data/outputs/report.md",
                "/mnt/user-data/outputs/appendix.md",
            ]
        ),
    )

    class PartiallyPresentingAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: PartiallyPresentingAgent(),
        graph_input={},
        config={},
    )

    delivery = (await _delivery_events(store, "thread-1", record.run_id))[0]["content"]
    assert delivery["stage"] == "presented"
    assert delivery["presented_paths"] == ["/mnt/user-data/outputs/report.md"]
    assert delivery["matched_paths"] == ["/mnt/user-data/outputs/report.md"]
    assert delivery["satisfied"] is True
    assert record.status == RunStatus.success


@pytest.mark.anyio
async def test_changed_outputs_fail_when_present_files_only_presents_an_unrelated_file(monkeypatch):
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    monkeypatch.setattr(
        "deerflow.runtime.runs.worker._produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    class UnrelatedPresentingAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/old-report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: UnrelatedPresentingAgent(),
        graph_input={},
        config={},
    )

    delivery = (await _delivery_events(store, "thread-1", record.run_id))[0]["content"]
    assert delivery["stage"] == "mismatched"
    assert delivery["presented_paths"] == ["/mnt/user-data/outputs/old-report.md"]
    assert delivery["matched_paths"] == []
    assert delivery["satisfied"] is False
    assert record.status == RunStatus.error


@pytest.mark.anyio
async def test_fenced_worker_leaves_delivery_receipt_to_peer_recovery():
    """A stale worker must not finalize the singleton delivery receipt."""
    run_manager = RunManager()
    record = await run_manager.create("thread-lease-lost")
    record.ownership_lost = True
    record.abort_event.set()
    record.status = RunStatus.error
    event_store = MemoryRunEventStore()
    thread_store = SimpleNamespace(
        update_display_name=AsyncMock(),
        update_status=AsyncMock(),
    )
    on_run_completed = AsyncMock()
    agent_factory = MagicMock(side_effect=AssertionError("fenced worker started the agent"))

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(
            checkpointer=None,
            event_store=event_store,
            thread_store=thread_store,
            on_run_completed=on_run_completed,
        ),
        agent_factory=agent_factory,
        graph_input={},
        config={},
    )

    assert await _delivery_events(event_store, record.thread_id, record.run_id) == []
    agent_factory.assert_not_called()
    thread_store.update_display_name.assert_not_awaited()
    thread_store.update_status.assert_not_awaited()
    on_run_completed.assert_not_awaited()


@pytest.mark.anyio
async def test_delivery_event_is_singleton_across_goal_continuations(monkeypatch):
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    stream_calls = 0
    continuation_calls = 0

    class ContinuingAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            nonlocal stream_calls
            stream_calls += 1
            journal = config["context"]["__run_journal"]
            tool_call_id = f"call_{stream_calls}"
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": tool_call_id, "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            artifacts = ["/mnt/user-data/outputs/report.md"]
            if stream_calls == 2:
                artifacts.append("/mnt/user-data/outputs/appendix.md")
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": artifacts,
                        "messages": [ToolMessage("Successfully presented files", tool_call_id=tool_call_id)],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    async def prepare_continuation(**kwargs):
        nonlocal continuation_calls
        continuation_calls += 1
        if continuation_calls == 1:
            return {"messages": []}
        return None

    monkeypatch.setattr("deerflow.runtime.runs.worker._prepare_goal_continuation_input", prepare_continuation)

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: ContinuingAgent(),
        graph_input={},
        config={},
    )

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert stream_calls == 2
    assert len(delivery) == 1
    assert delivery[0]["content"] == {
        "presented": 2,
        "paths": [
            "/mnt/user-data/outputs/report.md",
            "/mnt/user-data/outputs/appendix.md",
        ],
        "by_tool": {
            "present_files": [
                "/mnt/user-data/outputs/report.md",
                "/mnt/user-data/outputs/appendix.md",
            ]
        },
    }


@pytest.mark.anyio
async def test_delivery_event_emitted_exactly_once_on_error_path():
    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()

    class FailingAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            raise RuntimeError("boom")
            yield  # pragma: no cover - make this an async generator

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: FailingAgent(),
        graph_input={},
        config={},
    )
    await asyncio.sleep(0)

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert len(delivery) == 1
    assert delivery[0]["content"]["presented"] == 0
    fetched = await run_manager.get(record.run_id)
    assert fetched.status == RunStatus.error


@pytest.mark.anyio
async def test_delivery_is_durable_before_terminal_run_status():
    events = MemoryRunEventStore()

    class OrderingRunStore(MemoryRunStore):
        async def update_status(self, run_id, status, *, error=None, stop_reason=None):
            if status not in {"pending", "running"}:
                receipt = await events.list_events("thread-1", run_id, event_types=["run.delivery"])
                assert len(receipt) == 1
            return await super().update_status(run_id, status, error=error, stop_reason=stop_reason)

    run_store = OrderingRunStore()
    run_manager = RunManager(store=run_store)
    record = await run_manager.create("thread-1")

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=events),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    assert (await run_store.get(record.run_id))["status"] == "success"


@pytest.mark.anyio
async def test_delivery_write_retries_before_persisting_success():
    class FlakyReceiptStore(MemoryRunEventStore):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        async def put_if_absent(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("transient event store outage")
            return await super().put_if_absent(**kwargs)

    event_store = FlakyReceiptStore()
    run_store = MemoryRunStore()
    run_manager = RunManager(store=run_store)
    record = await run_manager.create("thread-1")

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=event_store),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    assert event_store.attempts == 2
    assert len(await _delivery_events(event_store, "thread-1", record.run_id)) == 1
    assert (await run_store.get(record.run_id))["status"] == "success"


@pytest.mark.anyio
async def test_delivery_write_failure_preserves_real_durable_terminal_status():
    class FailingReceiptStore(MemoryRunEventStore):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        async def put_if_absent(self, **kwargs):
            self.attempts += 1
            raise RuntimeError("event store unavailable")

    run_store = MemoryRunStore()
    run_manager = RunManager(store=run_store)
    record = await run_manager.create("thread-1")
    event_store = FailingReceiptStore()

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=event_store),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    # A receipt outage must not let lease recovery rewrite a genuine success
    # as an error. After bounded retries, preserve the worker's real outcome.
    assert event_store.attempts > 1
    assert record.status == RunStatus.success
    assert (await run_store.get(record.run_id))["status"] == "success"


@pytest.mark.anyio
async def test_produced_artifact_delivery_fails_closed_when_receipt_cannot_be_persisted(monkeypatch):
    class FailingReceiptStore(MemoryRunEventStore):
        async def put_if_absent(self, **kwargs):
            raise RuntimeError("event store unavailable")

    run_store = MemoryRunStore()
    run_manager = RunManager(store=run_store)
    record = await run_manager.create("thread-1")
    monkeypatch.setattr(
        "deerflow.runtime.runs.worker._produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    class PresentingAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=FailingReceiptStore()),
        agent_factory=lambda *, config: PresentingAgent(),
        graph_input={},
        config={},
    )

    assert record.status == RunStatus.error
    assert record.error == "Artifact delivery verification failed: terminal delivery receipt could not be persisted"
    assert (await run_store.get(record.run_id))["status"] == "error"


@pytest.mark.anyio
async def test_finalization_timeout_does_not_downgrade_success_before_late_receipt(monkeypatch):
    import deerflow.runtime.runs.worker as worker_module

    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    finalization_finished = asyncio.Event()

    monkeypatch.setattr(worker_module, "_FINALIZATION_DRAIN_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "capture_workspace_snapshot", AsyncMock(return_value=WorkspaceSnapshot()))
    monkeypatch.setattr(worker_module, "record_workspace_changes", AsyncMock(return_value=None))
    monkeypatch.setattr(
        worker_module,
        "_produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    async def late_finalization(*args, **kwargs):
        await asyncio.sleep(0.05)
        finalization_finished.set()
        return True

    monkeypatch.setattr(worker_module, "_persist_journal_and_delivery_receipt", late_finalization)

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    assert record.status == RunStatus.success
    assert record.error is None
    await asyncio.wait_for(finalization_finished.wait(), timeout=0.2)


@pytest.mark.anyio
async def test_late_finalization_failure_corrects_success_after_timeout(monkeypatch):
    import deerflow.runtime.runs.worker as worker_module

    run_store = MemoryRunStore()
    run_manager = RunManager(store=run_store)
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    finalization_finished = asyncio.Event()

    monkeypatch.setattr(worker_module, "_FINALIZATION_DRAIN_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "capture_workspace_snapshot", AsyncMock(return_value=WorkspaceSnapshot()))
    monkeypatch.setattr(worker_module, "record_workspace_changes", AsyncMock(return_value=None))
    monkeypatch.setattr(
        worker_module,
        "_produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    async def late_finalization(*args, **kwargs):
        await asyncio.sleep(0.05)
        finalization_finished.set()
        return False

    monkeypatch.setattr(worker_module, "_persist_journal_and_delivery_receipt", late_finalization)

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=lambda *, config: DummyAgent(),
        graph_input={},
        config={},
    )

    assert record.status == RunStatus.success
    await asyncio.wait_for(finalization_finished.wait(), timeout=0.2)
    async with asyncio.timeout(0.2):
        while record.status != RunStatus.error:
            await asyncio.sleep(0)

    assert record.error == "Artifact delivery verification failed: terminal delivery receipt could not be persisted"
    persisted = await run_store.get(record.run_id)
    assert persisted is not None
    assert persisted["status"] == RunStatus.error


@pytest.mark.anyio
@pytest.mark.parametrize("heartbeat_enabled", [False, True])
async def test_late_finalization_failure_before_terminal_write_is_not_lost(monkeypatch, heartbeat_enabled):
    import deerflow.runtime.runs.worker as worker_module

    class BlockingDeliveryStore(MemoryRunStore):
        def __init__(self):
            super().__init__()
            self.status_started = asyncio.Event()
            self.release_status = asyncio.Event()
            self.completion_started = asyncio.Event()
            self.release_completion = asyncio.Event()
            self.mark_started = asyncio.Event()
            self.release_mark = asyncio.Event()

        async def update_status(self, run_id, status, *, error=None, stop_reason=None):
            self.status_started.set()
            await self.release_status.wait()
            return await super().update_status(run_id, status, error=error, stop_reason=stop_reason)

        async def finalize_if_not_cancelled(self, run_id, *, status, error=None, stop_reason=None):
            self.status_started.set()
            await self.release_status.wait()
            return await super().finalize_if_not_cancelled(
                run_id,
                status=status,
                error=error,
                stop_reason=stop_reason,
            )

        async def mark_delivery_receipt_failed(self, run_id: str, *, error: str) -> bool:
            self.mark_started.set()
            await self.release_mark.wait()
            return await super().mark_delivery_receipt_failed(run_id, error=error)

        async def update_run_completion(self, run_id, *, status, **kwargs):
            self.completion_started.set()
            await self.release_completion.wait()
            return await super().update_run_completion(run_id, status=status, **kwargs)

    run_store = BlockingDeliveryStore()
    run_manager = RunManager(
        store=run_store,
        run_ownership_config=RunOwnershipConfig(heartbeat_enabled=heartbeat_enabled),
    )
    record = await run_manager.create("thread-1")
    event_store = MemoryRunEventStore()
    finalization_finished = asyncio.Event()

    monkeypatch.setattr(worker_module, "_FINALIZATION_DRAIN_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "capture_workspace_snapshot", AsyncMock(return_value=WorkspaceSnapshot()))
    monkeypatch.setattr(worker_module, "record_workspace_changes", AsyncMock(return_value=None))
    monkeypatch.setattr(
        worker_module,
        "_produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    async def late_finalization(*args, **kwargs):
        await asyncio.sleep(0.05)
        finalization_finished.set()
        return False

    monkeypatch.setattr(worker_module, "_persist_journal_and_delivery_receipt", late_finalization)

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    task = asyncio.create_task(
        run_agent(
            _make_bridge(),
            run_manager,
            record,
            ctx=RunContext(checkpointer=None, event_store=event_store),
            agent_factory=lambda *, config: DummyAgent(),
            graph_input={},
            config={},
        )
    )
    record.task = task
    try:
        await asyncio.sleep(0.05)
        await asyncio.wait_for(run_store.status_started.wait(), timeout=0.3)
        await asyncio.wait_for(finalization_finished.wait(), timeout=0.3)
        await asyncio.sleep(0)
        assert run_store.mark_started.is_set() is False

        run_store.release_status.set()
        await asyncio.wait_for(run_store.completion_started.wait(), timeout=0.3)
        await asyncio.sleep(0)
        assert run_store.mark_started.is_set() is False

        run_store.release_completion.set()
        await asyncio.wait_for(run_store.mark_started.wait(), timeout=0.3)
        persisted = await run_store.get(record.run_id)
        assert persisted is not None
        assert persisted["status"] == RunStatus.success.value

        run_store.release_mark.set()
        await asyncio.wait_for(task, timeout=0.3)
    finally:
        run_store.release_status.set()
        run_store.release_completion.set()
        run_store.release_mark.set()
        await asyncio.gather(task, return_exceptions=True)

    persisted = await run_store.get(record.run_id)
    assert persisted is not None
    async with asyncio.timeout(0.3):
        while persisted["status"] != RunStatus.error.value:
            await asyncio.sleep(0)
            persisted = await run_store.get(record.run_id)
            assert persisted is not None
    assert persisted["status"] == RunStatus.error.value
    assert persisted["error"] == "Artifact delivery verification failed: terminal delivery receipt could not be persisted"


@pytest.mark.anyio
async def test_cancelled_finalization_progress_releases_late_reconciliation(monkeypatch):
    """A cancellation during finalizing progress must not strand late reconciliation."""
    import deerflow.runtime.runs.worker as worker_module

    run_manager = RunManager()
    record = await run_manager.create("thread-1")
    event_store = MemoryRunEventStore()
    progress_started = asyncio.Event()
    release_progress = asyncio.Event()
    late_outcome_ready = asyncio.Event()

    monkeypatch.setattr(worker_module, "_FINALIZATION_DRAIN_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(worker_module, "capture_workspace_snapshot", AsyncMock(return_value=WorkspaceSnapshot()))
    monkeypatch.setattr(worker_module, "record_workspace_changes", AsyncMock(return_value=None))
    monkeypatch.setattr(
        worker_module,
        "_produced_output_paths",
        AsyncMock(return_value=["/mnt/user-data/outputs/report.md"]),
    )

    async def late_finalization(*args, **kwargs):
        await asyncio.sleep(0.05)
        late_outcome_ready.set()
        return False

    monkeypatch.setattr(worker_module, "_persist_journal_and_delivery_receipt", late_finalization)

    async def blocked_progress(run_id, **kwargs):
        progress_started.set()
        await release_progress.wait()

    monkeypatch.setattr(run_manager, "update_finalizing_progress", blocked_progress)

    class DummyAgent:
        async def astream(self, graph_input, config=None, stream_mode=None, subgraphs=False):
            journal = config["context"]["__run_journal"]
            journal._remember_current_run_tool_calls(
                AIMessage(content="", tool_calls=[{"id": "call_1", "name": "present_files", "args": {}}]),
                caller="lead_agent",
            )
            journal.on_tool_end(
                Command(
                    update={
                        "artifacts": ["/mnt/user-data/outputs/report.md"],
                        "messages": [ToolMessage("Successfully presented files", tool_call_id="call_1")],
                    }
                ),
                run_id=uuid4(),
            )
            yield {"messages": []}

    task = asyncio.create_task(
        run_agent(
            _make_bridge(),
            run_manager,
            record,
            ctx=RunContext(checkpointer=None, event_store=event_store),
            agent_factory=lambda *, config: DummyAgent(),
            graph_input={},
            config={},
        )
    )
    record.task = task
    try:
        await asyncio.wait_for(progress_started.wait(), timeout=2.0)
        await asyncio.wait_for(late_outcome_ready.wait(), timeout=2.0)
        assert len(run_manager._background_finalization_tasks) == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        async with asyncio.timeout(0.3):
            while run_manager._background_finalization_tasks:
                await asyncio.sleep(0)
        assert record.status == RunStatus.error
        assert record.error == "Artifact delivery verification failed: terminal delivery receipt could not be persisted"
    finally:
        release_progress.set()
        await asyncio.gather(task, return_exceptions=True)
        for finalization_task in tuple(run_manager._background_finalization_tasks):
            finalization_task.cancel()
        if run_manager._background_finalization_tasks:
            await asyncio.gather(*run_manager._background_finalization_tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_delivery_event_emitted_when_checkpoint_preflight_fails(monkeypatch):
    run_manager = RunManager()
    run_manager.update_run_completion = AsyncMock(wraps=run_manager.update_run_completion)
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    compatibility_check = AsyncMock(side_effect=RuntimeError("incompatible checkpoint"))
    monkeypatch.setattr("deerflow.runtime.runs.worker.aensure_checkpoint_mode_compatible", compatibility_check)

    def unexpected_agent_factory(**kwargs):
        raise AssertionError("agent must not be built after preflight failure")

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=object(), event_store=store),
        agent_factory=unexpected_agent_factory,
        graph_input={},
        config={},
    )

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert len(delivery) == 1
    assert delivery[0]["content"] == {"presented": 0, "paths": [], "by_tool": {}}
    fetched = await run_manager.get(record.run_id)
    assert fetched.status == RunStatus.error
    run_manager.update_run_completion.assert_not_awaited()


@pytest.mark.anyio
async def test_delivery_event_emitted_when_cancelled_waiting_for_prior_finalization(monkeypatch):
    run_manager = RunManager()
    run_manager.update_run_completion = AsyncMock(wraps=run_manager.update_run_completion)
    record = await run_manager.create("thread-1")
    store = MemoryRunEventStore()
    monkeypatch.setattr(
        run_manager,
        "wait_for_prior_finalizing",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    def unexpected_agent_factory(**kwargs):
        raise AssertionError("agent must not be built after preflight cancellation")

    await run_agent(
        _make_bridge(),
        run_manager,
        record,
        ctx=RunContext(checkpointer=None, event_store=store),
        agent_factory=unexpected_agent_factory,
        graph_input={},
        config={},
    )

    delivery = await _delivery_events(store, "thread-1", record.run_id)
    assert len(delivery) == 1
    assert delivery[0]["content"] == {"presented": 0, "paths": [], "by_tool": {}}
    fetched = await run_manager.get(record.run_id)
    assert fetched.status == RunStatus.interrupted
    run_manager.update_run_completion.assert_not_awaited()
