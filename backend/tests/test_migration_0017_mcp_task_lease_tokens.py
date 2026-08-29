"""Migration-chain coverage for ``0017_mcp_task_lease_tokens``."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

import deerflow.persistence.models  # noqa: F401 -- register the current ORM metadata
from deerflow.persistence.base import Base
from deerflow.persistence.bootstrap import bootstrap_schema
from deerflow.persistence.postgres_schema import build_asyncpg_connect_args

pytestmark = pytest.mark.asyncio

POSTGRES_URL = os.environ.get("DEERFLOW_TEST_POSTGRES_URL")
_LIBPQ_ONLY_QUERY_KEYS = {"sslmode", "channel_binding"}


def _asyncpg_url(url: str | None) -> str | None:
    """Normalize the CI/libpq DSN for SQLAlchemy's asyncpg driver."""
    if not url:
        return url
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    parts = urlsplit(url)
    if parts.query:
        kept = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key not in _LIBPQ_ONLY_QUERY_KEYS]
        url = urlunsplit(parts._replace(query=urlencode(kept)))
    return url


async def test_migration_chain_0016_to_0017_adds_nullable_mcp_task_lease_tokens(tmp_path: Path) -> None:
    """A versioned 0015 database must traverse both revisions without shape drift."""
    db_path = tmp_path / "deerflow.db"
    sync_engine = sa.create_engine(f"sqlite:///{db_path}")
    try:
        # Model metadata represents a deployment which has already acquired the
        # 0016 tables, while dropping only the 0017 columns restores the exact
        # pre-0017 mcp_tasks shape. Starting from 0015 still executes 0016
        # before 0017 and pins idempotence against an already-shaped database.
        Base.metadata.create_all(sync_engine)
        with sync_engine.begin() as conn:
            conn.execute(sa.text("ALTER TABLE mcp_tasks DROP COLUMN notification_lease_token"))
            conn.execute(sa.text("ALTER TABLE mcp_tasks DROP COLUMN lease_token"))
            conn.execute(sa.text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)"))
            conn.execute(sa.text("DELETE FROM alembic_version"))
            conn.execute(sa.text("INSERT INTO alembic_version (version_num) VALUES ('0015_scheduled_task_enqueue')"))
    finally:
        sync_engine.dispose()

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        await bootstrap_schema(engine, backend="sqlite")
        async with engine.connect() as conn:
            version = await conn.scalar(sa.text("SELECT version_num FROM alembic_version"))
            columns = {column["name"]: column for column in await conn.run_sync(lambda connection: sa.inspect(connection).get_columns("mcp_tasks"))}
            tables = set(await conn.run_sync(lambda connection: sa.inspect(connection).get_table_names()))

        assert version == "0017_mcp_task_lease_tokens"
        assert {"subagent_batches", "subagent_batch_items"} <= tables
        assert {"lease_token", "notification_lease_token"} <= columns.keys()
        assert columns["lease_token"]["nullable"] is True
        assert columns["notification_lease_token"]["nullable"] is True
    finally:
        await engine.dispose()


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set DEERFLOW_TEST_POSTGRES_URL to run the real PostgreSQL migration chain",
)
async def test_postgres_migration_chain_0016_to_0017_adds_nullable_mcp_task_lease_tokens() -> None:
    """Run the versioned 0015 -> 0016 -> 0017 path in an isolated PG schema."""
    schema = f"deerflow_migration_{uuid.uuid4().hex[:12]}"
    url = make_url(_asyncpg_url(POSTGRES_URL) or "")
    engine = create_async_engine(url, connect_args=build_asyncpg_connect_args(schema))
    try:
        async with engine.begin() as conn:
            await conn.execute(sa.schema.CreateSchema(schema))
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(sa.text("ALTER TABLE mcp_tasks DROP COLUMN notification_lease_token"))
            await conn.execute(sa.text("ALTER TABLE mcp_tasks DROP COLUMN lease_token"))
            await conn.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            await conn.execute(sa.text("INSERT INTO alembic_version (version_num) VALUES ('0015_scheduled_task_enqueue')"))

        await bootstrap_schema(engine, backend="postgres", postgres_schema=schema)

        async with engine.connect() as conn:
            version = await conn.scalar(sa.text("SELECT version_num FROM alembic_version"))
            columns = {column["name"]: column for column in await conn.run_sync(lambda connection: sa.inspect(connection).get_columns("mcp_tasks"))}

        assert version == "0017_mcp_task_lease_tokens"
        assert columns["lease_token"]["nullable"] is True
        assert columns["notification_lease_token"]["nullable"] is True
    finally:
        async with engine.begin() as conn:
            await conn.execute(sa.text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
