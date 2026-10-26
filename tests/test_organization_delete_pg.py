"""Deleting an organization on Postgres, whose foreign keys are enforced.

    scripts/run_pg_integration_tests.sh tests/test_organization_delete_pg.py

The session store and skill learning tables reference users and organizations
with no ORM cascade: the organization's delete must clear them first, or the
users' cascaded delete fails on their foreign keys.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from mirix.settings import settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not settings.mirix_pg_uri_no_default,
        reason="needs Postgres: run via scripts/run_pg_integration_tests.sh",
    ),
]


async def test_an_organization_with_session_records_can_be_deleted():
    import mirix.orm  # noqa: F401 -- register every ORM class before create_all
    from mirix.orm.conversation_session import ConversationSession
    from mirix.orm.organization import Organization
    from mirix.orm.user import User
    from mirix.server.server import db_context, engine, ensure_tables_created
    from mirix.services.organization_manager import OrganizationManager

    await ensure_tables_created()
    org_id = f"org-{uuid.uuid4()}"
    user_id = f"user-{uuid.uuid4()}"
    now = datetime.now(timezone.utc)
    async with db_context() as db:
        db.add(Organization(id=org_id, name=org_id))
        await db.flush()
        db.add(
            User(
                id=user_id,
                name=user_id,
                organization_id=org_id,
                status="active",
                timezone="UTC",
            )
        )
        await db.flush()
        db.add(
            ConversationSession(
                organization_id=org_id,
                user_id=user_id,
                session_id="s1",
                first_message_at=now,
                last_message_at=now,
            )
        )
        await db.commit()

    await OrganizationManager().delete_organization_by_id(org_id)

    async with db_context() as db:
        for model, column in (
            (Organization, Organization.id),
            (User, User.id),
            (ConversationSession, ConversationSession.user_id),
        ):
            value = org_id if model is Organization else user_id
            count = (
                await db.execute(select(func.count()).select_from(model).where(column == value))
            ).scalar_one()
            assert count == 0, model.__tablename__
    await engine.dispose()
