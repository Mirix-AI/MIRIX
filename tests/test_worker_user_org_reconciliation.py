"""Unit tests for save-path org reconciliation (returning-user stale-org bug).

The ``users`` row is a global, id-only-PK shared stub (VEPAGE-1155): its
``organization_id`` is pinned to whichever org first created the user. On the
save path the worker resolves the user via ``get_user_by_id``, which returns
that stub row verbatim — so a user first seen in org A, then saved into org B,
arrives carrying org A. Downstream block scoping and the temporal guard then
operate against org A (the stale org) instead of org B (the client's org),
which (a) creates/loads the per-user block in the wrong org and (b) lets the
temporal guard find citations the current org's reset never touched.

The fix reconciles the resolved user's org to the client/actor's org before the
save proceeds. These tests pin that behavior in isolation (no DB, no server).
"""

from types import SimpleNamespace

from mirix.queue.worker import reconcile_user_org_to_actor
from mirix.schemas.user import User as PydanticUser


def _user(org_id):
    return PydanticUser(
        id="user-returning",
        name="returning",
        organization_id=org_id,
        timezone="UTC",
    )


def test_returning_user_stale_org_is_reconciled_to_actor_org():
    """A user stub pinned to its first-seen org is corrected to the actor's org."""
    user = _user("org-FIRST-SEEN")
    actor = SimpleNamespace(organization_id="org-CURRENT")

    result = reconcile_user_org_to_actor(user, actor)

    assert result.organization_id == "org-CURRENT"
    # Identity is unchanged — only the org is corrected.
    assert result.id == "user-returning"


def test_matching_org_returns_user_unchanged():
    """When the stub org already matches the actor org, nothing changes."""
    user = _user("org-CURRENT")
    actor = SimpleNamespace(organization_id="org-CURRENT")

    result = reconcile_user_org_to_actor(user, actor)

    assert result.organization_id == "org-CURRENT"


def test_does_not_mutate_input_user():
    """Reconciliation must not mutate the passed-in user object in place."""
    user = _user("org-FIRST-SEEN")
    actor = SimpleNamespace(organization_id="org-CURRENT")

    reconcile_user_org_to_actor(user, actor)

    assert user.organization_id == "org-FIRST-SEEN"


def test_none_user_is_passed_through():
    """A None user (no user_id on the message) is returned as-is."""
    actor = SimpleNamespace(organization_id="org-CURRENT")

    assert reconcile_user_org_to_actor(None, actor) is None


def test_actor_without_org_leaves_user_untouched():
    """If the actor has no org, do not blank out the user's org."""
    user = _user("org-FIRST-SEEN")
    actor = SimpleNamespace(organization_id=None)

    result = reconcile_user_org_to_actor(user, actor)

    assert result.organization_id == "org-FIRST-SEEN"
