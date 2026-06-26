from typing import Any, Dict, List, Optional

from sqlalchemy import select

from mirix.log import get_logger
from mirix.orm.block import Block as BlockModel
from mirix.orm.enums import AccessType
from mirix.orm.errors import NoResultFound
from mirix.schemas.block import Block
from mirix.schemas.block import Block as PydanticBlock
from mirix.schemas.block import BlockUpdate
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.user import User as PydanticUser
from mirix.utils import enforce_types

logger = get_logger(__name__)


class BlockManager:
    """Manager class to handle business logic related to Blocks."""

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def create_or_update_block(
        self,
        block: Block,
        actor: PydanticClient,
        user: Optional["PydanticUser"] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
    ) -> PydanticBlock:
        """
        Create a new block or update an existing one (with Redis Hash caching).

        Scope is always auto-injected from actor.write_scope unless filter_tags
        already contains a "scope" key. This ensures every block is scoped to the
        client's write_scope. Callers do not need to pass scope explicitly.

        Args:
            block: Block data to create
            actor: Client for audit trail and scope resolution
            user: Optional user for data scoping
            filter_tags: Optional extra filter tags (e.g. scope, env). Scope is auto-injected from
                         actor.write_scope if not already present. Only used when creating a new block.

        Returns:
            PydanticBlock: The created or updated block
        """
        if filter_tags is None:
            filter_tags = {}
        if "scope" not in filter_tags and actor.write_scope:
            filter_tags["scope"] = actor.write_scope

        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            existing = await provider.read("block", block.id, actor=actor)
            if existing:
                update_data = BlockUpdate(**block.model_dump(exclude_none=True))
                return await self.update_block(block.id, update_data, actor, user=user)
            data = block.model_dump(
                exclude_none=True,
                exclude={"organization_id", "user_id", "filter_tags"},
            )
            final_user_id = user.id if user else None
            data["organization_id"] = actor.organization_id
            data["user_id"] = final_user_id
            data["filter_tags"] = filter_tags or None
            result = await provider.create("block", data, actor=actor)
            return PydanticBlock(**result)

        db_block = await self.get_block_by_id(block.id, user=None)
        if db_block:
            update_data = BlockUpdate(**block.model_dump(exclude_none=True))
            return await self.update_block(block.id, update_data, actor, user=user)
        else:
            async with self.session_maker() as session:
                data = block.model_dump(
                    exclude_none=True,
                    exclude={"organization_id", "user_id", "filter_tags"},
                )
                final_user_id = user.id if user else None
                scope = filter_tags.get("scope")
                logger.debug(
                    "Creating block with user_id=%s, scope=%s, org_id=%s",
                    final_user_id,
                    scope,
                    actor.organization_id,
                )
                block = BlockModel(
                    **data,
                    organization_id=actor.organization_id,
                    user_id=final_user_id,
                    filter_tags=filter_tags or None,
                )
                await block.create_with_redis(session, actor=actor)
                logger.debug("Block %s created with user_id=%s, scope=%s", block.id, block.user_id, scope)
            return block.to_pydantic()

    @enforce_types
    async def seed_template_block_for_actor_scope_if_necessary(
        self,
        label: str,
        value: str,
        limit: int,
        actor: PydanticClient,
        default_user: PydanticUser,
    ) -> PydanticBlock | None:
        """
        Ensure a template block exists for (user, scope, label). Idempotent.

        If a block with the same (user_id, scope, label) already exists, this
        function no-ops.

        Args:
            label: Block label (e.g. "human", "persona")
            value: Initial block content
            limit: Character limit
            actor: Client for audit trail and scope resolution

        Returns:
            PydanticBlock: The created or updated block
        """
        if actor.write_scope is None:
            return None
        scope = actor.write_scope

        # Look for existing block by key: (user_id, scope, label)
        existing = await self.get_blocks(
            user=default_user,
            any_scopes=[scope],
            label=label,
            auto_create_from_default=False,
        )
        if existing:
            return None

        # Create new block
        new_block = Block(
            label=label,
            value=value,
            limit=limit,
            filter_tags={"scope": scope},
            organization_id=actor.organization_id,
            user_id=default_user.id,
            created_by_id=default_user.id,
            last_updated_by_id=default_user.id,
        )
        logger.debug(
            "Creating template block: label=%s, scope=%s, user_id=%s",
            label,
            scope,
            default_user.id,
        )
        return await self.create_or_update_block(
            block=new_block,
            actor=actor,
            user=default_user,
        )

    @enforce_types
    async def _invalidate_block_cache(self, block_id: str) -> None:
        """
        Invalidate caches for a block.
        Called when a block is updated or deleted to maintain cache consistency.
        """
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()
            if cache_provider:
                cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                await cache_provider.delete(cache_key)
        except Exception as e:
            logger.warning("Failed to invalidate cache for block %s: %s", block_id, e)

    @enforce_types
    async def update_block(
        self,
        block_id: str,
        block_update: BlockUpdate,
        actor: PydanticClient,
        user: Optional["PydanticUser"] = None,
    ) -> PydanticBlock:
        """
        Update a block by its ID (with Redis Hash caching).

        Args:
            block_id: ID of the block to update
            block_update: BlockUpdate with fields to update
            actor: Client for audit trail (last_updated_by_id)
            user: Optional user if updating user field
        """
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            existing = await provider.read("block", block_id, actor=actor)
            if existing is None:
                raise NoResultFound(f"Block {block_id} not found")
            update_data = block_update.model_dump(exclude_unset=True, exclude_none=True)
            if user is not None:
                update_data["user_id"] = user.id
            update_data["last_updated_by_id"] = actor.id
            result = await provider.update("block", block_id, update_data, actor=actor)
            await self._invalidate_block_cache(block_id)
            return PydanticBlock(**result)

        async with self.session_maker() as session:
            block = await BlockModel.read(
                db_session=session, identifier=block_id, actor=actor, user=user, access_type=AccessType.USER
            )
            update_data = block_update.model_dump(exclude_unset=True, exclude_none=True)

            for key, value in update_data.items():
                setattr(block, key, value)

            if user is not None:
                block.user_id = user.id

            await block.update_with_redis(db_session=session, actor=actor)

            return block.to_pydantic()

    @enforce_types
    async def update_block_filter_tags(
        self,
        block_id: str,
        new_filter_tags: Dict[str, Any],
        actor: PydanticClient,
        user: Optional["PydanticUser"] = None,
    ) -> None:
        """
        Update only the filter_tags on a block and persist to DB + cache.
        """
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            existing = await provider.read("block", block_id, actor=actor)
            if existing is None:
                raise NoResultFound(f"Block {block_id} not found")
            await provider.update("block", block_id, {"filter_tags": new_filter_tags}, actor=actor)
            await self._invalidate_block_cache(block_id)
            return

        async with self.session_maker() as session:
            block = await BlockModel.read(
                db_session=session, identifier=block_id, actor=actor, user=user, access_type=AccessType.USER
            )
            block.filter_tags = new_filter_tags
            await block.update_with_redis(db_session=session, actor=actor)

    @enforce_types
    async def delete_block(self, block_id: str, actor: PydanticClient) -> PydanticBlock:
        """Delete a block by its ID (removes from cache)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            existing = await provider.read("block", block_id, actor=actor)
            if existing is None:
                raise NoResultFound(f"Block {block_id} not found")
            await provider.delete("block", block_id, actor=actor)
            await self._invalidate_block_cache(block_id)
            return PydanticBlock(**existing)

        from mirix.database.cache_provider import get_cache_provider

        async with self.session_maker() as session:
            block = await BlockModel.read(db_session=session, identifier=block_id)

            cache_provider = get_cache_provider()
            if cache_provider:
                cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                await cache_provider.delete(cache_key)

            await self._invalidate_block_cache(block_id)

            await block.hard_delete(db_session=session, actor=actor)
            return block.to_pydantic()

    @enforce_types
    async def get_blocks(
        self,
        user: Optional[PydanticUser] = None,
        any_scopes: Optional[List[str]] = None,
        label: Optional[str] = None,
        id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = 50,
        auto_create_from_default: bool = True,
        organization_id: Optional[str] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        filter_tags_set_on_create: Optional[Dict[str, Any]] = None,
    ) -> List[PydanticBlock]:
        """
        Retrieve blocks based on various optional filters.

        Args:
            user: User to get blocks for. If None, org-wide query (requires organization_id and any_scopes).
            any_scopes: If provided, only return blocks whose scope matches any value
                        in this list. Pass a single-element list for exact scope match
                        (e.g. [client.write_scope]) or a multi-element list for read
                        access across scopes (e.g. client.read_scopes).
                        An empty list means no scope access — returns [].
            label: Optional label filter
            id: Optional block ID filter
            cursor: Pagination cursor
            limit: Max results
            auto_create_from_default: If True and any_scopes has exactly one scope,
                                      copy default user's template blocks when none exist.
                                      Ignored when user is None (org-wide).
            organization_id: Required when user is None (org-wide query). Ignored when user is set.
            filter_tags: Optional dict; when provided, only blocks whose filter_tags contain
                         these keys/values are returned (passed to list_by_scopes).
            filter_tags_set_on_create: Optional dict; applied only when new blocks are created (e.g. from
                              default user template). Existing blocks are never updated.
        """
        from mirix.database.relational_provider import get_relational_provider

        # Blocks are read from the relational store (strong read-after-write),
        # never the search index. Core memory has no ranking semantics — a caller
        # always wants every block for a (user, scope) — and several of these
        # reads gate a write decision (the auto-create-from-template branch below,
        # the template existence check in seed_template_block_for_actor_scope_if_necessary)
        # or feed the agent pipeline moments after a sibling step wrote a block.
        # The search index is eventually consistent, so a read against it inside
        # the propagation window returns a stale empty result, which both (a) drives
        # spurious template duplication and (b) crashes core_memory_append when the
        # agent's loaded blocks get clobbered by an empty reload.
        relational_provider = get_relational_provider()
        if relational_provider:
            list_kwargs: Dict[str, Any] = {}
            if label is not None:
                list_kwargs["label"] = label
            if id is not None:
                list_kwargs["id"] = id

            effective_limit = limit or 50

            if user is None:
                if organization_id is None or any_scopes is None or not any_scopes:
                    return []
                results = await relational_provider.list(
                    "block",
                    user_id=None,
                    organization_id=organization_id,
                    filter_tags=filter_tags,
                    scopes=any_scopes,
                    limit=effective_limit,
                    **list_kwargs,
                )
                return [PydanticBlock(**r) for r in results]

            org_id = user.organization_id
            if any_scopes is not None and not any_scopes:
                return []

            if any_scopes is not None:
                scope_list = any_scopes
            else:
                scope_list = None

            results = await relational_provider.list(
                "block",
                user_id=user.id,
                organization_id=org_id,
                filter_tags=filter_tags,
                scopes=scope_list,
                limit=effective_limit,
                **list_kwargs,
            )
            pydantic_blocks = [PydanticBlock(**r) for r in results]
            if not pydantic_blocks and auto_create_from_default and any_scopes and len(any_scopes) == 1:
                scope = any_scopes[0]
                assert org_id is not None
                logger.debug(
                    "No blocks found for user %s, scope %s. Creating from default user template via provider.",
                    user.id,
                    scope,
                )
                return await self._copy_blocks_from_default_user_via_provider(
                    target_user=user,
                    scope=scope,
                    organization_id=org_id,
                    block_filter_tags=filter_tags_set_on_create,
                    relational_provider=relational_provider,
                )
            return pydantic_blocks

        async with self.session_maker() as session:
            # Org-wide path: user is None — require organization_id and any_scopes
            if user is None:
                if organization_id is None or any_scopes is None or not any_scopes:
                    return []
                blocks = await BlockModel.list_by_scopes(
                    db_session=session,
                    user_id=None,
                    organization_id=organization_id,
                    scopes=any_scopes,
                    label=label,
                    id=id,
                    limit=limit or 50,
                    filter_tags=filter_tags,
                )
                return [block.to_pydantic() for block in blocks]

            # Single-user path
            org_id = user.organization_id
            if any_scopes is not None:
                if not any_scopes:
                    return []
                # Scope-filtered query — pushed into SQL, hits the btree index
                blocks = await BlockModel.list_by_scopes(
                    db_session=session,
                    user_id=user.id,
                    organization_id=org_id,
                    scopes=any_scopes,
                    label=label,
                    id=id,
                    limit=limit or 50,
                    filter_tags=filter_tags,
                )
            else:
                # Unscoped query — returns all blocks for this user
                filters = {
                    "organization_id": org_id,
                    "user_id": user.id,
                }
                if label:
                    filters["label"] = label
                if id:
                    filters["id"] = id
                blocks = await BlockModel.list(db_session=session, cursor=cursor, limit=limit, **filters)

            # Auto-create from default user template if no blocks found.
            # Only auto-create when filtering by exactly one scope (write path).
            if not blocks and auto_create_from_default and any_scopes and len(any_scopes) == 1:
                scope = any_scopes[0]
                assert org_id is not None
                logger.debug(
                    "No blocks found for user %s, scope %s. Creating from default user template.",
                    user.id,
                    scope,
                )
                blocks = await self._copy_blocks_from_default_user(
                    session=session,
                    target_user=user,
                    scope=scope,
                    organization_id=org_id,
                    block_filter_tags=filter_tags_set_on_create,
                )

            return [block.to_pydantic() for block in blocks]

    async def search_blocks(
        self,
        *,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        query: str = "",
        search_method: str = "bm25",
        search_field: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        filter_tags: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 50,
    ) -> List[PydanticBlock]:
        """Search core memory blocks via the search provider (IPS Search).

        This is the read path for query-driven retrieval (the search endpoints),
        and is the only block read path that honours ``filter_tags`` — they are
        applied server-side by the search provider's query builder.

        It is deliberately separate from :meth:`get_blocks`, which reads from the
        relational store for the save/agent pipeline (strong read-after-write,
        no ranking, gates write decisions). ``search_blocks`` must NOT be used on
        the save path: the search index is eventually consistent.

        Pass ``user_id`` for a single-user search, or leave it ``None`` and pass
        ``organization_id`` for an org-wide (cross-user) search.
        """
        from mirix.database.search_provider import get_search_provider

        search_provider = get_search_provider()
        if not search_provider:
            # PostgreSQL fallback (no search provider registered): read blocks
            # directly from the relational store, applying scope + filter_tags
            # via the operator-aware ORM builder. Unlike the save-path get_blocks,
            # this honours filter_tags ($contains etc.) so block_filter_tags work
            # in non-provider mode. A non-empty query does a case-insensitive
            # substring match on the block value (no BM25 ranking in fallback).
            if organization_id is None:
                return []
            async with self.session_maker() as session:
                blocks = await BlockModel.list_by_scopes(
                    db_session=session,
                    user_id=user_id,
                    organization_id=organization_id,
                    scopes=scopes or [],
                    limit=limit or 50,
                    filter_tags=filter_tags,
                )
                if query:
                    needle = query.lower()
                    blocks = [b for b in blocks if b.value and needle in b.value.lower()]
                return [b.to_pydantic() for b in blocks]

        results, _cursor = await search_provider.search(
            "block",
            query_text=query,
            search_method=search_method,
            search_field=search_field,
            user_id=user_id,
            organization_id=organization_id,
            filter_tags=filter_tags,
            scopes=scopes,
            limit=limit,
        )
        return [PydanticBlock(**r) for r in results]

    async def _copy_blocks_from_default_user_via_provider(
        self,
        target_user: PydanticUser,
        scope: str,
        organization_id: str,
        block_filter_tags: Optional[Dict[str, Any]] = None,
        *,
        relational_provider: Any,
    ) -> List[PydanticBlock]:
        """Provider-aware counterpart of :meth:`_copy_blocks_from_default_user`.

        Finds template blocks for ``scope`` on the org default user via the
        Relational provider, then creates per-user copies through
        ``provider.create``. Templates are read from the relational store
        (strong read-after-write) rather than the search index: this read gates
        a write (the per-user copy), and a stale empty/duplicated result from
        the eventually-consistent search index would either skip the copy or
        fan out N redundant blocks per label.
        """
        from mirix.schemas.block import Block as PydanticBlock
        from mirix.services.user_manager import UserManager

        user_manager = UserManager()
        try:
            org_default_user = await user_manager.get_or_create_org_default_user(org_id=organization_id)
            default_user_id = org_default_user.id
        except Exception as exc:
            logger.warning("Failed to get org default user, falling back to global admin: %s", exc)
            default_user_id = UserManager.ADMIN_USER_ID

        # Find template blocks for this scope on the default user via the
        # Relational provider (strong read-after-write). Reading from the
        # search index here would risk a stale empty result (skip the copy
        # entirely) or stale duplicates (fan out redundant per-user blocks).
        template_results = await relational_provider.list(
            "block",
            user_id=default_user_id,
            organization_id=organization_id,
            filter_tags=None,
            scopes=[scope],
            limit=100,
        )

        if not template_results:
            logger.warning(
                "No template blocks found for scope %s via Relational provider. "
                "Ensure create_meta_agent was called with a blocks config for this scope. "
                "User %s will have no blocks.",
                scope,
                target_user.id,
            )
            return []

        # De-duplicate templates by ``label`` to mirror the PG ORM contract,
        # where ``seed_template_block_for_actor_scope_if_necessary`` enforces
        # exactly one template per (user, scope, label). Keep the first template
        # seen per label.
        deduped_templates: List[Dict[str, Any]] = []
        seen_labels: set[str] = set()
        for tpl in template_results:
            tpl_label = tpl.get("label")
            if not tpl_label or tpl_label in seen_labels:
                continue
            seen_labels.add(tpl_label)
            deduped_templates.append(tpl)
        if len(deduped_templates) != len(template_results):
            logger.info(
                "Deduped %d template duplicates for scope=%s on default " "user %s (kept %d unique label(s): %s)",
                len(template_results) - len(deduped_templates),
                scope,
                default_user_id,
                len(deduped_templates),
                sorted(seen_labels),
            )
        template_results = deduped_templates

        sanitized_bft = {k: v for k, v in (block_filter_tags or {}).items() if k != "scope"}
        merged_tags = {**sanitized_bft, "scope": scope}

        new_blocks: List[PydanticBlock] = []
        for template in template_results:
            try:
                new_block_id = PydanticBlock._generate_id()
                data = {
                    "id": new_block_id,
                    "label": template.get("label"),
                    "value": template.get("value"),
                    "limit": template.get("limit"),
                    "user_id": target_user.id,
                    "organization_id": organization_id,
                    "filter_tags": merged_tags,
                    "_created_by_id": target_user.id,
                    "_last_updated_by_id": target_user.id,
                }
                result = await relational_provider.create("block", data)
                new_blocks.append(PydanticBlock(**result))
            except Exception as exc:
                logger.error(
                    "Failed to copy template block %s for user %s: %s",
                    template.get("id"),
                    target_user.id,
                    exc,
                    exc_info=True,
                )
                continue

        logger.info(
            "Created %d blocks for user %s from provider template (scope=%s)",
            len(new_blocks),
            target_user.id,
            scope,
        )
        return new_blocks

    async def _copy_blocks_from_default_user(
        self,
        session,
        target_user: PydanticUser,
        scope: str,
        organization_id: str,
        block_filter_tags: Optional[Dict[str, Any]] = None,
    ) -> List[BlockModel]:
        """
        Copy template blocks from the default user to the target user for a given scope.

        Template blocks are seeded by create_meta_agent when a client provides a blocks
        config. They live on the org's default user with filter_tags={"scope": "<write_scope>"}.
        Clients sharing the same write_scope share the same template blocks.

        Args:
            session: Database session
            target_user: User to create blocks for
            scope: Scope to match template blocks and assign to new blocks
            organization_id: Organization ID

        Returns:
            List of newly created BlockModel instances
        """
        from mirix.services.user_manager import UserManager

        user_manager = UserManager()
        try:
            org_default_user = await user_manager.get_or_create_org_default_user(org_id=organization_id)
            default_user_id = org_default_user.id
            logger.debug(
                "Using organization default user %s as template for user %s in org %s",
                default_user_id,
                target_user.id,
                organization_id,
            )
        except Exception as e:
            logger.warning("Failed to get org default user, falling back to global admin: %s", e)
            default_user_id = UserManager.ADMIN_USER_ID

        # Find template blocks for this scope on the default user (SQL-level scope filter)
        default_blocks = await BlockModel.list_by_scopes(
            db_session=session,
            user_id=default_user_id,
            organization_id=organization_id,
            scopes=[scope],
            limit=100,
        )

        logger.debug(
            "Found %d template blocks for scope %s (default_user=%s, org=%s)",
            len(default_blocks),
            scope,
            default_user_id,
            organization_id,
        )

        if not default_blocks:
            logger.warning(
                "No template blocks found for scope %s. "
                "Ensure create_meta_agent was called with a blocks config for this scope. "
                "User %s will have no blocks.",
                scope,
                target_user.id,
            )
            return []

        new_blocks = []
        logger.debug("Starting to copy %d blocks for user %s (scope=%s)", len(default_blocks), target_user.id, scope)

        for template_block in default_blocks:
            logger.debug("Copying block %s (label=%s) from template user", template_block.id, template_block.label)

            try:
                from mirix.schemas.block import Block as PydanticBlock

                new_block_id = PydanticBlock._generate_id()

                sanitized_bft = {k: v for k, v in (block_filter_tags or {}).items() if k != "scope"}
                merged_tags = {**sanitized_bft, "scope": scope}
                new_block = BlockModel(
                    id=new_block_id,
                    label=template_block.label,
                    value=template_block.value,
                    limit=template_block.limit,
                    user_id=target_user.id,
                    organization_id=organization_id,
                    filter_tags=merged_tags,
                    created_by_id=target_user.id,
                    last_updated_by_id=target_user.id,
                )

                session.add(new_block)
                await session.commit()
                await session.refresh(new_block)

                try:
                    await new_block._update_redis_cache(operation="create", actor=None)
                    logger.debug("Cached copied block %s to cache", new_block.id)
                except Exception as e:
                    logger.warning("Failed to cache block %s to cache: %s", new_block.id, e)

                new_blocks.append(new_block)

                logger.debug(
                    "Created block %s (label=%s) for user %s, scope=%s from template %s",
                    new_block.id,
                    new_block.label,
                    target_user.id,
                    scope,
                    template_block.id,
                )
            except Exception as e:
                logger.error(
                    "Failed to copy block %s for user %s: %s", template_block.id, target_user.id, e, exc_info=True
                )
                await session.rollback()
                continue

        logger.info(
            "Created %d blocks for user %s from default user template (scope=%s)",
            len(new_blocks),
            target_user.id,
            scope,
        )

        return new_blocks

    @enforce_types
    async def get_block_by_id(self, block_id: str, user: Optional[PydanticUser] = None) -> Optional[PydanticBlock]:
        """Retrieve a block by its ID (with cache - Redis or Cache provider)."""
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            result = await provider.read("block", block_id)
            if result is None:
                return None
            return PydanticBlock(**result)

        cache_provider = None
        try:
            from mirix.database.cache_provider import get_cache_provider

            cache_provider = get_cache_provider()

            if cache_provider:
                cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                cached_data = await cache_provider.get_hash(cache_key)
                if cached_data:
                    if "value" not in cached_data or cached_data["value"] is None:
                        cached_data["value"] = ""
                    return PydanticBlock(**cached_data)
        except Exception as e:
            logger.debug("Cache read failed for block %s: %s", block_id, e)

        async with self.session_maker() as session:
            try:
                block = await BlockModel.read(
                    db_session=session,
                    identifier=block_id,
                    user=user,
                    access_type=AccessType.USER,
                )
                pydantic_block = block.to_pydantic()

                try:
                    if cache_provider:
                        from mirix.settings import settings

                        cache_key = f"{cache_provider.BLOCK_PREFIX}{block_id}"
                        data = pydantic_block.model_dump(mode="json")
                        await cache_provider.set_hash(cache_key, data, ttl=settings.redis_ttl_blocks)
                except Exception as e:
                    logger.warning("Failed to populate cache for block %s: %s", block_id, e)

                return pydantic_block
            except NoResultFound:
                return None

    @enforce_types
    async def get_all_blocks_by_ids(
        self, block_ids: List[str], user: Optional[PydanticUser] = None
    ) -> List[PydanticBlock]:
        blocks = []
        for block_id in block_ids:
            block = await self.get_block_by_id(block_id, user=user)
            blocks.append(block)
        return blocks

    async def soft_delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk soft delete all blocks for a user (updates Redis cache).

        Args:
            user_id: ID of the user whose blocks to soft delete

        Returns:
            Number of records soft deleted
        """
        from mirix.database.redis_client import get_redis_client
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            count = await provider.bulk_delete_with_events(
                "block",
                filters={"user_id": user_id},
                soft=True,
            )
            return int(count)

        async with self.session_maker() as session:
            stmt = select(BlockModel).where(
                BlockModel.user_id == user_id,
                BlockModel.is_deleted == False,
            )
            result = await session.execute(stmt)
            blocks = result.scalars().all()

            count = len(blocks)
            if count == 0:
                return 0

            block_ids = [block.id for block in blocks]

            for block in blocks:
                block.is_deleted = True
                block.set_updated_at()

            await session.commit()

        for block_id in block_ids:
            await self._invalidate_block_cache(block_id)

        redis_client = get_redis_client()
        if redis_client:
            for block_id in block_ids:
                redis_key = f"{redis_client.BLOCK_PREFIX}{block_id}"
                try:
                    await redis_client.client.hset(redis_key, "is_deleted", "true")
                except Exception:
                    await redis_client.delete(redis_key)

        return count

    async def delete_by_user_id(self, user_id: str) -> int:
        """
        Bulk hard delete all blocks for a user (removes from Redis cache).

        Args:
            user_id: ID of the user whose blocks to delete

        Returns:
            Number of records deleted
        """
        from sqlalchemy import delete

        from mirix.database.redis_client import get_redis_client
        from mirix.database.relational_provider import get_relational_provider

        provider = get_relational_provider()
        if provider:
            count = await provider.bulk_delete_with_events(
                "block",
                filters={"user_id": user_id},
                soft=False,
            )
            return int(count)

        async with self.session_maker() as session:
            stmt = select(BlockModel.id).where(BlockModel.user_id == user_id)
            result = await session.execute(stmt)
            block_ids = [row[0] for row in result.all()]

            count = len(block_ids)
            if count == 0:
                return 0

            for block_id in block_ids:
                await self._invalidate_block_cache(block_id)

            await session.execute(delete(BlockModel).where(BlockModel.user_id == user_id))
            await session.commit()

        redis_client = get_redis_client()
        if redis_client and block_ids:
            redis_keys = [f"{redis_client.BLOCK_PREFIX}{block_id}" for block_id in block_ids]

            BATCH_SIZE = 1000
            for i in range(0, len(redis_keys), BATCH_SIZE):
                batch = redis_keys[i : i + BATCH_SIZE]
                await redis_client.client.delete(*batch)

        return count
