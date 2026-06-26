from typing import List, Optional

from mirix.orm.provider import Provider as ProviderModel
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.providers import Provider as PydanticProvider
from mirix.schemas.providers import ProviderUpdate
from mirix.utils import enforce_types


class ProviderManager:
    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    async def insert_provider(
        self, name: str, api_key: str, organization_id: str, actor: PydanticClient
    ) -> PydanticProvider:
        """Insert a new provider into the database."""
        return await self.create_provider(
            PydanticProvider(
                name=name,
                api_key=api_key,
                organization_id=organization_id,
            ),
            actor=actor,
        )

    @enforce_types
    async def upsert_provider(
        self, name: str, api_key: str, organization_id: str, actor: PydanticClient
    ) -> PydanticProvider:
        """Insert or update a provider. Updates if exists, creates if not."""
        existing_providers = [p for p in await self.list_providers(actor=actor) if p.name == name]

        if existing_providers:
            existing_provider = existing_providers[0]
            provider_update = ProviderUpdate(id=existing_provider.id, api_key=api_key)
            return await self.update_provider(existing_provider.id, provider_update, actor)
        return await self.create_provider(
            PydanticProvider(
                name=name,
                api_key=api_key,
                organization_id=organization_id,
            ),
            actor=actor,
        )

    @enforce_types
    async def create_provider(self, provider: PydanticProvider, actor: PydanticClient) -> PydanticProvider:
        """Create a new provider if it doesn't already exist."""
        from mirix.database.relational_provider import get_relational_provider

        rp = get_relational_provider()
        if rp:
            provider.organization_id = actor.organization_id
            provider.resolve_identifier()
            provider_data = provider.model_dump(exclude_unset=True)
            provider_data["_created_by_id"] = actor.id
            result = await rp.create("providers", provider_data)
            return PydanticProvider(**result)
        async with self.session_maker() as session:
            provider.organization_id = actor.organization_id
            provider.resolve_identifier()
            new_provider = ProviderModel(**provider.model_dump(exclude_unset=True))
            await new_provider.create(session, actor=actor)
            return new_provider.to_pydantic()

    @enforce_types
    async def update_provider(
        self, provider_id: str, provider_update: ProviderUpdate, actor: PydanticClient
    ) -> PydanticProvider:
        """Update provider details."""
        from mirix.database.relational_provider import get_relational_provider

        rp = get_relational_provider()
        if rp:
            update_data = provider_update.model_dump(exclude_unset=True, exclude_none=True)
            result = await rp.update("providers", provider_id, update_data)
            return PydanticProvider(**result)
        async with self.session_maker() as session:
            existing_provider = await ProviderModel.read(db_session=session, identifier=provider_id, actor=actor)
            update_data = provider_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_provider, key, value)
            await existing_provider.update(session, actor=actor)
            return existing_provider.to_pydantic()

    @enforce_types
    async def delete_provider_by_id(self, provider_id: str, actor: PydanticClient) -> None:
        """Delete a provider."""
        from mirix.database.relational_provider import get_relational_provider

        rp = get_relational_provider()
        if rp:
            await rp.update("providers", provider_id, {"api_key": None})
            await rp.delete("providers", provider_id, soft=True)
            return
        async with self.session_maker() as session:
            existing_provider = await ProviderModel.read(db_session=session, identifier=provider_id, actor=actor)
            existing_provider.api_key = None
            await existing_provider.update(session, actor=actor)
            await existing_provider.delete(session, actor=actor)
            await session.commit()

    @enforce_types
    async def list_providers(
        self,
        after: Optional[str] = None,
        limit: Optional[int] = 50,
        actor: PydanticClient = None,
    ) -> List[PydanticProvider]:
        """List all providers with optional pagination."""
        from mirix.database.relational_provider import get_relational_provider

        rp = get_relational_provider()
        if rp:
            organization_id = actor.organization_id if actor is not None else None
            results = await rp.find_using_named_query(
                "providers",
                "provider_manager.list_providers",
                params={"organizationId": organization_id},
                page_size=limit or 50,
            )
            return [PydanticProvider(**r) for r in results]
        async with self.session_maker() as session:
            providers = await ProviderModel.list(
                db_session=session,
                cursor=after,
                limit=limit,
                actor=actor,
            )
            return [provider.to_pydantic() for provider in providers]

    @enforce_types
    async def get_provider_by_id(self, provider_id: str, actor: Optional[PydanticClient] = None) -> PydanticProvider:
        """Fetch a provider by ID, scoped by actor.organization_id when available."""
        from mirix.database.relational_provider import get_relational_provider
        from mirix.orm.errors import NoResultFound as _NRF

        rp = get_relational_provider()
        if rp:
            organization_id = actor.organization_id if actor is not None else None
            rows = await rp.find_using_named_query(
                "providers",
                "provider_manager.get_provider_by_id",
                params={"id": provider_id, "organizationId": organization_id},
                page_size=1,
            )
            if not rows:
                raise _NRF(f"Provider {provider_id} not found")
            return PydanticProvider(**rows[0])

        async with self.session_maker() as session:
            provider = await ProviderModel.read(db_session=session, identifier=provider_id, actor=actor)
            return provider.to_pydantic()

    @enforce_types
    async def get_anthropic_override_provider_id(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "anthropic"]
        return providers[0].id if providers else None

    @enforce_types
    async def get_anthropic_override_key(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "anthropic"]
        return providers[0].api_key if providers else None

    @enforce_types
    async def get_gemini_override_provider_id(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "google_ai"]
        return providers[0].id if providers else None

    @enforce_types
    async def get_gemini_override_key(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "google_ai"]
        return providers[0].api_key if providers else None

    @enforce_types
    async def get_openai_override_provider_id(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "openai"]
        return providers[0].id if providers else None

    @enforce_types
    async def get_openai_override_key(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "openai"]
        return providers[0].api_key if providers else None

    @enforce_types
    async def get_azure_openai_override_provider_id(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "azure_openai"]
        return providers[0].id if providers else None

    @enforce_types
    async def get_azure_openai_override_key(self) -> Optional[str]:
        providers = [p for p in await self.list_providers() if p.name == "azure_openai"]
        return providers[0].api_key if providers else None
