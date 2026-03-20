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
        async with self.session_maker() as session:
            providers = await ProviderModel.list(
                db_session=session,
                cursor=after,
                limit=limit,
                actor=actor,
            )
            return [provider.to_pydantic() for provider in providers]

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
