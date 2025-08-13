from typing import List, Optional

from mirix.orm.provider import Provider as ProviderModel
from mirix.schemas.providers import Provider as PydanticProvider
from mirix.schemas.providers import ProviderUpdate
from mirix.schemas.user import User as PydanticUser
from mirix.utils import enforce_types


class ProviderManager:

    def __init__(self):
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def insert_provider(self, name: str, api_key: str, organization_id: str, actor: PydanticUser) -> PydanticProvider:
        """Insert a new provider into the database."""
        self.create_provider(
            PydanticProvider(
                name=name,
                api_key=api_key,
                organization_id=organization_id,
            ),
            actor=actor,
        )
    
    @enforce_types
    def upsert_provider(self, name: str, api_key: str, organization_id: str, actor: PydanticUser) -> PydanticProvider:
        """Insert or update a provider in the database. Updates if exists, creates if not."""
        with self.session_maker() as session:
            # Check if provider already exists for this organization
            existing_providers = [p for p in self.list_providers(actor=actor) if p.name == name]
            
            if existing_providers:
                # Update existing provider
                existing_provider = existing_providers[0]
                provider_update = ProviderUpdate(id=existing_provider.id, api_key=api_key)
                return self.update_provider(existing_provider.id, provider_update, actor)
            else:
                # Create new provider
                return self.create_provider(
                    PydanticProvider(
                        name=name,
                        api_key=api_key,
                        organization_id=organization_id,
                    ),
                    actor=actor,
                )

    @enforce_types
    def create_provider(self, provider: PydanticProvider, actor: PydanticUser) -> PydanticProvider:
        """Create a new provider if it doesn't already exist."""
        with self.session_maker() as session:
            # Assign the organization id based on the actor
            provider.organization_id = actor.organization_id

            # Lazily create the provider id prior to persistence
            provider.resolve_identifier()

            new_provider = ProviderModel(**provider.model_dump(exclude_unset=True))
            new_provider.create(session, actor=actor)
            return new_provider.to_pydantic()

    @enforce_types
    def update_provider(self, provider_id: str, provider_update: ProviderUpdate, actor: PydanticUser) -> PydanticProvider:
        """Update provider details."""
        with self.session_maker() as session:
            # Retrieve the existing provider by ID
            existing_provider = ProviderModel.read(db_session=session, identifier=provider_id, actor=actor)

            # Update only the fields that are provided in ProviderUpdate
            update_data = provider_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_provider, key, value)

            # Commit the updated provider
            existing_provider.update(session, actor=actor)
            return existing_provider.to_pydantic()

    @enforce_types
    def delete_provider_by_id(self, provider_id: str, actor: PydanticUser):
        """Delete a provider."""
        with self.session_maker() as session:
            # Clear api key field
            existing_provider = ProviderModel.read(db_session=session, identifier=provider_id, actor=actor)
            existing_provider.api_key = None
            existing_provider.update(session, actor=actor)

            # Soft delete in provider table
            existing_provider.delete(session, actor=actor)

            session.commit()

    @enforce_types
    def list_providers(self, after: Optional[str] = None, limit: Optional[int] = 50, actor: PydanticUser = None) -> List[PydanticProvider]:
        """List all providers with optional pagination."""
        with self.session_maker() as session:
            providers = ProviderModel.list(
                db_session=session,
                cursor=after,
                limit=limit,
                actor=actor,
            )
            return [provider.to_pydantic() for provider in providers]

    @enforce_types
    def get_anthropic_override_provider_id(self) -> Optional[str]:
        """Helper function to fetch custom anthropic provider id for v0 BYOK feature"""
        anthropic_provider = [provider for provider in self.list_providers() if provider.name == "anthropic"]
        if len(anthropic_provider) != 0:
            return anthropic_provider[0].id
        return None

    @enforce_types
    def get_anthropic_override_key(self) -> Optional[str]:
        """Helper function to fetch custom anthropic key for v0 BYOK feature"""
        anthropic_provider = [provider for provider in self.list_providers() if provider.name == "anthropic"]
        if len(anthropic_provider) != 0:
            return anthropic_provider[0].api_key
        return None

    @enforce_types
    def get_gemini_override_provider_id(self) -> Optional[str]:
        """Helper function to fetch custom gemini provider id for v0 BYOK feature"""
        gemini_provider = [provider for provider in self.list_providers() if provider.name == "google_ai"]
        if len(gemini_provider) != 0:
            return gemini_provider[0].id
        return None

    @enforce_types
    def get_gemini_override_key(self) -> Optional[str]:
        """Helper function to fetch custom gemini key for v0 BYOK feature"""
        gemini_provider = [provider for provider in self.list_providers() if provider.name == "google_ai"]
        if len(gemini_provider) != 0:
            return gemini_provider[0].api_key
        return None

    @enforce_types
    def get_openai_override_provider_id(self) -> Optional[str]:
        """Helper function to fetch custom openai provider id for v0 BYOK feature"""
        openai_provider = [provider for provider in self.list_providers() if provider.name == "openai"]
        if len(openai_provider) != 0:
            return openai_provider[0].id
        return None

    @enforce_types
    def get_openai_override_key(self) -> Optional[str]:
        """Helper function to fetch custom openai key for v0 BYOK feature"""
        openai_provider = [provider for provider in self.list_providers() if provider.name == "openai"]
        if len(openai_provider) != 0:
            return openai_provider[0].api_key
        return None

    @enforce_types
    def get_azure_openai_override_provider_id(self) -> Optional[str]:
        """Helper function to fetch custom azure openai provider id for v0 BYOK feature"""
        azure_openai_provider = [provider for provider in self.list_providers() if provider.name == "azure_openai"]
        if len(azure_openai_provider) != 0:
            return azure_openai_provider[0].id
        return None

    @enforce_types
    def get_azure_openai_override_key(self) -> Optional[str]:
        """Helper function to fetch custom azure openai key for v0 BYOK feature"""
        azure_openai_provider = [provider for provider in self.list_providers() if provider.name == "azure_openai"]
        if len(azure_openai_provider) != 0:
            return azure_openai_provider[0].api_key
        return None