"""
Test script to verify raw memory embeddings are saved to the database with real API calls.

This script creates raw memories with real Gemini embeddings (requires GOOGLE_API_KEY).
Run this manually to verify embeddings are saved to PostgreSQL.

Usage:
    # Option 1: Set environment variable directly
    export GOOGLE_API_KEY="your-google-api-key"

    # Option 2: Create a .env file in the project root with:
    # GOOGLE_API_KEY=your-google-api-key
    # or
    # MIRIX_GOOGLE_API_KEY=your-google-api-key

    poetry run python tests/test_raw_memory_with_real_embeddings.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv

dotenv_path = project_root / ".env"
load_dotenv(dotenv_path=dotenv_path)
print(f"[INFO] Loaded environment variables from: {dotenv_path}")
if not dotenv_path.exists():
    print(f"       Note: .env file not found at {dotenv_path}")

from mirix.schemas.agent import CreateAgent
from mirix.schemas.client import Client as PydanticClient
from mirix.schemas.embedding_config import EmbeddingConfig
from mirix.schemas.llm_config import LLMConfig
from mirix.schemas.organization import Organization as PydanticOrganization
from mirix.schemas.raw_memory import RawMemoryItemCreate
from mirix.schemas.user import User as PydanticUser
from mirix.services.agent_manager import AgentManager
from mirix.services.client_manager import ClientManager
from mirix.services.organization_manager import OrganizationManager
from mirix.services.raw_memory_manager import RawMemoryManager
from mirix.services.user_manager import UserManager


def main():
    """Create a raw memory with real Gemini embeddings."""

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("MIRIX_GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY, GOOGLE_API_KEY, or MIRIX_GOOGLE_API_KEY environment variable not set")
        print("        Set it with: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    print("[OK] Google API key found")

    # Check BUILD_EMBEDDINGS_FOR_MEMORY setting
    from mirix.constants import BUILD_EMBEDDINGS_FOR_MEMORY

    print(f"[INFO] BUILD_EMBEDDINGS_FOR_MEMORY setting: {BUILD_EMBEDDINGS_FOR_MEMORY}")
    if not BUILD_EMBEDDINGS_FOR_MEMORY:
        print("[WARNING] BUILD_EMBEDDINGS_FOR_MEMORY is disabled!")
        print("          Embeddings will NOT be generated even with agent_state")
        print("          Set MIRIX_BUILD_EMBEDDINGS_FOR_MEMORY=true to enable")

    # Initialize managers
    org_mgr = OrganizationManager()
    client_mgr = ClientManager()
    user_mgr = UserManager()
    agent_mgr = AgentManager()
    raw_memory_mgr = RawMemoryManager()

    # Create organization
    org_id = "test-org-embeddings"
    try:
        org = org_mgr.get_organization_by_id(org_id)
        print(f"[OK] Using existing organization: {org_id}")
    except Exception:
        org = org_mgr.create_organization(PydanticOrganization(id=org_id, name="Test Organization for Embeddings"))
        print(f"[OK] Created organization: {org_id}")

    # Create client
    client_id = "test-client-embeddings"
    try:
        client = client_mgr.get_client_by_id(client_id)
        print(f"[OK] Using existing client: {client_id}")
    except Exception:
        client = client_mgr.create_client(
            PydanticClient(
                id=client_id,
                organization_id=org_id,
                name="Test Client for Embeddings",
                scope="read_write",
            )
        )
        print(f"[OK] Created client: {client_id}")

    # Create user
    user_id = "test-user-embeddings"
    try:
        user = user_mgr.get_user_by_id(user_id)
        print(f"[OK] Using existing user: {user_id}")
    except Exception:
        user = user_mgr.create_user(
            PydanticUser(
                id=user_id,
                organization_id=org_id,
                name="Test User for Embeddings",
                timezone="UTC",
            )
        )
        print(f"[OK] Created user: {user_id}")

    # Create agent with Gemini embedding config
    agent_id = "test-agent-gemini-embeddings"
    try:
        agent = agent_mgr.get_agent_by_id(agent_id, actor=client)
        print(f"[OK] Using existing agent: {agent_id}")
    except Exception:
        # Load config from mirix_gemini.yaml (same pattern as test_memory_server.py)
        from pathlib import Path

        import yaml

        config_path = Path("mirix/configs/examples/mirix_gemini.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        agent = agent_mgr.create_agent(
            CreateAgent(
                name="Test Agent Gemini Embeddings",
                description="Test agent with real Gemini embeddings from mirix_gemini.yaml",
                llm_config=LLMConfig(**config["llm_config"]),
                embedding_config=EmbeddingConfig(**config["embedding_config"]),
            ),
            actor=client,
        )
        print(f"[OK] Created agent: {agent.id}")

    # Create raw memory WITH agent_state (to generate embeddings)
    print("\n[INFO] Creating raw memory with embeddings...")
    memory_data = RawMemoryItemCreate(
        context="This is a test raw memory for verifying that embeddings are "
        "properly generated and saved to the PostgreSQL database using "
        "Google's Gemini text-embedding-004 model. The embedding should "
        "be a 768-dimensional vector.",
        filter_tags={
            "scope": "CARE",
            "test_type": "real_embeddings",
            "model": "gemini",
        },
        user_id=user.id,
        organization_id=org.id,
    )

    try:
        created_memory = raw_memory_mgr.create_raw_memory(
            raw_memory=memory_data,
            actor=client,
            agent_state=agent,  # Pass agent_state to generate embeddings
            client_id=client.id,
            user_id=user.id,
            use_cache=False,
        )

        print(f"[SUCCESS] Raw memory created successfully!")
        print(f"          ID: {created_memory.id}")
        print(f"          Context: {created_memory.context[:80]}...")
        print(f"          Has embedding: {created_memory.context_embedding is not None}")

        if created_memory.context_embedding:
            print(f"          Embedding dimension: {len(created_memory.context_embedding)}")
            print(f"          Embedding config model: {created_memory.embedding_config.embedding_model}")
            print(f"          First 5 embedding values: {created_memory.context_embedding[:5]}")
            print(f"\n[SUCCESS] Embeddings are saved to the database!")
            print(f"\nYou can verify in PostgreSQL:")
            print(f"   SELECT id, context, embedding_config, ")
            print(f"          array_length(context_embedding, 1) as embedding_dim")
            print(f"   FROM raw_memory WHERE id = '{created_memory.id}';")
        else:
            print(f"\n[WARNING] No embeddings were generated!")
            print(f"          This might be due to:")
            print(f"          - BUILD_EMBEDDINGS_FOR_MEMORY is disabled")
            print(f"          - Google API key is invalid")
            print(f"          - Network connectivity issues")

        # Cleanup (optional - comment out to keep the record)
        # raw_memory_mgr.delete_raw_memory(created_memory.id, client)
        # print(f"\n[INFO] Cleaned up test memory")

    except Exception as e:
        print(f"\n[ERROR] Error creating raw memory with embeddings:")
        print(f"        {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
