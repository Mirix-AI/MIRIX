"""
Example demonstrating the new MirixClient API structure.

This example shows how to:
1. Start a server
2. Create a MirixClient with project support
3. Initialize a meta agent with config
4. Use the new memory endpoints: add, retrieve_with_conversation, retrieve_with_topic, search
"""

import os

from mirix import MirixClient, load_config


def example_basic_usage():
    """Basic example of the new MirixClient API."""
    print("=" * 80)
    print("Example: New MirixClient API")
    print("=" * 80)

    # Set API key via environment variable
    os.environ["MIRIX_API_KEY"] = "your-api-key"

    # Create client with project name
    client = MirixClient(project="test")

    print("\n1. Loading configuration...")
    # Load config from YAML file
    config = load_config("mirix/configs/mirix.yaml")

    print("\n2. Initializing meta agent...")
    # Initialize meta agent with config
    meta_agent = client.initialize_meta_agent(config=config)
    print(f"   Meta agent initialized: {meta_agent.id}")

    print("\n3. Adding conversation to memory...")
    # Add conversation turns (must end with assistant turn)
    result = client.add(
        user_id="user_123",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "I went to have dinner with my wife at Sichuan Chef"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "That sounds great! How was the food?"}]},
        ],
    )
    print(f"   Memory added: {result['success']}")

    print("\n4. Retrieving memories with conversation context...")
    # Retrieve relevant memories (must end with user turn)
    memories = client.retrieve_with_conversation(
        user_id="user_123",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "I want to go to the place I went with my wife again."}],
            }
        ],
    )
    print(f"   Retrieved {len(memories.get('memories', {}).get('episodic', []))} episodic memories")

    print("\n5. Retrieving memories by topic...")
    # Retrieve by topic
    topic_memories = client.retrieve_with_topic(user_id="user_123", topic="dinner")
    print(f"   Topic search completed: {topic_memories['success']}")

    print("\n6. Searching memories...")
    # Search memories
    search_results = client.search(user_id="user_123", query="restaurants", limit=5)
    print(f"   Search completed: {search_results['success']}")
    print(f"   Found {search_results['count']} results")


def example_with_config_dict():
    """Example using inline config dictionary instead of file."""
    print("\n" + "=" * 80)
    print("Example: Using Inline Config")
    print("=" * 80)

    # Create client
    client = MirixClient(project="test", api_key="your-api-key")  # Can also pass API key directly

    # Define config inline
    config = {
        "llm_config": {
            "model": "gemini-2.0-flash",
            "model_endpoint_type": "google_ai",
            "context_window": 1048576,
        },
        "embedding_config": {
            "model": "text-embedding-004",
            "model_endpoint_type": "google_ai",
            "embedding_dim": 768,
        },
    }

    print("\n1. Initializing meta agent with inline config...")
    meta_agent = client.initialize_meta_agent(config=config)
    print(f"   Meta agent initialized: {meta_agent.id}")

    print("\n2. Adding simple conversation...")
    client.add(
        user_id="user_456",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "I'm doing well, thank you!"}]},
        ],
    )
    print("   Conversation added to memory")


def example_multiple_projects():
    """Example showing how to work with multiple projects."""
    print("\n" + "=" * 80)
    print("Example: Multiple Projects")
    print("=" * 80)

    # Create clients for different projects
    client1 = MirixClient(project="project_a")
    client2 = MirixClient(project="project_b")

    config = {"llm_config": {"model": "gemini-2.0-flash"}, "embedding_config": {"model": "text-embedding-004"}}

    print("\n1. Initializing meta agents for different projects...")
    meta_agent_a = client1.initialize_meta_agent(config=config)
    meta_agent_b = client2.initialize_meta_agent(config=config)

    print(f"   Project A meta agent: {meta_agent_a.id}")
    print(f"   Project B meta agent: {meta_agent_b.id}")

    print("\n2. Each project has isolated memory...")
    # Add to project A
    client1.add(
        user_id="user_1",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Data for project A"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Stored in project A"}]},
        ],
    )

    # Add to project B
    client2.add(
        user_id="user_1",
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "Data for project B"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Stored in project B"}]},
        ],
    )
    print("   Projects have separate memory spaces")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "NEW MIRIX CLIENT API EXAMPLES" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\nNote: Make sure the Mirix server is running before executing these examples.")
    print("Start server with: python scripts/start_server.py --reload")
    print()

    try:
        example_basic_usage()
        example_with_config_dict()
        example_multiple_projects()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n\n⚠️  Error: {e}")
        print("\nMake sure:")
        print("1. The Mirix server is running at http://localhost:8000")
        print("2. You have set MIRIX_API_KEY environment variable")
        print("3. The server has the necessary configuration files")
        print("\nTo start the server:")
        print("   python scripts/start_server.py --reload")
