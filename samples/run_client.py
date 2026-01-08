#!/usr/bin/env python3
"""
Simple Mirix MirixClient script to connect to a remote server,
set up meta agent, and demonstrate memory operations (add and retrieve).

Prerequisites:
- Start the server first: python scripts/start_server.py --reload
- Or set MIRIX_API_URL environment variable to your server URL
"""

import logging
import os
from pathlib import Path

from mirix import MirixClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# stay in the source code root directory
# Before running the script, run the following command:
# python scripts/start_server.py
# python samples/generate_demo_api_key.py
#      The above command will output the api key, which we use "your_api_key_here" to denote
# export MIRIX_API_KEY=your_api_key_here # for windows it should be: $env:MIRIX_API_KEY = "sk-your-key-here"
# then run python samples/run_client.py

def print_memories(memories):
    """Print retrieved memories in a formatted way.
    
    Args:
        memories: Dictionary containing retrieved memories from retrieve_with_conversation
    """
    # Debug: Show temporal filtering info
    if memories.get("temporal_expression"):
        print(f"  🕐 Temporal expression detected: '{memories.get('temporal_expression')}'")
    if memories.get("date_range"):
        date_range = memories.get("date_range")
        print(f"  📅 Date range applied:")
        print(f"     Start: {date_range.get('start')}")
        print(f"     End: {date_range.get('end')}")
    
    if memories.get("memories"):
        for memory_type, data in memories["memories"].items():
            if data and data.get("total_count", 0) > 0:
                print(f"\n  {'=' * 60}")
                print(f"  {memory_type.upper()} MEMORY (Total: {data['total_count']})")
                print(f"  {'=' * 60}")
                
                # Debug: Check what keys are in the data
                items = data.get("items", [])
                
                # Episodic memory uses 'recent' and 'relevant' keys instead of 'items'
                if not items and memory_type == "episodic":
                    # Combine both recent and relevant episodic memories
                    recent = data.get("recent", [])
                    relevant = data.get("relevant", [])
                    # Merge and deduplicate by ID
                    seen_ids = set()
                    items = []
                    for item in recent + relevant:
                        if item.get('id') not in seen_ids:
                            items.append(item)
                            seen_ids.add(item.get('id'))
                
                # Some other memory types may also use 'recent' as fallback
                if not items and "recent" in data:
                    items = data.get("recent", [])
                
                # Debug: If still no items but total_count > 0, log the issue
                if not items and data.get("total_count", 0) > 0:
                    print(f"  ⚠️  DEBUG: No items found despite total_count={data['total_count']}")
                    print(f"      Available keys in data: {list(data.keys())}")
                    if data.keys():
                        for key in data.keys():
                            if key not in ['total_count', 'items', 'recent']:
                                print(f"      {key}: {type(data[key])} (length: {len(data[key]) if isinstance(data[key], (list, dict)) else 'N/A'})")
                    print()
                
                if memory_type == "core":
                    # Core memory: blocks with label and value
                    for i, item in enumerate(items, 1):
                        print(f"  [{i}] {item.get('label', 'N/A')}: {item.get('value', 'N/A')}")
                    print()
                
                elif memory_type == "episodic":
                    # Episodic memory: event with full details
                    for i, item in enumerate(items, 1):
                        summary = item.get('summary', 'N/A')
                        # API returns 'timestamp' not 'occurred_at'
                        occurred = item.get('timestamp', item.get('occurred_at', 'N/A'))
                        event_type = item.get('event_type', 'N/A')
                        details = item.get('details', '')
                        actor = item.get('actor', 'N/A')
                        
                        print(f"  [{i}] {summary}")
                        print(f"      Time: {occurred}")
                        if event_type != 'N/A':
                            print(f"      Type: {event_type}")
                        if actor != 'N/A':
                            print(f"      Actor: {actor}")
                        if details:
                            print(f"      Details: {details}")
                        print()
                
                elif memory_type == "procedural":
                    # Procedural memory: procedure with summary (API doesn't return full details)
                    for i, item in enumerate(items, 1):
                        # API returns 'summary' not 'description'
                        summary = item.get('summary', item.get('description', 'N/A'))
                        entry_type = item.get('entry_type', 'N/A')
                        # API doesn't return 'steps' in retrieve endpoint (only id, entry_type, summary)
                        steps = item.get('steps', [])
                        
                        print(f"  [{i}] [{entry_type}] {summary}")
                        if steps:
                            print(f"      Steps ({len(steps)}):")
                            for step_num, step in enumerate(steps, 1):
                                print(f"        {step_num}. {step}")
                        else:
                            print("      Steps: Not included in response (use search API for full details)")
                        print()
                
                elif memory_type == "semantic":
                    # Semantic memory: concept with name and summary
                    for i, item in enumerate(items, 1):
                        name = item.get('name', 'N/A')
                        summary = item.get('summary', 'N/A')
                        source = item.get('source', 'N/A')
                        details = item.get('details', '')
                        
                        print(f"  [{i}] {name}")
                        print(f"      Summary: {summary}")
                        if details:
                            print(f"      Details: {details}")
                        print(f"      Source: {source}")
                        print()
                
                elif memory_type == "resource":
                    # Resource memory: document summary (API doesn't return content in retrieve endpoint)
                    for i, item in enumerate(items, 1):
                        title = item.get('title', 'N/A')
                        res_type = item.get('resource_type', 'N/A')
                        summary = item.get('summary', 'N/A')
                        # API doesn't return 'content' in retrieve endpoint (only id, title, summary, resource_type)
                        content = item.get('content', '')
                        
                        print(f"  [{i}] [{res_type}] {title}")
                        print(f"      Summary: {summary}")
                        
                        if content:
                            content_len = len(content)
                            print(f"      Content Length: {content_len} characters")
                            preview = content[:200].replace('\n', ' ')
                            if len(content) > 200:
                                preview += "..."
                            print(f"      Preview: {preview}")
                        else:
                            print("      Content: Not included in response (use search API for full details)")
                        print()
                
                elif memory_type == "knowledge_vault":
                    # Knowledge vault: API only returns id and caption in retrieve endpoint
                    for i, item in enumerate(items, 1):
                        caption = item.get('caption', 'N/A')
                        # API doesn't return entry_type, source, sensitivity in retrieve endpoint
                        entry_type = item.get('entry_type', 'N/A')
                        source = item.get('source', 'N/A')
                        sensitivity = item.get('sensitivity', 'N/A')
                        
                        print(f"  [{i}] {caption}")
                        if entry_type != 'N/A':
                            print(f"      Type: {entry_type}")
                        if source != 'N/A':
                            print(f"      Source: {source}")
                        if sensitivity != 'N/A':
                            print(f"      Sensitivity: {sensitivity}")
                        # Don't print secret_value for security
                        print("      Secret: [REDACTED for security]")
                        print()
                
                else:
                    # Fallback for unknown types
                    for i, item in enumerate(items, 1):
                        print(f"  [{i}] {item}")
        else:
            print("  No memories found yet (may need more time to process)")


def convert_search_results_to_memory_format(search_results):
    """Convert search API results to retrieve_with_conversation format.
    
    Args:
        search_results: Dictionary from client.search() with flat 'results' list
        
    Returns:
        Dictionary in the format expected by print_memories()
    """
    # Initialize the structure
    converted = {
        "memories": {}
    }
    
    # Get the flat results list
    results = search_results.get("results", [])
    
    if not results:
        return converted
    
    # Group results by memory_type
    memory_types = {}
    for result in results:
        mem_type = result.get("memory_type", "unknown")
        if mem_type not in memory_types:
            memory_types[mem_type] = []
        memory_types[mem_type].append(result)
    
    # Convert to the nested format
    for mem_type, items in memory_types.items():
        converted["memories"][mem_type] = {
            "total_count": len(items),
            "items": items
        }
    
    return converted


def test_retrieve_with_conversation(client, user_id):
    """Test the retrieve_with_conversation API.
    
    Args:
        client: MirixClient instance
        user_id: User ID for the retrieval
    """
    print("Step 4: Retrieving memories with conversation context...")
    print("-" * 70)
    try:
        filter_tags = {} #{"expert_id": "expert-234"}

        memories = client.retrieve_with_conversation(
            user_id=user_id,
            messages=[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "What did we discuss on QuickBooks in last 4 days?"
                    }]
                }
            ],
            limit=10,  # Retrieve up to 10 items per memory type
            filter_tags = filter_tags
        )

        print("[OK] Retrieved memories successfully")
        print_memories(memories)
    except Exception as e:
        print(f"[ERROR] Error retrieving memories: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_search(client, user_id):
    """Test the search API.
    
    Args:
        client: MirixClient instance
        user_id: User ID for the search
    """
    print("Step 5: Searching memories with BM25...")
    print("-" * 70)
    try:
        # Example 1: Search all memory types
        results = client.search(
            user_id=user_id,
            query="support",
            memory_type="all",
            search_method="bm25",
            #similarity_threshold=0.50,
            limit=30
        )

        print(f"[OK] Search completed: {results.get('success', False)}")
        print(f"  Query: '{results.get('query', 'N/A')}'")
        print(f"  Memory Type: {results.get('memory_type', 'N/A')}")
        print(f"  Search Method: {results.get('search_method', 'N/A')}")
        print(f"  Total Results: {results.get('count', 0)}")
        print()
        
        # Convert search results to memory format and use print_memories
        if results.get("count", 0) > 0:
            converted_memories = convert_search_results_to_memory_format(results)
            print_memories(converted_memories)
        else:
            print("  No results found")
            
    except Exception as e:
        print(f"[ERROR] Error searching memories: {e}")
        import traceback
        traceback.print_exc()
    print()


def test_search_all_users(client):
    """Test the search_all_users API to search across all users in the organization.
    
    Args:
        client: MirixClient instance
        client_id: Client ID for organization and scope filtering
    """
    print("Step 6: Searching memories across ALL users in organization with BM25...")
    print("-" * 70)
    try:
        # Search all memory types across all users
        results = client.search_all_users(
            query="music",
            memory_type="all",
            search_method="embedding",
            limit=20,  # Total results across all users
            client_id=client.client_id,
            similarity_threshold=0.25
        )

        print(f"[OK] Cross-user search completed: {results.get('success', False)}")
        print(f"  Query: '{results.get('query', 'N/A')}'")
        print(f"  Memory Type: {results.get('memory_type', 'N/A')}")
        print(f"  Search Method: {results.get('search_method', 'N/A')}")
        print(f"  Organization: {results.get('organization_id', 'N/A')}")
        print(f"  Client Scope: {results.get('client_scope', 'N/A')}")
        print(f"  Total Results: {results.get('count', 0)}")
        print()
        
        # Convert search results to memory format and use print_memories
        if results.get("count", 0) > 0:
            converted_memories = convert_search_results_to_memory_format(results)
            
            # Print header with user context
            print("  🔍 Results from multiple users (filtered by client scope):")
            print()
            
            print_memories(converted_memories)
        else:
            print("  No results found across all users")
            
    except Exception as e:
        print(f"[ERROR] Error searching memories across all users: {e}")
        import traceback
        traceback.print_exc()
    print()


def main():
    
    # Resolve config path relative to project root
    # Get the directory where this script is located (samples/)
    script_dir = Path(__file__).parent
    # Navigate to project root (parent of samples/)
    project_root = script_dir.parent
    # Build path to config file
    config_path = project_root / "mirix" / "configs" / "examples" / "mirix_gemini.yaml"
    
    # Verify the config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Create MirixClient (connects to server via REST API)
    client_id = 'sales-loader-client' #'demo-client-app'  # Identifies the client application
    user_id = 'caring cathy' #'demo-user'  # Identifies the end-user within the client app
    org_id = 'demo-org'
    
    client = MirixClient(
        #api_key=api_key,
        client_id="sales-loader-client",
        client_scope="Sales",
        org_id="demo-org",
        debug=True,
    )

    client.initialize_meta_agent(
        config_path=str(config_path),
        # Alternative: config_path=str(project_root / "mirix/configs/examples/mirix_openai.yaml"),
        update_agents=False,
    )
    
    """
    result = client.add(
       user_id=user_id,  # Optional - uses admin user if None
       messages=[
           {
               "role": "user",
               "content": [{
                   "type": "text",
                   "text": (
                       "Hi! My name is David, and I'm a senior software engineer at TechCorp. "
                       "I prefer Python over JavaScript, and my favorite IDE is VS Code. "
                       "Yesterday, I attended the quarterly planning meeting where we discussed the new AI features roadmap. "
                       "Last week, I completed the database migration project successfully. "
                       "I've learned that microservices architecture requires careful API design and that "
                       "distributed tracing is essential for debugging complex systems. "
                       "I reviewed the Q4 Performance Report and the System Architecture Documentation from our wiki. "
                       "For deploying our application, the process is: first run the test suite, then build the Docker image, "
                       "push to the registry, and finally apply the Kubernetes manifests in staging before production. "
                       "My production database password is db_prod_2024! and the API key for our payment gateway is sk-live-abc123xyz."
                       "Please update all memories to reflect my latest activities and preferences."
                   )
               }]
           },
       ],
       chaining=False
    )
    print(f"[OK] Memory added successfully: {result.get('success', False)}")
    """

    # 4. Example: Retrieve memories using new API
    test_retrieve_with_conversation(client, user_id)

    # 5. Example: Search memories using BM25
    #test_search(client, user_id)

    # 6. Example: Search across all users in organization
    #test_search_all_users(client)

    # 7. Example: Retrieve by topic
    # print("Step 5: Retrieving by topic...")
    # print("-" * 70)
    # try:
    #    topic_memories = client.retrieve_with_topic(
    #        user_id=user_id,
    #        topic="design",
    #        limit=5  # Retrieve up to 5 items per memory type
    #    )
    #    print(f"[OK] Topic retrieval completed: {topic_memories.get('success', False)}")
    #    if topic_memories.get("memories"):
    #        print(f"  Topics searched: {topic_memories.get('topic')}")
    #        for memory_type, items in topic_memories["memories"].items():
    #            if items and items.get('total_count', 0) > 0:
    #                print(f"  {memory_type.upper()}: {items['total_count']} total, {len(items.get('items', items.get('recent', [])))} retrieved")
    #    except Exception as e:
    #        print(f"[ERROR] Error retrieving by topic: {e}")
    #        import traceback
    #        traceback.print_exc()
    #    print()

    # 6. Example: Search memories - returns flat list of results
    # print("Step 6: Searching memories...")
    # print("-" * 70)
    # try:
    #    # Example 1: Search all memory types
    #    results = client.search(
    #        user_id=user_id,
    #        query="meeting design",
    #        memory_type="all",
    #        limit=5
    #    )
    #    print(f"[OK] Search completed: {results.get('success', False)}")
    #    print(f"  Found {results.get('count', 0)} total results across all memory types")
        
        # Display sample results
    #    if results.get("results"):
    #        print(f"  Sample results:")
    #        for i, item in enumerate(results["results"][:3], 1):
    #            mem_type = item.get("memory_type", "unknown").upper()
    #            summary = item.get("summary", item.get("caption", item.get("name", "N/A")))
    #            print(f"    {i}. [{mem_type}] {summary[:60]}...")
        
        # Example 2: Search only episodic memories
    #    print("\n  Searching only episodic memories in details field...")
    #    episodic_results = client.search(
    #        user_id=user_id,
    #        query="Sarah",
    #        memory_type="episodic",
    #        search_field="details",
    #        search_method='bm25',
    #        limit=5
    #    )
    #    print(f"  Found {episodic_results.get('count', 0)} episodic results")
    #    print()

    print("=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
