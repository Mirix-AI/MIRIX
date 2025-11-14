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

from mirix.schemas.agent import AgentType
from mirix import MirixClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    
    # Create MirixClient (connects to server via REST API)
    client_id = 'demo-client-app'  # Identifies the client application
    user_id = 'demo-user'  # Identifies the end-user within the client app
    org_id = 'demo-org'
    
    client = MirixClient(
        api_key=None, # TODO: add authentication later
        client_id=client_id,
        org_id=org_id,
        debug=True,
    )

    client.initialize_meta_agent(
        config_path="mirix/configs/examples/mirix_gemini.yaml",
        # config_path="mirix/configs/examples/mirix_openai.yaml",
        update_agents=False,
    )

    # result = client.add(
    #    user_id=user_id,
    #    messages=[
    #        {
    #            "role": "user",
    #            "content": [{
    #                "type": "text",
    #                "text": "I just had a meeting with Sarah from the design team at 2 PM today. We discussed the new UI mockups and she showed me three different color schemes."
    #            }]
    #        },
    #    ],
    #    chaining=False
    #)
    #print(f"[OK] Memory added successfully: {result.get('success', False)}")

    # 4. Example: Retrieve memories using new API
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
                        "text": ""
                    }]
                }
            ],
            limit=10,  # Retrieve up to 10 items per memory type
            filter_tags = filter_tags
        )

        print("[OK] Retrieved memories successfully")
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
    except Exception as e:
        print(f"[ERROR] Error retrieving memories: {e}")
        import traceback
        traceback.print_exc()
    print()

    # 5. Example: Retrieve by topic
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
