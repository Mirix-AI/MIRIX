"""
Quick smoke test: send one conversation chunk and verify memories are created.
"""
import asyncio
import sys
sys.path.insert(0, ".")

from mirix_memory_system import MirixMemorySystem

config_path = "configs/skill_evolve_openrouter.yaml"
user_id = "smoke-required"

print("Creating MirixMemorySystem...")
ms = MirixMemorySystem(user_id=user_id, mirix_config_path=config_path,
                        client_id="smoke-required", org_id="smoke-required")

chunk = """Session 1 (2024-01-15)
User1: I had a wonderful dinner with my friend Sarah last night at Olive Garden.
User2: That sounds great! Did you enjoy the food?
User1: Yes, the pasta was amazing. Sarah also recommended a book called 'Atomic Habits' by James Clear.
User2: I've heard great things about that book!
User1: I'm planning to start reading it this weekend. Also, I just adopted a golden retriever puppy named Max!"""

print(f"Adding chunk ({len(chunk)} chars)...")
response = ms.add_chunk(chunk)
print(f"add_chunk response: {response}")

print("\nSearching memories to verify they were created...")
for mem_type in ["episodic", "semantic", "knowledge", "all"]:
    for method in ["embedding", "bm25"]:
        results = asyncio.run(ms.client.search(
            user_id=user_id,
            query="Sarah dinner book puppy",
            memory_type=mem_type,
            search_method=method,
            limit=10,
        ))
        if results.get("success"):
            count = len(results.get("results", []))
            print(f"  {mem_type}/{method}: {count} results")
            for r in results.get("results", [])[:3]:
                summary = r.get("summary") or r.get("description") or r.get("name") or r.get("value", "")
                print(f"    - {str(summary)[:120]}")
        else:
            print(f"  {mem_type}/{method}: search failed - {results.get('error', 'unknown')}")

print("\nDone!")
