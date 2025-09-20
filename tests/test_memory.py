"""
Comprehensive Memory System Testing Script

This script tests all memory types in the Mirix system and demonstrates different search methods.

Features:
1. Tests all 5 memory types: Episodic, Procedural, Resource, Knowledge Vault, and Semantic
2. For each memory type, it:
   - Sends a test message to the memory agent
   - Performs a search using the list_* method with different search methods
3. Demonstrates 4 different search methods:
   - bm25: Full-text search using database capabilities
   - embedding: Vector similarity search using embeddings
   - string_match: Simple string containment search
4. Lists all content currently stored in each memory type
5. Tests different search fields for each memory type
6. Tests text-only memorization using both normal and immediate save methods:
   - Normal memorization: Accumulates messages until temporary_message_limit is reached
   - Immediate save: Bypasses accumulation and saves content immediately to memory
7. Tests batch immediate memory save functionality for processing multiple items at once
8. Includes performance comparison between normal and immediate save methods

Available search fields by memory type:
- Episodic: summary, details
- Procedural: description, steps  
- Resource: summary, content
- Knowledge Vault: description, secret_value
- Semantic: name, summary, details

New Test Functions:
- test_text_only_memorization(): Tests both normal and immediate text memorization
- test_immediate_memory_functionality(): Comprehensive testing of immediate save features

Usage:
    python test_memory.py

The script will run all tests automatically and show results for each memory type and search method.
"""

import os
import sys

# Add the project root to Python path so we can import mirix
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

# Now import the rest
from mirix.agent import AgentWrapper
from datetime import datetime
import traceback


class TestTracker:
    """Class to track test results and provide summary reporting"""
    
    def __init__(self):
        self.tests = []
        self.current_test = None
        
    def start_test(self, test_name, description=""):
        """Start tracking a new test"""
        self.current_test = {
            'name': test_name,
            'description': description,
            'status': 'running',
            'error': None,
            'subtests': []
        }
        print(f"\n🚀 Starting: {test_name}")
        if description:
            print(f"   Description: {description}")
    
    def start_subtest(self, subtest_name):
        """Start tracking a subtest within the current test"""
        if not self.current_test:
            print("Warning: No current test to add subtest to")
            return
            
        subtest = {
            'name': subtest_name,
            'status': 'running',
            'error': None
        }
        self.current_test['subtests'].append(subtest)
        print(f"  ▶️ {subtest_name}")
        return len(self.current_test['subtests']) - 1  # Return index for reference
    
    def pass_subtest(self, subtest_index=None, message=""):
        """Mark the current or specified subtest as passed"""
        if not self.current_test:
            return
            
        if subtest_index is None:
            subtest_index = len(self.current_test['subtests']) - 1
            
        if 0 <= subtest_index < len(self.current_test['subtests']):
            self.current_test['subtests'][subtest_index]['status'] = 'passed'
            subtest_name = self.current_test['subtests'][subtest_index]['name']
            print(f"  ✅ {subtest_name}" + (f" - {message}" if message else ""))
    
    def fail_subtest(self, error, subtest_index=None):
        """Mark the current or specified subtest as failed"""
        if not self.current_test:
            return
            
        if subtest_index is None:
            subtest_index = len(self.current_test['subtests']) - 1
            
        if 0 <= subtest_index < len(self.current_test['subtests']):
            self.current_test['subtests'][subtest_index]['status'] = 'failed'
            self.current_test['subtests'][subtest_index]['error'] = str(error)
            subtest_name = self.current_test['subtests'][subtest_index]['name']
            print(f"  ❌ {subtest_name} - ERROR: {error}")
    
    def pass_test(self, message=""):
        """Mark the current test as passed"""
        if not self.current_test:
            return
            
        self.current_test['status'] = 'passed'
        print(f"✅ PASSED: {self.current_test['name']}" + (f" - {message}" if message else ""))
        self.tests.append(self.current_test)
        self.current_test = None
    
    def fail_test(self, error):
        """Mark the current test as failed"""
        if not self.current_test:
            return
            
        self.current_test['status'] = 'failed'
        self.current_test['error'] = str(error)
        print(f"❌ FAILED: {self.current_test['name']} - ERROR: {error}")
        self.tests.append(self.current_test)
        self.current_test = None
    
    def get_summary(self):
        """Get a summary of all test results"""
        total_tests = len(self.tests)
        passed_tests = len([t for t in self.tests if t['status'] == 'passed'])
        failed_tests = len([t for t in self.tests if t['status'] == 'failed'])
        
        total_subtests = sum(len(t['subtests']) for t in self.tests)
        passed_subtests = sum(len([s for s in t['subtests'] if s['status'] == 'passed']) for t in self.tests)
        failed_subtests = sum(len([s for s in t['subtests'] if s['status'] == 'failed']) for t in self.tests)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'total_subtests': total_subtests,
            'passed_subtests': passed_subtests,
            'failed_subtests': failed_subtests,
            'tests': self.tests
        }
    
    def print_summary(self):
        """Print a detailed summary of all test results"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("🏁 TEST EXECUTION SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   ✅ Passed Tests: {summary['passed_tests']}")
        if summary['failed_tests'] > 0:
            print(f"   ❌ Failed Tests: {summary['failed_tests']}")
        print(f"   📈 Success Rate: {(summary['passed_tests']/summary['total_tests']*100):.1f}%" if summary['total_tests'] > 0 else "   📈 Success Rate: N/A")
        
        if summary['total_subtests'] > 0:
            print(f"\n🔍 SUBTEST DETAILS:")
            print(f"   Total Subtests: {summary['total_subtests']}")
            print(f"   ✅ Passed Subtests: {summary['passed_subtests']}")
            if summary['failed_subtests'] > 0:
                print(f"   ❌ Failed Subtests: {summary['failed_subtests']}")
            print(f"   📈 Subtest Success Rate: {(summary['passed_subtests']/summary['total_subtests']*100):.1f}%")
        
        # Show failed tests details
        failed_tests = [t for t in summary['tests'] if t['status'] == 'failed']
        if failed_tests:
            print(f"\n❌ FAILED TESTS DETAILS:")
            for i, test in enumerate(failed_tests, 1):
                print(f"   {i}. {test['name']}")
                print(f"      Error: {test['error']}")
                
                # Show failed subtests
                failed_subtests = [s for s in test['subtests'] if s['status'] == 'failed']
                if failed_subtests:
                    print(f"      Failed Subtests:")
                    for subtest in failed_subtests:
                        print(f"        - {subtest['name']}: {subtest['error']}")
        
        # Show passed tests summary
        passed_tests = [t for t in summary['tests'] if t['status'] == 'passed']
        if passed_tests:
            print(f"\n✅ PASSED TESTS:")
            for i, test in enumerate(passed_tests, 1):
                subtest_count = len(test['subtests'])
                passed_subtest_count = len([s for s in test['subtests'] if s['status'] == 'passed'])
                print(f"   {i}. {test['name']} ({passed_subtest_count}/{subtest_count} subtests passed)")
        
        print("\n" + "="*80)
        
        return summary


# Create global test tracker
test_tracker = TestTracker()

def run_tracked_test(test_name, test_description, test_function, *args, **kwargs):
    """
    Utility function to run a test with automatic tracking
    
    Args:
        test_name: Name of the test
        test_description: Description of what the test does
        test_function: The test function to run
        *args, **kwargs: Arguments to pass to the test function
    """
    test_tracker.start_test(test_name, test_description)
    
    try:
        result = test_function(*args, **kwargs)
        test_tracker.pass_test("Test completed successfully")
        return result
    except Exception as e:
        test_tracker.fail_test(f"Test failed: {e}")
        traceback.print_exc()
        return None

def test_search_methods(agent):
    """Test different search methods across all memory types"""
    test_tracker.start_test("Search Methods Test", "Testing different search methods across all memory types")
    
    try:
        # Define search methods to test
        search_methods = ['bm25', 'embedding', 'string_match']
        
        # Test queries for each memory type
        test_queries = {
            'episodic': 'grocery',
            'procedural': 'pizza', 
            'resource': 'python',
            'knowledge_vault': 'password',
            'semantic': 'machine learning'
        }
        
        # Test search fields for each memory type
        search_fields = {
            'episodic': ['summary', 'details'],
            'procedural': ['summary', 'steps'],
            'resource': ['summary', 'content'],
            'knowledge_vault': ['caption', 'secret_value'],
            'semantic': ['name', 'summary', 'details']
        }
        
        # Test Episodic Memory Search Methods
        subtest_idx = test_tracker.start_subtest("Testing Episodic Memory Search Methods")
        try:
            for method in search_methods:
                for field in search_fields['episodic']:
                    try:
                        results = agent.client.server.episodic_memory_manager.list_episodic_memory(
                            agent_state=agent.agent_states.episodic_memory_agent_state,
                            actor=agent.client.user,
                            query=test_queries['episodic'],
                            search_method=method,
                            search_field=field,
                            limit=5
                        )
                        print(f"  {method} on {field}: {len(results)} results")
                    except Exception as e:
                        print(f"  {method} on {field}: Error - {e}")
                        raise e
            test_tracker.pass_subtest(subtest_idx, "All episodic search methods completed")
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
        
        # Test Procedural Memory Search Methods
        subtest_idx = test_tracker.start_subtest("Testing Procedural Memory Search Methods")
        try:
            for method in search_methods:
                for field in search_fields['procedural']:
                    try:
                        results = agent.client.server.procedural_memory_manager.list_procedures(
                            agent_state=agent.agent_states.procedural_memory_agent_state,
                            actor=agent.client.user,
                            query=test_queries['procedural'],
                            search_method=method,
                            search_field=field,
                            limit=5
                        )
                        print(f"  {method} on {field}: {len(results)} results")
                    except Exception as e:
                        print(f"  {method} on {field}: Error - {e}")
                        raise e
            test_tracker.pass_subtest(subtest_idx, "All procedural search methods completed")
        except Exception as e:
            import ipdb; ipdb.set_trace()
            test_tracker.fail_subtest(e, subtest_idx)
        
        # Test Resource Memory Search Methods
        subtest_idx = test_tracker.start_subtest("Testing Resource Memory Search Methods")
        try:
            for method in search_methods:
                for field in search_fields['resource']:
                    # Skip embedding search on content field (no embeddings)
                    if method == 'embedding' and field == 'content':
                        print(f"  {method} on {field}: Skipped (no embeddings for this field)")
                        continue
                    try:
                        results = agent.client.server.resource_memory_manager.list_resources(
                            agent_state=agent.agent_states.resource_memory_agent_state,
                            actor=agent.client.user,
                            query=test_queries['resource'],
                            search_method=method,
                            search_field=field,
                            limit=5
                        )
                        print(f"  {method} on {field}: {len(results)} results")
                    except Exception as e:
                        print(f"  {method} on {field}: Error - {e}")
                        raise e
            test_tracker.pass_subtest(subtest_idx, "All resource search methods completed")
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
        
        # Test Knowledge Vault Search Methods
        subtest_idx = test_tracker.start_subtest("Testing Knowledge Vault Search Methods")
        try:
            for method in search_methods:
                for field in search_fields['knowledge_vault']:
                    # Skip embedding search on secret_value field (no embeddings)
                    if method == 'embedding' and field == 'secret_value':
                        print(f"  {method} on {field}: Skipped (no embeddings for this field)")
                        continue
                    try:
                        results = agent.client.server.knowledge_vault_manager.list_knowledge(
                            agent_state=agent.agent_states.knowledge_vault_agent_state,
                            actor=agent.client.user,
                            query=test_queries['knowledge_vault'],
                            search_method=method,
                            search_field=field,
                            limit=5
                        )
                        print(f"  {method} on {field}: {len(results)} results")
                    except Exception as e:
                        print(f"  {method} on {field}: Error - {e}")
                        raise e
            test_tracker.pass_subtest(subtest_idx, "All knowledge vault search methods completed")
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
        
        # Test Semantic Memory Search Methods
        subtest_idx = test_tracker.start_subtest("Testing Semantic Memory Search Methods")
        try:
            for method in search_methods:
                for field in search_fields['semantic']:
                    try:
                        results = agent.client.server.semantic_memory_manager.list_semantic_items(
                            agent_state=agent.agent_states.semantic_memory_agent_state,
                            actor=agent.client.user,
                            query=test_queries['semantic'],
                            search_method=method,
                            search_field=field,
                            limit=5
                        )
                        print(f"  {method} on {field}: {len(results)} results")
                    except Exception as e:
                        print(f"  {method} on {field}: Error - {e}")
                        raise e
            test_tracker.pass_subtest(subtest_idx, "All semantic search methods completed")
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
        
        test_tracker.pass_test("All search methods tested successfully")
        
    except Exception as e:
        test_tracker.fail_test(f"Search methods test failed: {e}")
        traceback.print_exc()

def test_fts5_comprehensive(agent):
    """Comprehensive FTS5 testing across all memory types"""
    print("=== Comprehensive FTS5 Testing ===")
    
    # Test various FTS5 features across all memory types
    test_cases = [
        {
            'name': 'Single word search',
            'query': 'python',
            'description': 'Test single word matching'
        },
        {
            'name': 'Multi-word OR search',
            'query': 'python programming',
            'description': 'Test multi-word OR matching (default behavior)'
        },
        {
            'name': 'Phrase search',
            'query': '"machine learning"',
            'description': 'Test exact phrase matching'
        },
        {
            'name': 'Complex multi-word',
            'query': 'artificial intelligence algorithms',
            'description': 'Test complex multi-word OR search'
        }
    ]
    
    memory_types = [
        ('episodic', agent.client.server.episodic_memory_manager.list_episodic_memory, 
         agent.agent_states.episodic_memory_agent_state, ['summary', 'details']),
        ('procedural', agent.client.server.procedural_memory_manager.list_procedures, 
         agent.agent_states.procedural_memory_agent_state, ['summary', 'steps']),
        ('resource', agent.client.server.resource_memory_manager.list_resources, 
         agent.agent_states.resource_memory_agent_state, ['title', 'content']),
        ('knowledge_vault', agent.client.server.knowledge_vault_manager.list_knowledge, 
         agent.agent_states.knowledge_vault_agent_state, ['caption', 'secret_value']),
        ('semantic', agent.client.server.semantic_memory_manager.list_semantic_items, 
         agent.agent_states.semantic_memory_agent_state, ['name', 'summary'])
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']}: {test_case['description']} ---")
        print(f"Query: '{test_case['query']}'")
        
        for memory_type, manager_method, agent_state, fields in memory_types:
            print(f"\n{memory_type.title()} Memory:")
            
            # Test across all fields for this memory type
            for field in fields:
                try:
                    results = manager_method(
                        agent_state=agent_state,
                        actor=agent.client.user,
                        query=test_case['query'],
                        search_method='bm25',
                        search_field=field,
                        limit=5
                    )
                    print(f"  {field}: {len(results)} results")
                except Exception as e:
                    print(f"  {field}: Error - {str(e)[:50]}...")
            
            # Test search across all fields
            try:
                results = manager_method(
                    agent_state=agent_state,
                    actor=agent.client.user,
                    query=test_case['query'],
                    search_method='bm25',
                    limit=5
                )
                print(f"  all_fields: {len(results)} results")
            except Exception as e:
                print(f"  all_fields: Error - {str(e)[:50]}...")
    
    print("\nComprehensive FTS5 testing completed.\n")

def test_fts5_performance_comparison(agent):
    """Compare FTS5 performance against other search methods"""
    print("=== FTS5 Performance Comparison ===")
    
    import time
    
    # Test queries for performance comparison
    test_queries = [
        ('simple', 'python'),
        ('medium', 'machine learning'),
        ('complex', 'artificial intelligence programming')
    ]
    
    search_methods = ['bm25', 'string_match', 'fuzzy_match']
    
    # Test on episodic memory (typically has the most data)
    print("Testing on Episodic Memory:")
    
    for query_type, query in test_queries:
        print(f"\n--- {query_type.title()} Query: '{query}' ---")
        
        results_summary = {}
        
        for method in search_methods:
            try:
                start_time = time.time()
                
                if method == 'bm25':
                    results = agent.client.server.episodic_memory_manager.list_episodic_memory(
                        agent_state=agent.agent_states.episodic_memory_agent_state,
                        actor=agent.client.user,
                        query=query,
                        search_method=method,
                        limit=50
                    )
                elif method == 'string_match':
                    results = agent.client.server.episodic_memory_manager.list_episodic_memory(
                        agent_state=agent.agent_states.episodic_memory_agent_state,
                        actor=agent.client.user,
                        query=query,
                        search_method=method,
                        search_field='summary',
                        limit=50
                    )
                elif method == 'fuzzy_match':
                    results = agent.client.server.episodic_memory_manager.list_episodic_memory(
                        agent_state=agent.agent_states.episodic_memory_agent_state,
                        actor=agent.client.user,
                        query=query,
                        search_method=method,
                        search_field='summary',
                        limit=50
                    )
                
                elapsed_time = time.time() - start_time
                results_summary[method] = {
                    'time': elapsed_time,
                    'count': len(results),
                    'success': True
                }
                
                print(f"  {method}: {len(results)} results in {elapsed_time:.4f}s")
                
            except Exception as e:
                results_summary[method] = {
                    'time': None,
                    'count': 0,
                    'success': False,
                    'error': str(e)[:50]
                }
                print(f"  {method}: Error - {str(e)[:50]}...")
        
        # Calculate performance improvements
        if results_summary.get('bm25', {}).get('success') and results_summary.get('string_match', {}).get('success'):
            fts5_time = results_summary['bm25']['time']
            string_time = results_summary['string_match']['time']
            if fts5_time > 0:
                improvement = string_time / fts5_time
                print(f"  FTS5 is {improvement:.1f}x faster than string search")
        
        if results_summary.get('bm25', {}).get('success') and results_summary.get('fuzzy_match', {}).get('success'):
            fts5_time = results_summary['bm25']['time']
            fuzzy_time = results_summary['fuzzy_match']['time']
            if fts5_time > 0:
                improvement = fuzzy_time / fts5_time
                print(f"  FTS5 is {improvement:.1f}x faster than fuzzy search")
    
    print("\nFTS5 performance comparison completed.\n")

def test_fts5_advanced_features(agent):
    """Test advanced FTS5 features like field-specific search and query syntax"""
    print("=== Advanced FTS5 Features Testing ===")
    
    # Test field-specific searches
    print("\n--- Field-Specific Search Testing ---")
    
    # Test on episodic memory with field-specific queries
    field_tests = [
        ('summary', 'grocery shopping'),
        ('details', 'apples bananas'),
        ('actor', 'user'),
        ('event_type', 'shopping')
    ]
    
    for field, query in field_tests:
        print(f"\nTesting field-specific search on '{field}' with query '{query}':")
        try:
            results = agent.client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=agent.agent_states.episodic_memory_agent_state,
                actor=agent.client.user,
                query=query,
                search_method='bm25',
                search_field=field,
                limit=10
            )
            print(f"  Found {len(results)} results")
            
            # Show sample results
            for i, result in enumerate(results[:2]):
                field_value = getattr(result, field, 'N/A')
                print(f"    {i+1}. {field}: {str(field_value)[:50]}...")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test query syntax variations
    print("\n--- Query Syntax Testing ---")
    
    syntax_tests = [
        ('Standard search', 'machine learning'),
        ('Phrase search', '"machine learning"'),
        ('OR search explicit', 'machine OR learning'),
        ('Multiple terms', 'artificial intelligence programming'),
        ('Single character', 'a'),
        ('Numbers', '12345'),
        ('Special characters', 'api_key'),
    ]
    
    for test_name, query in syntax_tests:
        print(f"\n{test_name} - Query: '{query}':")
        try:
            # Test across different memory types
            memory_tests = [
                ('Episodic', agent.client.server.episodic_memory_manager.list_episodic_memory, 
                 agent.agent_states.episodic_memory_agent_state),
                ('Semantic', agent.client.server.semantic_memory_manager.list_semantic_items, 
                 agent.agent_states.semantic_memory_agent_state),
                ('Knowledge Vault', agent.client.server.knowledge_vault_manager.list_knowledge, 
                 agent.agent_states.knowledge_vault_agent_state)
            ]
            
            for memory_name, method, agent_state in memory_tests:
                try:
                    results = method(
                        agent_state=agent_state,
                        actor=agent.client.user,
                        query=query,
                        search_method='bm25',
                        limit=5
                    )
                    print(f"  {memory_name}: {len(results)} results")
                except Exception as e:
                    print(f"  {memory_name}: Error - {str(e)[:30]}...")
                    
        except Exception as e:
            print(f"  General error: {e}")
    
    # Test empty and edge cases
    print("\n--- Edge Cases Testing ---")
    
    edge_cases = [
        ('Empty query', ''),
        ('Whitespace only', '   '),
        ('Very long query', 'machine learning artificial intelligence deep learning neural networks' * 5),
        ('Special characters only', '!@#$%^&*()'),
    ]
    
    for test_name, query in edge_cases:
        print(f"\n{test_name} - Query: '{query[:30]}{'...' if len(query) > 30 else ''}':")
        try:
            results = agent.client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=agent.agent_states.episodic_memory_agent_state,
                actor=agent.client.user,
                query=query,
                search_method='bm25',
                limit=5
            )
            print(f"  Results: {len(results)}")
        except Exception as e:
            print(f"  Error: {str(e)[:50]}...")
    
    print("\nAdvanced FTS5 features testing completed.\n")

def list_all_memory_content(agent):
    """List all content in each memory type (no search query)"""
    print("=== Listing All Memory Content ===")
    
    print("\n--- All Episodic Events ---")
    try:
        episodic_memory = agent.client.server.episodic_memory_manager.list_episodic_memory(
            agent_state=agent.agent_states.episodic_memory_agent_state,
            actor=agent.client.user,
            query='',  # Empty query to get all
            limit=50
        )
        print(f"Total episodic events: {len(episodic_memory)}")
        for i, event in enumerate(episodic_memory[:5]):  # Show first 5
            print(f"  {i+1}. [{event.event_type}] {event.summary} (Actor: {event.actor})")
        if len(episodic_memory) > 5:
            print(f"  ... and {len(episodic_memory) - 5} more")
    except Exception as e:
        print(f"Error listing episodic events: {e}")
    
    print("\n--- All Procedural Memory ---")
    try:
        procedures = agent.client.server.procedural_memory_manager.list_procedures(
            agent_state=agent.agent_states.procedural_memory_agent_state,
            actor=agent.client.user,
            query='',  # Empty query to get all
            limit=50
        )
        print(f"Total procedures: {len(procedures)}")
        for i, proc in enumerate(procedures[:5]):  # Show first 5
            print(f"  {i+1}. [{proc.entry_type}] {proc.summary}")
        if len(procedures) > 5:
            print(f"  ... and {len(procedures) - 5} more")
    except Exception as e:
        print(f"Error listing procedures: {e}")
    
    print("\n--- All Resource Memory ---")
    try:
        resources = agent.client.server.resource_memory_manager.list_resources(
            agent_state=agent.agent_states.resource_memory_agent_state,
            actor=agent.client.user,
            query='',  # Empty query to get all
            limit=50
        )
        print(f"Total resources: {len(resources)}")
        for i, res in enumerate(resources[:5]):  # Show first 5
            print(f"  {i+1}. [{res.resource_type}] {res.title}: {res.summary}")
        if len(resources) > 5:
            print(f"  ... and {len(resources) - 5} more")
    except Exception as e:
        print(f"Error listing resources: {e}")
    
    print("\n--- All Knowledge Vault ---")
    try:
        knowledge_items = agent.client.server.knowledge_vault_manager.list_knowledge(
            agent_state=agent.agent_states.knowledge_vault_agent_state,
            actor=agent.client.user,
            query='',  # Empty query to get all
            limit=50
        )
        print(f"Total knowledge vault items: {len(knowledge_items)}")
        for i, kv in enumerate(knowledge_items[:5]):  # Show first 5
            print(f"  {i+1}. [{kv.entry_type}] {kv.caption} (Sensitivity: {kv.sensitivity})")
        if len(knowledge_items) > 5:
            print(f"  ... and {len(knowledge_items) - 5} more")
    except Exception as e:
        print(f"Error listing knowledge vault: {e}")
    
    print("\n--- All Semantic Memory ---")
    try:
        semantic_items = agent.client.server.semantic_memory_manager.list_semantic_items(
            agent_state=agent.agent_states.semantic_memory_agent_state,
            actor=agent.client.user,
            query='',  # Empty query to get all
            limit=50
        )
        print(f"Total semantic memory items: {len(semantic_items)}")
        for i, sem in enumerate(semantic_items[:5]):  # Show first 5
            print(f"  {i+1}. {sem.name}: {sem.summary}")
        if len(semantic_items) > 5:
            print(f"  ... and {len(semantic_items) - 5} more")
    except Exception as e:
        print(f"Error listing semantic memory: {e}")
    
    print("\nMemory content listing completed.\n")

def test_specific_memory_search(agent, memory_type, query, search_method='bm25', search_field=None, limit=10):
    """
    Test a specific memory type with a specific search method and query.
    
    Args:
        agent: The AgentWrapper instance
        memory_type: One of 'episodic', 'procedural', 'resource', 'knowledge_vault', 'semantic'
        query: The search query string
        search_method: One of 'bm25', 'embedding', 'string_match', 'fuzzy_match'
        search_field: The field to search in (if None, uses default for memory type)
        limit: Maximum number of results to return
    
    Returns:
        List of search results
    """
    
    # Default search fields for each memory type
    default_fields = {
        'episodic': 'summary',
        'procedural': 'description', 
        'resource': 'summary',
        'knowledge_vault': 'description',
        'semantic': 'name'
    }
    
    if search_field is None:
        search_field = default_fields.get(memory_type, 'summary')
    
    print(f"=== Testing {memory_type.title()} Memory ===")
    print(f"Query: '{query}'")
    print(f"Search Method: {search_method}")
    print(f"Search Field: {search_field}")
    print(f"Limit: {limit}")
    
    try:
        if memory_type == 'episodic':
            results = agent.client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=agent.agent_states.episodic_memory_agent_state,
                actor=agent.client.user,
                query=query,
                search_method=search_method,
                search_field=search_field,
                limit=limit
            )
            print(f"\nFound {len(results)} results:")
            for i, event in enumerate(results):
                print(f"  {i+1}. [{event.event_type}] {event.summary}")
                print(f"      Actor: {event.actor}, Time: {event.occurred_at}")
                if event.details:
                    print(f"      Details: {event.details[:100]}...")
                
        elif memory_type == 'procedural':
            results = agent.client.server.procedural_memory_manager.list_procedures(
                agent_state=agent.agent_states.procedural_memory_agent_state,
                actor=agent.client.user,
                query=query,
                search_method=search_method,
                search_field=search_field,
                limit=limit
            )
            print(f"\nFound {len(results)} results:")
            for i, proc in enumerate(results):
                print(f"  {i+1}. [{proc.entry_type}] {proc.description}")
                if proc.steps:
                    print(f"      Steps: {proc.steps[:100]}...")
                    
        elif memory_type == 'resource':
            results = agent.client.server.resource_memory_manager.list_resources(
                agent_state=agent.agent_states.resource_memory_agent_state,
                actor=agent.client.user,
                query=query,
                search_method=search_method,
                search_field=search_field,
                limit=limit
            )
            print(f"\nFound {len(results)} results:")
            for i, res in enumerate(results):
                print(f"  {i+1}. [{res.resource_type}] {res.title}")
                print(f"      Summary: {res.summary}")
                if res.content:
                    print(f"      Content: {res.content[:100]}...")
                    
        elif memory_type == 'knowledge_vault':
            results = agent.client.server.knowledge_vault_manager.list_knowledge(
                agent_state=agent.agent_states.knowledge_vault_agent_state,
                actor=agent.client.user,
                query=query,
                search_method=search_method,
                search_field=search_field,
                limit=limit
            )
            print(f"\nFound {len(results)} results:")
            for i, kv in enumerate(results):
                print(f"  {i+1}. [{kv.entry_type}] {kv.description}")
                print(f"      Source: {kv.source}, Sensitivity: {kv.sensitivity}")
                print(f"      Secret Value: {kv.secret_value[:20]}..." if len(kv.secret_value) > 20 else f"      Secret Value: {kv.secret_value}")
                
        elif memory_type == 'semantic':
            results = agent.client.server.semantic_memory_manager.list_semantic_items(
                agent_state=agent.agent_states.semantic_memory_agent_state,
                actor=agent.client.user,
                query=query,
                search_method=search_method,
                search_field=search_field,
                limit=limit
            )
            print(f"\nFound {len(results)} results:")
            for i, sem in enumerate(results):
                print(f"  {i+1}. {sem.name}")
                print(f"      Summary: {sem.summary}")
                if sem.details:
                    print(f"      Details: {sem.details[:100]}...")
                if sem.source:
                    print(f"      Source: {sem.source}")
        else:
            print(f"Unknown memory type: {memory_type}")
            return []
            
        print(f"\nSearch completed. Total results: {len(results)}\n")
        return results
        
    except Exception as e:
        print(f"Error during search: {e}")
        return []

def test_text_only_memorization(agent):
    """Test text-only memorization using both normal and immediate save methods"""
    print("=== Testing Text-Only Memorization ===")
    
    # Test 1: Normal text memorization (accumulation method)
    print("\n--- Test 1: Normal text memorization (accumulation method) ---")
    response = agent.send_message(
        message="I just finished reading a great book about artificial intelligence and machine learning.",
        memorizing=True
    )
    print(f"Response from normal text memorization: {response}")
    
    print("\nText-only memorization test completed.\n")

def test_all_direct_memory_operations(agent):
    """Run all direct memory operations tests (using manager.insert/update methods)"""
    test_tracker.start_test("Direct Memory Operations", "Testing direct manager method calls for all memory types")
    
    try:
        # Direct tests for each memory type
        test_episodic_memory_direct(agent)
        test_procedural_memory_direct(agent)
        test_resource_memory_direct(agent)
        test_knowledge_vault_direct(agent)
        test_semantic_memory_direct(agent)
        test_resource_memory_update_direct(agent)
        # test_tree_path_functionality_direct(agent)
        
        test_tracker.pass_test("All direct memory operations completed successfully")
    
    except Exception as e:
        test_tracker.fail_test(f"Direct memory operations failed: {e}")
        traceback.print_exc()

def test_all_indirect_memory_operations(agent):
    """Run all indirect memory operations tests (sending messages to agents)"""
    test_tracker.start_test("Indirect Memory Operations", "Testing message-based interactions with memory agents")
    
    try:
        # Indirect tests for each memory type
        test_episodic_memory_indirect(agent)
        test_procedural_memory_indirect(agent)
        test_resource_memory_indirect(agent)
        test_knowledge_vault_indirect(agent)
        test_semantic_memory_indirect(agent)
        test_resource_memory_update_indirect(agent)
        test_text_only_memorization(agent)

        test_core_memory_update_using_chat_agent(agent)
        # test_core_memory_update_using_meta_memory_manager(agent)
        # test_core_memory_replace(agent)
        
        test_tracker.pass_test("All indirect memory operations completed successfully")
    
    except Exception as e:
        test_tracker.fail_test(f"Indirect memory operations failed: {e}")
        traceback.print_exc()

def test_all_search_and_performance_operations(agent):
    """Run all search method and performance tests"""
    test_tracker.start_test("Search and Performance Tests", "Testing different search methods and performance comparisons")
    
    try:
        # Search and performance tests
        test_search_methods(agent)
        test_fts5_comprehensive(agent)
        test_fts5_performance_comparison(agent)
        test_fts5_advanced_features(agent)
        list_all_memory_content(agent)
        
        test_tracker.pass_test("All search and performance tests completed successfully")
    
    except Exception as e:
        test_tracker.fail_test(f"Search and performance tests failed: {e}")
        traceback.print_exc()

def test_all_memories():
    print("Starting comprehensive memory system tests...\n")
    
    # Initialize the agent with config file
    import sys
    from pathlib import Path
    
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        config_path = bundle_dir / 'mirix' / 'configs' / 'mirix_monitor.yaml'
    else:
        # Running in development
        config_path = Path('mirix/configs/mirix_monitor.yaml')
    
    agent = AgentWrapper(str(config_path))
    
    # agent.save_agent("./tmp/temp_agent")

    try:
        # Phase 1: Direct memory operations (manager method calls)
        test_all_direct_memory_operations(agent)

        # agent.reflexion_on_memory()
        
        # Phase 2: Indirect memory operations (message-based)
        test_all_indirect_memory_operations(agent)
        
        # Phase 3: Search methods and performance testing
        test_all_search_and_performance_operations(agent)
        
        print("All memory tests completed successfully!")

        # print("Loading agent from saved state...")
        # agent = AgentWrapper("configs/mirix.yaml", load_from="./tmp/temp_agent")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        traceback.print_exc()
    
    finally:
        # Print test summary
        test_tracker.print_summary()

def test_greeting():
    agent = AgentWrapper("configs/mirix.yaml")
    response = agent.send_message(
        message="Hello, how are you?",
        memorizing=False
    )
    print(response)

    agent.set_model("gpt-4.1")

    response = agent.send_message(
        message="Hello again!",
        memorizing=False
    )
    print(response)

def test_greeting_with_images():


    import base64
    def encode_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    agent = AgentWrapper("configs/mirix.yaml")

    # # case 1: image_url

    # message = [
    #     {'type': 'text', 'text': 'Show me what is in this image'},
    #     {'type': 'image_url', 'image_url': {'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg'}}
    # ]

    # # gpt-4.1 first
    # agent.set_model("gpt-4.1")
    # response = agent.send_message(
    #     message=message,
    #     memorizing=False
    # )
    # print("GPT-4.1 Success")

    # # Gemini
    # agent.set_model("gemini-2.0-flash")
    # response = agent.send_message(
    #     message=message,
    #     memorizing=False
    # )
    # print("Gemini Success")

    # # Claude Sonnet 3.5
    # agent.set_model("claude-sonnet-4-20250514")
    # response = agent.send_message(
    #     message=message,
    #     memorizing=False
    # )
    # print("Claude Sonnet 3.5 Success")

    # # case 2: base64_image

    # message = [
    #     {'type': 'text', 'text': 'Show me what is in this image'},
    #     # {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{encode_image('img_1.jpg')}", 'detail': 'auto'}}
    #     {'type': 'image_data', 'image_data': {'data': f"data:image/jpeg;base64,{encode_image('img_1.jpg')}", 'detail': 'auto'}}
    # ]

    # # gpt-4.1 first
    # agent.set_model("gpt-4.1")
    # response = agent.send_message(
    #     message=message,
    #     memorizing=False
    # )
    # print("GPT-4.1 Base64 Success")

    # # Gemini
    # agent.set_model("gemini-2.0-flash")
    # response = agent.send_message(
    #     message=message,
    #     memorizing=False
    # )
    # print("Gemini Base64 Success")

    # # Claude Sonnet 3.5
    # agent.set_model("claude-sonnet-4-20250514")
    # response = agent.send_message(
    #     message=message,
    #     memorizing=False
    # )
    # print("Claude Sonnet 3.5 Base64 Success")


    # case 3: file_path already submitted to google cloud
    from google import genai
    from dotenv import load_dotenv
    from datetime import datetime
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    google_client = genai.Client(api_key=api_key)
    file_ref = google_client.files.upload(file="img_1.jpg")
    agent.client.server.cloud_file_mapping_manager.add_mapping(file_ref.uri, "img_1.jpg", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    message = [
        {'type': 'text', 'text': 'Show me what is in this image'},
        {'type': 'google_cloud_file_uri', 'google_cloud_file_uri': file_ref.uri}
    ]

    # Gemini first
    agent.set_model("gemini-2.0-flash")
    response = agent.send_message(
        message=message,
        memorizing=False,
    )
    print("Gemini Cloud File Success") 

    # gpt-4.1
    agent.set_model("gpt-4.1")
    response = agent.send_message(
        message=message,
        memorizing=False
    )
    print("GPT-4.1 Cloud File Success")

    # Claude Sonnet 3.5
    agent.set_model("claude-sonnet-4-20250514")
    response = agent.send_message(
        message=message,
        memorizing=False
    )
    print("Claude Sonnet 3.5 Cloud File Success")

def test_greeting_with_files(file_path):
    """Test file handling functionality with OpenAI, Claude, and Google AI"""
    print("=== Testing File Handling with Multiple AI Providers ===")
    
    agent = AgentWrapper("configs/mirix.yaml")
    
    # Test 1: OpenAI-style file format
    print("\n--- Test 1: OpenAI-style file format ---")
    message = [
        {
            "type": "file_uri",
            "file_uri": file_path
        },
        {
            "type": "text",
            "text": "Please analyze this document and provide a summary of its contents.",
        },
    ]
    
    # Test with GPT-4.1
    print("Testing with GPT-4.1...")
    agent.set_model("gpt-4.1")
    response = agent.send_message(
        message=message,
        memorizing=False
    )
    print("GPT-4.1 File Success - Response received")
    
    # Test with Claude Sonnet 3.5
    print("Testing with Claude Sonnet 3.5...")
    agent.set_model("claude-sonnet-4-20250514")
    try:
        response = agent.send_message(
            message=message,
            role="user",
            memorizing=False
        )
        print("Claude Sonnet 3.5 File Success - Response received")
    except Exception as e:
        print(f"Claude Sonnet 3.5 File Error: {e}")
    
    # Test with Gemini
    print("Testing with Gemini...")
    agent.set_model("gemini-2.0-flash")
    try:
        response = agent.send_message(
            message=message,
            role="user",
            memorizing=False
        )
        print("Gemini File Success - Response received")
    except Exception as e:
        print(f"Gemini File Error: {e}")
    
    # Test 2: file_uri format
    print("\n--- Test 2: file_uri format ---")
    message_uri = [
        {
            "type": "file_uri",
            "file_uri": file_path
        },
        {
            "type": "text",
            "text": "What are the key topics discussed in this document?",
        },
    ]
    
    # Test with different models
    for model_name in ["gpt-4.1", "claude-sonnet-4-20250514", "gemini-2.0-flash"]:
        print(f"Testing file_uri format with {model_name}...")
        agent.set_model(model_name)
        try:
            response = agent.send_message(
                message=message_uri,
                role="user",
                memorizing=False
            )
            print(f"{model_name} file_uri Success - Response received")
        except Exception as e:
            print(f"{model_name} file_uri Error: {e}")
    
    # Test 3: Mixed content (text + file)
    print("\n--- Test 3: Mixed content (text + file) ---")
    mixed_message = [
        {
            "type": "text",
            "text": "I need help analyzing the attached document. Please focus on:"
        },
        {
            "type": "file",
            "file": {
                "file_path": file_path,
            }
        },
        {
            "type": "text",
            "text": "1. Main themes\n2. Key findings\n3. Recommendations\n\nProvide a structured analysis."
        },
    ]
    
    # Test mixed content with GPT-4.1
    print("Testing mixed content with GPT-4.1...")
    agent.set_model("gpt-4.1")
    try:
        response = agent.send_message(
            message=mixed_message,
            role="user",
            memorizing=False
        )
        print("GPT-4.1 Mixed Content Success - Response received")
    except Exception as e:
        print(f"GPT-4.1 Mixed Content Error: {e}")
    
    print("\nFile handling tests completed!")

def test_file_types():
    """Test different file types handling"""
    print("=== Testing Different File Types ===")
    
    agent = AgentWrapper("configs/mirix.yaml")
    agent.set_model("gpt-4.1")
    
    # Create test files of different types (for demonstration)
    test_files = [
        ("exp1.pdf", "PDF document"),
        # Add more file types as needed
    ]
    
    for file_path, file_desc in test_files:
        if os.path.exists(file_path):
            print(f"\n--- Testing {file_desc} ---")
            message = [
                {
                    "type": "file",
                    "file": {
                        "file_path": file_path,
                    }
                },
                {
                    "type": "text",
                    "text": f"Please analyze this {file_desc} and tell me what type of content it contains.",
                },
            ]
            
            try:
                response = agent.send_message(
                    message=message,
                    role="user",
                    memorizing=False
                )
                print(f"Successfully processed {file_desc}")
            except Exception as e:
                print(f"Error processing {file_desc}: {e}")
        else:
            print(f"File not found: {file_path}")
    
    print("\nFile type tests completed!")

def test_file_with_memory():
    """Test file handling with memory enabled"""
    print("=== Testing File Handling with Memory ===")
    
    agent = AgentWrapper("configs/mirix.yaml")
    agent.set_model("gpt-4.1")
    
    file_path = "exp1.pdf"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Test memorizing file content
    message = [
        {
            "type": "file",
            "file": {
                "file_path": file_path,
            }
        },
        {
            "type": "text",
            "text": "Please read and remember the contents of this document for future reference.",
        },
    ]
    
    try:
        response = agent.send_message(
            message=message,
            role="user",
            memorizing=True  # Enable memory
        )
        print("File content memorized successfully")
        
        # Follow up question to test if file content was remembered
        follow_up = agent.send_message(
            message="What were the main points from the document I shared earlier?",
            role="user",
            memorizing=False
        )
        print("Follow-up question answered based on memorized file content")
        
    except Exception as e:
        print(f"Error in memory test: {e}")
    
    print("\nFile memory test completed!")

def run_file_tests():
    """Run all file-related tests"""
    test_tracker.start_test("File Handling Tests", "Testing file uploading and processing with multiple AI providers")
    
    try:
        file_path = "exp1.pdf"
        
        # Check if test file exists
        if not os.path.exists(file_path):
            test_tracker.fail_test(f"Test file {file_path} not found. Please ensure it exists in the current directory.")
            return
        
        # Run all file tests
        test_greeting_with_files(file_path)
        test_file_types()
        test_file_with_memory()
        
        test_tracker.pass_test("All file tests completed successfully")
        
    except Exception as e:
        test_tracker.fail_test(f"File testing failed: {e}")
        traceback.print_exc()
    
    finally:
        # Print test summary for file tests
        test_tracker.print_summary()

# =================================================================
# DIRECT MEMORY OPERATIONS TESTS (using manager.insert/update methods)
# =================================================================

def test_episodic_memory_direct(agent):
    """Test direct episodic memory operations using manager methods"""
    test_tracker.start_test("Direct Episodic Memory", "Testing direct episodic memory operations using manager methods")
    
    try:
        # Test 1: Direct insert with basic event
        subtest_idx = test_tracker.start_subtest("Direct Event Insert")
        try:
            event = agent.client.server.episodic_memory_manager.insert_event(
                actor=agent.client.user,
                agent_state=agent.agent_states.episodic_memory_agent_state,
                event_type='activity',
                timestamp=datetime.now(agent.timezone),
                event_actor='user',
                summary='Started working on a coding project',
                details='User began working on a new Python project for data analysis',
                organization_id=agent.client.org_id,
                tree_path=['work', 'projects', 'coding']
            )
            print(f"Inserted event with ID: {event.id}")
            test_tracker.pass_subtest(subtest_idx, f"Event inserted with ID: {event.id}")
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
            raise e

        # Test 2: Direct search operations
        subtest_idx = test_tracker.start_subtest("Direct Search Operations")
        try:
            search_response = agent.client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=agent.agent_states.episodic_memory_agent_state,
                actor=agent.client.user,
                query="coding",
                search_method='embedding',
                search_field='details',
                limit=10
            )
            print(f"Semantic search found {len(search_response)} results")
            
            search_response = agent.client.server.episodic_memory_manager.list_episodic_memory(
                agent_state=agent.agent_states.episodic_memory_agent_state,
                actor=agent.client.user,
                query="coding",
                search_method='bm25',
                search_field='summary',
                limit=10
            )
            print(f"FTS5 search found {len(search_response)} results")
            test_tracker.pass_subtest(subtest_idx, "Search operations completed successfully")
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
            raise e
        
        # Test 3: Update event (episodic_memory_merge)
        subtest_idx = test_tracker.start_subtest("Direct Event Update")
        try:
            updated_event = agent.client.server.episodic_memory_manager.update_event(
                event_id=event.id,
                new_summary="Continued working on coding project with progress",
                new_details="Added data visualization features and completed the main algorithm implementation",
                actor=agent.client.user
            )
            print(f"Updated event - New summary: {updated_event.summary}")
            print(f"Updated details: {updated_event.details}")
            test_tracker.pass_subtest(subtest_idx, "Event updated successfully")
        except Exception as e:
            test_tracker.fail_subtest(e, subtest_idx)
            # Don't raise here as this might be expected behavior
        
        test_tracker.pass_test("All direct episodic memory operations completed successfully")
        
    except Exception as e:
        test_tracker.fail_test(f"Direct episodic memory test failed: {e}")
        traceback.print_exc()

def test_procedural_memory_direct(agent):
    """Test direct procedural memory operations using manager methods"""
    print("=== Direct Procedural Memory Tests ===")
    
    # Test 1: Direct insert
    print("\n--- Test 1: Direct Procedure Insert ---")
    procedure = agent.client.server.procedural_memory_manager.insert_procedure(
        agent_state=agent.agent_states.procedural_memory_agent_state,
        entry_type='process',
        summary='How to make coffee',
        steps=[
            'Boil water',
            'Add coffee grounds', 
            'Pour hot water',
            'Wait 4 minutes',
            'Serve'
        ],
        actor=agent.client.user,
        tree_path=['recipes', 'beverages', 'hot'],
        organization_id=agent.client.org_id
    )
    print(f"Inserted procedure with ID: {procedure.id}")

    # Test 2: Direct search operations
    print("\n--- Test 2: Direct Procedure Search Operations ---")
    search_results = agent.client.server.procedural_memory_manager.list_procedures(
        agent_state=agent.agent_states.procedural_memory_agent_state,
        actor=agent.client.user,
        query="coffee",
        search_method='embedding',
        search_field='summary',
        limit=10
    )
    print(f"Semantic search found {len(search_results)} procedures")
    
    search_results = agent.client.server.procedural_memory_manager.list_procedures(
        agent_state=agent.agent_states.procedural_memory_agent_state,
        actor=agent.client.user,
        query="coffee",
        search_method='bm25',
        search_field='summary',
        limit=10
    )
    print(f"FTS5 search found {len(search_results)} procedures")
    
    # Cleanup
    agent.client.server.procedural_memory_manager.delete_procedure_by_id(procedure_id=procedure.id, actor=agent.client.user)
    print("Cleaned up test procedure")
    print("Direct procedural memory tests completed.\n")

def test_resource_memory_direct(agent):
    """Test direct resource memory operations using manager methods"""
    print("=== Direct Resource Memory Tests ===")
    
    # Test 1: Direct insert
    print("\n--- Test 1: Direct Resource Insert ---")
    resource = agent.client.server.resource_memory_manager.insert_resource(
        actor=agent.client.user,
        agent_state=agent.agent_states.resource_memory_agent_state,
        title='Python Documentation Test',
        summary='Test resource for direct operations',
        resource_type='documentation',
        content='This is test content for direct resource operations',
        tree_path=['test', 'documentation'],
        organization_id=agent.client.org_id
    )
    print(f"Inserted resource with ID: {resource.id}")
    
    # Test 2: Direct search operations
    print("\n--- Test 2: Direct Resource Search Operations ---")
    search_results = agent.client.server.resource_memory_manager.list_resources(
        agent_state=agent.agent_states.resource_memory_agent_state,
        actor=agent.client.user,
        query="Python",
        search_method='embedding',
        search_field='summary',
        limit=10
    )
    print(f"Semantic search found {len(search_results)} resources")
    
    search_results = agent.client.server.resource_memory_manager.list_resources(
        agent_state=agent.agent_states.resource_memory_agent_state,
        actor=agent.client.user,
        query="Python",
        search_method='bm25',
        search_field='title',
        limit=10
    )
    print(f"FTS5 search found {len(search_results)} resources")
    
    # Cleanup
    agent.client.server.resource_memory_manager.delete_resource_by_id(resource_id=resource.id, actor=agent.client.user)
    print("Cleaned up test resource")
    print("Direct resource memory tests completed.\n")

def test_knowledge_vault_direct(agent):
    """Test direct knowledge vault operations using manager methods"""
    print("=== Direct Knowledge Vault Tests ===")
    
    # Test 1: Direct insert
    print("\n--- Test 1: Direct Knowledge Insert ---")
    knowledge = agent.client.server.knowledge_vault_manager.insert_knowledge(
        actor=agent.client.user,
        agent_state=agent.agent_states.knowledge_vault_agent_state,
        entry_type='credential',
        source='development_environment',
        sensitivity='medium',
        secret_value='test_api_key_12345',
        caption='Test API key for direct operations',
        organization_id=agent.client.org_id
    )
    print(f"Inserted knowledge with ID: {knowledge.id}")
    
    search_results = agent.client.server.knowledge_vault_manager.list_knowledge(
        agent_state=agent.agent_states.knowledge_vault_agent_state,
        actor=agent.client.user,
        query="test_api_key",
        search_method='bm25',
        search_field='secret_value',
        limit=10
    )
    print(f"FTS5 search found {len(search_results)} knowledge items")
    
    # Cleanup
    agent.client.server.knowledge_vault_manager.delete_knowledge_by_id(knowledge_vault_item_id=knowledge.id, actor=agent.client.user)
    print("Cleaned up test knowledge")
    print("Direct knowledge vault tests completed.\n")

def test_semantic_memory_direct(agent):
    """Test direct semantic memory operations using manager methods"""
    print("=== Direct Semantic Memory Tests ===")
    
    # Test 1: Direct insert
    print("\n--- Test 1: Direct Semantic Insert ---")
    semantic_item = agent.client.server.semantic_memory_manager.insert_semantic_item(
        actor=agent.client.user,
        agent_state=agent.agent_states.semantic_memory_agent_state,
        name='Test Machine Learning Concept',
        summary='A test concept for direct operations',
        details='This is detailed information about the test concept for direct operations',
        source='test_source',
        tree_path=['test', 'concepts', 'ai'],
        organization_id=agent.client.org_id
    )
    print(f"Inserted semantic item with ID: {semantic_item.id}")
    
    # Test 2: Direct search operations
    print("\n--- Test 2: Direct Semantic Search Operations ---")
    search_results = agent.client.server.semantic_memory_manager.list_semantic_items(
        agent_state=agent.agent_states.semantic_memory_agent_state,
        actor=agent.client.user,
        query="Test Machine Learning",
        search_method='embedding',
        search_field='name',
        limit=10
    )
    print(f"Semantic search found {len(search_results)} semantic items")
    
    search_results = agent.client.server.semantic_memory_manager.list_semantic_items(
        agent_state=agent.agent_states.semantic_memory_agent_state,
        actor=agent.client.user,
        query="Test Machine Learning",
        search_method='bm25',
        search_field='name',
        limit=10
    )
    print(f"FTS5 search found {len(search_results)} semantic items")
    
    # Cleanup
    agent.client.server.semantic_memory_manager.delete_semantic_item_by_id(semantic_memory_id=semantic_item.id, actor=agent.client.user)
    print("Cleaned up test semantic item")
    print("Direct semantic memory tests completed.\n")

def test_resource_memory_update_direct(agent):
    """Test direct resource memory update operations using manager methods"""
    print("=== Direct Resource Memory Update Tests ===")
    
    # Test 1: Create resources for update testing
    print("\n--- Test 1: Create Test Resources ---")
    resource1 = agent.client.server.resource_memory_manager.insert_resource(
        actor=agent.client.user,
        agent_state=agent.agent_states.resource_memory_agent_state,
        title='Initial Test Resource 1',
        summary='Initial summary for update testing',
        resource_type='documentation',
        content='Initial content for resource update testing',
        tree_path=['test', 'updates', 'resource1'],
        organization_id=agent.client.org_id
    )
    
    resource2 = agent.client.server.resource_memory_manager.insert_resource(
        actor=agent.client.user,
        agent_state=agent.agent_states.resource_memory_agent_state,
        title='Initial Test Resource 2',
        summary='Second resource for update testing',
        resource_type='guide',
        content='Second resource content for testing',
        tree_path=['test', 'updates', 'resource2'],
        organization_id=agent.client.org_id
    )
    
    print(f"Created test resources: {resource1.id}, {resource2.id}")
    
    # Test 2: Search after creation
    print("\n--- Test 2: Search After Creation ---")
    search_results = agent.client.server.resource_memory_manager.list_resources(
        agent_state=agent.agent_states.resource_memory_agent_state,
        actor=agent.client.user,
        query="Initial Test",
        search_method='embedding',
        search_field='summary',
        limit=10
    )
    print(f"Semantic search found {len(search_results)} resources with 'Initial Test'")
    
    search_results = agent.client.server.resource_memory_manager.list_resources(
        agent_state=agent.agent_states.resource_memory_agent_state,
        actor=agent.client.user,
        query="Initial Test",
        search_method='bm25',
        search_field='title',
        limit=10
    )
    print(f"FTS5 search found {len(search_results)} resources with 'Initial Test'")
    
    # Test 3: Cleanup test resources
    print("\n--- Test 3: Cleanup Test Resources ---")
    try:
        agent.client.server.resource_memory_manager.delete_resource_by_id(resource_id=resource1.id, actor=agent.client.user)
        agent.client.server.resource_memory_manager.delete_resource_by_id(resource_id=resource2.id, actor=agent.client.user)
        print("Cleaned up test resources")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    print("Direct resource memory update tests completed.\n")

def test_tree_path_functionality_direct(agent):
    """Test tree_path functionality using direct manager operations"""
    print("=== Direct Tree Path Functionality Tests ===")
    
    # Test 1: Insert items with various tree paths
    print("\n--- Test 1: Insert Items with Tree Paths ---")
    
    # Episodic with tree paths
    episodic_event = agent.client.server.episodic_memory_manager.insert_event(
        agent_state=agent.agent_states.episodic_memory_agent_state,
        event_type='activity',
        timestamp=datetime.now(agent.timezone),
        actor='user',
        summary='Tree path test event',
        details='Testing tree path functionality with direct operations',
        organization_id=agent.client.org_id,
        tree_path=['test', 'tree_paths', 'episodic']
    )
    print(f"Inserted episodic event with tree path: {' > '.join(episodic_event.tree_path)}")
    
    # Procedural with tree paths
    procedure = agent.client.server.procedural_memory_manager.insert_procedure(
        agent_state=agent.agent_states.procedural_memory_agent_state,
        entry_type='process',
        summary='Tree path test procedure',
        steps=[
            'Test step 1',
            'Test step 2', 
            'Complete'
        ],
        actor=agent.client.user,
        tree_path=['test', 'tree_paths', 'procedural'],
        organization_id=agent.client.org_id
    )
    print(f"Inserted procedure with tree path: {' > '.join(procedure.tree_path)}")
    
    # Resource with tree paths
    resource = agent.client.server.resource_memory_manager.insert_resource(
        agent_state=agent.agent_states.resource_memory_agent_state,
        title='Tree Path Test Resource',
        summary='Resource for testing tree path functionality',
        content='This resource tests tree path functionality',
        resource_type='test',
        tree_path=['test', 'tree_paths', 'resource'],
        organization_id=agent.client.org_id
    )
    print(f"Inserted resource with tree path: {' > '.join(resource.tree_path)}")
    
    # Semantic with tree paths
    semantic_item = agent.client.server.semantic_memory_manager.insert_semantic_item(
        agent_state=agent.agent_states.semantic_memory_agent_state,
        name='Tree Path Test Concept',
        summary='Concept for testing tree path functionality',
        details='This concept tests tree path functionality in semantic memory',
        source='test_source',
        tree_path=['test', 'tree_paths', 'semantic'],
        organization_id=agent.client.org_id
    )
    print(f"Inserted semantic item with tree path: {' > '.join(semantic_item.tree_path)}")
    
    # Test 2: Verify tree paths in search results
    print("\n--- Test 2: Verify Tree Paths in Search Results ---")
    
    # Search and verify tree paths using semantic search
    episodic_results = agent.client.server.episodic_memory_manager.list_episodic_memory(
        agent_state=agent.agent_states.episodic_memory_agent_state,
        actor=agent.client.user,
        query="tree path test",
        search_method='embedding',
        search_field='summary',
        limit=10
    )
    print(f"Semantic search found {len(episodic_results)} episodic results")
    for result in episodic_results:
        if hasattr(result, 'tree_path') and result.tree_path:
            print(f"Episodic: {result.summary} -> Tree Path: {' > '.join(result.tree_path)}")
    
    # Search and verify tree paths using FTS5 search
    episodic_results = agent.client.server.episodic_memory_manager.list_episodic_memory(
        agent_state=agent.agent_states.episodic_memory_agent_state,
        actor=agent.client.user,
        query="tree path test",
        search_method='bm25',
        limit=10
    )
    print(f"FTS5 search found {len(episodic_results)} episodic results")
    for result in episodic_results:
        if hasattr(result, 'tree_path') and result.tree_path:
            print(f"Episodic: {result.summary} -> Tree Path: {' > '.join(result.tree_path)}")
    
    # Cleanup
    print("\n--- Test 3: Cleanup Tree Path Test Items ---")
    try:
        agent.client.server.episodic_memory_manager.delete_event_by_id(episodic_event.id, actor=agent.client.user)
        agent.client.server.procedural_memory_manager.delete_procedure_by_id(procedure.id, actor=agent.client.user)
        agent.client.server.resource_memory_manager.delete_resource_by_id(resource.id, actor=agent.client.user)
        agent.client.server.semantic_memory_manager.delete_semantic_item_by_id(semantic_item.id, actor=agent.client.user)
        print("Cleaned up all tree path test items")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    print("Direct tree path functionality tests completed.\n")

# =================================================================
# INDIRECT MEMORY OPERATIONS TESTS (message-based interactions)
# =================================================================

def test_core_memory_update_using_chat_agent(agent):
    """Test episodic memory update using chat agent"""
    print("=== Episodic Memory Update Using Chat Agent Tests ===")
    
    # Test 1: Send message to episodic memory agent
    print("\n--- Test 1: Message to Episodic Agent ---")
    test_message = "Update Core Memory with the user's name as John Doe"

    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.agent_state.id,
        kwargs={
            'message': test_message,
        },
        agent_type='chat'
    )
    print(f"Response from episodic memory agent: {response}")
    print(f"Agent type: {agent_type}")

    print("Episodic memory update using chat agent tests completed.\n")

def test_core_memory_update_using_meta_memory_manager(agent):
    """Test episodic memory update using meta memory manager"""
    print("=== Episodic Memory Update Using Meta Memory Manager Tests ===")
    
    # Test 1: Send message to episodic memory agent
    print("\n--- Test 1: Message to Episodic Agent ---")
    test_message = "Update Core Memory with the user's name as John Doe"

    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.meta_memory_agent_state.id,
        kwargs={
            'message': test_message,
            'message_queue': agent.message_queue
        },
        agent_type='meta_memory_agent'
    )
    print(f"Response from episodic memory agent: {response}")
    print(f"Agent type: {agent_type}")
    
    print("Episodic memory update using meta memory manager tests completed.\n")

# def test_core_memory_replace(agent):
#     """Test core memory replace functionality by adding and deleting items"""
#     print("=== Core Memory Replace Tests ===")
    
#     # Test 1: Add something to core memory
#     print("\n--- Test 1: Add to Core Memory ---")
#     add_message = "Please add to core memory that the user's favorite programming language is Python"
    
#     response, agent_type = agent.message_queue.send_message_in_queue(
#         agent.client,
#         agent.agent_states.core_memory_agent_state.id,
#         kwargs={
#             'message': add_message,
#         },
#         agent_type='core_memory'
#     )
#     print(f"Response from core memory agent (add): {response}")
#     print(f"Agent type: {agent_type}")
    
#     # Test 2: Add another item to core memory
#     print("\n--- Test 2: Add another item to Core Memory ---")
#     add_message2 = "Please add to core memory that the user's favorite IDE is VSCode"
    
#     response2, agent_type2 = agent.message_queue.send_message_in_queue(
#         agent.client,
#         agent.agent_states.core_memory_agent_state.id,
#         kwargs={
#             'message': add_message2,
#         },
#         agent_type='core_memory'
#     )
#     print(f"Response from core memory agent (add 2): {response2}")
#     print(f"Agent type: {agent_type2}")
    
#     # Test 3: Delete something from core memory
#     print("\n--- Test 3: Delete from Core Memory ---")
#     delete_message = "Please remove from core memory any information about the user's favorite programming language"
    
#     response3, agent_type3 = agent.message_queue.send_message_in_queue(
#         agent.client,
#         agent.agent_states.core_memory_agent_state.id,
#         kwargs={
#             'message': delete_message,
#         },
#         agent_type='core_memory'
#     )
#     print(f"Response from core memory agent (delete): {response3}")
#     print(f"Agent type: {agent_type3}")
    
#     # Test 4: Replace core memory item
#     print("\n--- Test 4: Replace Core Memory Item ---")
#     replace_message = "Please replace the user's favorite IDE in core memory from VSCode to IntelliJ IDEA"
    
#     response4, agent_type4 = agent.message_queue.send_message_in_queue(
#         agent.client,
#         agent.agent_states.core_memory_agent_state.id,
#         kwargs={
#             'message': replace_message,
#         },
#         agent_type='core_memory'
#     )
#     print(f"Response from core memory agent (replace): {response4}")
#     print(f"Agent type: {agent_type4}")
    
#     print("Core memory replace tests completed.\n")
#     import ipdb; ipdb.set_trace()

def test_episodic_memory_indirect(agent):
    """Test episodic memory through message-based interactions"""
    print("=== Indirect Episodic Memory Tests ===")
    
    # Test 1: Send message to episodic memory agent
    print("\n--- Test 1: Message to Episodic Agent ---")
    test_message = "This is a test memory about going to the grocery store yesterday. I bought apples, bananas, and milk."
    
    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.episodic_memory_agent_state.id,
        kwargs={
            'message': test_message,
        },
        agent_type='episodic_memory'
    )
    print(f"Response from episodic memory agent: {response}")
    print(f"Agent type: {agent_type}")
    
    # Test 2: Another message for variety
    test_message2 = "This is a test memory about going to the grocery store two days ago. I bought chicken, beef and pork."
    
    response2, agent_type2 = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.episodic_memory_agent_state.id,
        kwargs={
            'message': test_message2,
        },
        agent_type='episodic_memory'
    )
    print(f"Second response: {response2}")

    test_message3 = "Update the last episodic memory, because I didn't buy chicken two days ago."

    response3, agent_type3 = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.episodic_memory_agent_state.id,
        kwargs={
            'message': test_message3,
        },
        agent_type='episodic_memory'
    )
    
    print("Indirect episodic memory tests completed.\n")

def test_procedural_memory_indirect(agent):
    """Test procedural memory through message-based interactions"""
    print("=== Indirect Procedural Memory Tests ===")
    
    # Test 1: Send procedure message
    print("\n--- Test 1: Message to Procedural Agent ---")
    test_message = "This is how you can cook a pizza: 1. Preheat oven to 450 degrees F (230 degrees C). 2. In a large bowl, mix together pizza dough, pizza sauce, cheese, and pepperoni. 3. Bake in the preheated oven for 10 minutes, or until the crust is golden brown and the cheese is bubbly. 4. Let cool for 5 minutes before serving."

    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.procedural_memory_agent_state.id,
        kwargs={
            'message': test_message,
        },
        agent_type='procedural_memory'
    )
    
    print(f"Response from procedural memory agent: {response}")
    print(f"Agent type: {agent_type}")
    
    print("Indirect procedural memory tests completed.\n")

def test_resource_memory_indirect(agent):
    """Test resource memory through message-based interactions"""
    print("=== Indirect Resource Memory Tests ===")
    
    # Test 1: Send resource message
    print("\n--- Test 1: Message to Resource Agent ---")
    test_message = "Here is the documentation for the Python programming language: Python is a programming language that lets you work quickly and integrate systems more effectively."
    
    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.resource_memory_agent_state.id,
        kwargs={
            'message': test_message,
        },
        agent_type='resource_memory'
    )
    
    print(f"Response from resource memory agent: {response}")
    print(f"Agent type: {agent_type}")
    
    print("Indirect resource memory tests completed.\n")

def test_knowledge_vault_indirect(agent):
    """Test knowledge vault through message-based interactions"""
    print("=== Indirect Knowledge Vault Tests ===")
    
    # Test 1: Send sensitive information message
    print("\n--- Test 1: Message to Knowledge Vault Agent ---")
    test_message = "The following is the password for my apartment: w42x532"
    
    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.knowledge_vault_agent_state.id,
        kwargs={
            'message': test_message,
        },
        agent_type='knowledge_vault'
    )
    
    print(f"Response from knowledge vault agent: {response}")
    print(f"Agent type: {agent_type}")
    
    print("Indirect knowledge vault tests completed.\n")

def test_semantic_memory_indirect(agent):
    """Test semantic memory through message-based interactions"""
    print("=== Indirect Semantic Memory Tests ===")
    
    # Test 1: Send conceptual knowledge message
    print("\n--- Test 1: Message to Semantic Agent ---")
    test_message = "I want to store some conceptual knowledge about machine learning and AI: Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
    
    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.semantic_memory_agent_state.id,
        {
            'message': test_message,
        },
        agent_type='semantic_memory'
    )
    
    print(f"Response from semantic memory agent: {response}")
    print(f"Agent type: {agent_type}")
    
    print("Indirect semantic memory tests completed.\n")

def test_resource_memory_update_indirect(agent):
    """Test resource memory updates through message-based interactions"""
    print("=== Indirect Resource Memory Update Tests ===")
    
    # Test 1: Send update message
    print("\n--- Test 1: Update Message to Resource Agent ---")
    
    # First create some resources via direct method for testing updates
    resource1 = agent.client.server.resource_memory_manager.insert_resource(
        actor=agent.client.user,
        agent_state=agent.agent_states.resource_memory_agent_state,
        title='Update Test Resource',
        summary='Resource for update testing via messages',
        resource_type='documentation',
        content='Initial content for message-based update testing',
        tree_path=['test', 'message_updates'],
        organization_id=agent.client.org_id
    )

    print("all resources:", [res.id for res in agent.client.server.resource_memory_manager.list_resources(
        agent_state=agent.agent_states.resource_memory_agent_state,
        actor=agent.client.user,
        limit=100,
    )])

    update_message = f"""I want to update my resource memory. Please update resource {resource1.id} to have title "Updated via Message", summary "Updated through message-based interaction", and content "This content was updated via agent message"."""
    
    response, agent_type = agent.message_queue.send_message_in_queue(
        agent.client,
        agent.agent_states.resource_memory_agent_state.id,
        kwargs={
            'message': update_message,
        },
        agent_type='resource_memory'
    )
    
    print(f"Update response from resource memory agent: {response}")
    print(f"Agent type: {agent_type}")
    
    print("Indirect resource memory update tests completed.\n")

if __name__ == "__main__":

    delete_after_test = True

    test_all_memories() 
    # test_greeting_with_images()
    # run_file_tests()
