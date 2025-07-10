import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

from mem0 import MemoryClient

load_dotenv()


class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
            # org_id=os.getenv("MEM0_ORGANIZATION_ID"),
            # project_id=os.getenv("MEM0_PROJECT_ID"),
        )
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def load_existing_results(self):
        """Load existing results from file if it exists."""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r") as f:
                    existing_data = json.load(f)
                    print(f"Loaded existing results from {self.output_path}")
                    # Convert string keys back to integers and load into defaultdict
                    for key, value in existing_data.items():
                        self.results[int(key)] = value
                    return True
            except (json.JSONDecodeError, Exception) as e:
                print(f"Failed to load existing results: {str(e)}. Starting fresh.")
                return False
        return False

    def get_progress_info(self, data):
        """Get information about current progress."""
        total_conversations = len(data)
        total_questions = sum(len(item["qa"]) for item in data)
        
        processed_conversations = len(self.results)
        processed_questions = sum(len(questions) for questions in self.results.values())
        
        return {
            'total_conversations': total_conversations,
            'total_questions': total_questions,
            'processed_conversations': processed_conversations,
            'processed_questions': processed_questions
        }

    def should_skip_conversation(self, conv_idx, expected_questions):
        """Check if this conversation has already been fully processed."""
        if conv_idx in self.results:
            if len(self.results[conv_idx]) >= expected_questions:
                return True
        return False

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    print("Searching with graph")
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        top_k=self.top_k,
                        filter_memories=self.filter_memories,
                        enable_graph=True,
                        output_format="v1.1",
                    )
                else:
                    memories = self.mem0_client.search(
                        query, user_id=user_id, top_k=self.top_k, filter_memories=self.filter_memories
                    )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        # Load existing results if any
        self.load_existing_results()
        
        # Print progress information
        progress = self.get_progress_info(data)
        print(f"Progress: {progress['processed_questions']}/{progress['total_questions']} questions processed")
        print(f"Conversations: {progress['processed_conversations']}/{progress['total_conversations']} completed")

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            # Skip if this conversation is already fully processed
            if self.should_skip_conversation(idx, len(qa)):
                print(f"Conversation {idx} already completed, skipping...")
                continue

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            # Determine starting question index for this conversation
            start_question_idx = len(self.results[idx]) if idx in self.results else 0
            
            if start_question_idx > 0:
                print(f"Resuming conversation {idx} from question {start_question_idx}")

            for question_idx, question_item in tqdm(
                enumerate(qa[start_question_idx:], start=start_question_idx), 
                total=len(qa) - start_question_idx, 
                desc=f"Processing questions for conversation {idx}", 
                leave=False
            ):
                try:
                    result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                    self.results[idx].append(result)
                except Exception as e:
                    print(f"Error processing question {question_idx} in conversation {idx}: {str(e)}")
                    # Create an error result instead of crashing
                    error_result = {
                        "question": question_item.get("question", ""),
                        "answer": question_item.get("answer", ""),
                        "category": question_item.get("category", -1),
                        "evidence": question_item.get("evidence", []),
                        "response": f"Error: {str(e)}",
                        "adversarial_answer": question_item.get("adversarial_answer", ""),
                        "speaker_1_memories": [],
                        "speaker_2_memories": [],
                        "num_speaker_1_memories": 0,
                        "num_speaker_2_memories": 0,
                        "speaker_1_memory_time": 0,
                        "speaker_2_memory_time": 0,
                        "speaker_1_graph_memories": [],
                        "speaker_2_graph_memories": [],
                        "response_time": 0,
                    }
                    self.results[idx].append(error_result)
                    continue

            # Save results after each conversation is completed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)

        print("Processing completed successfully!")

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="results/mem0_search_results.json", help="Path to save results")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", help="Whether to use graph search")
    args = parser.parse_args()
    
    memory_search = MemorySearch(
        output_path=args.output_path,
        top_k=args.top_k,
        filter_memories=args.filter_memories,
        is_graph=args.is_graph
    )
    memory_search.process_data_file("../../dataset/locomo10.json")
