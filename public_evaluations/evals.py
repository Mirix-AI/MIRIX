import argparse
import concurrent.futures
import json
import threading
from collections import defaultdict
import nltk
import os

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')


def process_item(item):
    """Process a single item from the results"""
    gt_answer = str(item["answer"])
    pred_answer = str(item["response"])
    question = str(item["question"])
    
    # Get category if it exists in metadata
    category = None
    if "metadata" in item and item["metadata"] is not None:
        #category = str(item["metadata"].get("category", "Unknown")) ##TODO: temp fix
        pass 

    # Clean up question if it contains the approach instructions
    if "Question:" in question:
        question = question.split("\n\n")[-1].strip().split("Question:")[1].strip()
    else:
        question = question

    try:
        metrics = calculate_metrics(pred_answer, gt_answer)
        bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        result = {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "bleu_score": bleu_scores.get("bleu1", 0),
            "f1_score": metrics.get("f1", 0),
            "llm_score": llm_score,
        }
        
        # Add category only if it exists
        if category is not None:
            result["category"] = category
            
        return result
    except Exception as e:
        print(f"Error processing item: {e}")
        print(f"Question: {question}")
        print(f"GT Answer: {gt_answer}")
        print(f"Predicted Answer: {pred_answer}")
        # Return default values if processing fails
        result = {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "bleu_score": 0.0,
            "f1_score": 0.0,
            "llm_score": 0.0,
        }
        if category is not None:
            result["category"] = category
        return result


def calculate_category_stats(results):
    """Calculate statistics grouped by category if categories exist"""
    category_stats = {}
    
    # Check if any results have categories
    has_categories = any("category" in item for item in results)
    if not has_categories:
        return None
    
    # Define the five categories
    categories = ["1", "2", "3", "4"]
    
    # Group results by category
    category_results = defaultdict(list)
    for item in results:
        if "category" in item:
            category_results[item["category"]].append(item)
    
    # Calculate stats for each category
    for category in categories:
        items = category_results.get(category, [])
        if items:
            bleu_scores = [item["bleu_score"] for item in items]
            f1_scores = [item["f1_score"] for item in items]
            llm_scores = [item["llm_score"] for item in items]
            
            category_stats[category] = {
                "num_questions": len(items),
                "avg_bleu_score": sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
                "avg_f1_score": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
                "avg_llm_score": sum(llm_scores) / len(llm_scores) if llm_scores else 0,
            }
        else:
            category_stats[category] = {
                "num_questions": 0,
                "avg_bleu_score": 0,
                "avg_f1_score": 0,
                "avg_llm_score": 0,
            }
    
    return category_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate memory agent results")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input results file or directory from eval.py"
    )
    parser.add_argument(
        "--output_file", type=str, default="./evaluation_metrics.json", help="Path to save the evaluation results"
    )
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")

    args = parser.parse_args()

    all_results = []
    results_lock = threading.Lock()

    if os.path.isdir(args.input_file):
        # Process all numbered directories
        numbered_dirs = [d for d in os.listdir(args.input_file) if os.path.isdir(os.path.join(args.input_file, d)) and d.isdigit()]
        numbered_dirs.sort(key=int)  # Sort numerically
        
        print(f"Found {len(numbered_dirs)} directories to process")
        
        for dir_num in numbered_dirs:
            results_file = os.path.join(args.input_file, dir_num, "results.json")
            if not os.path.exists(results_file):
                print(f"Warning: No results.json found in {dir_num}")
                continue
                
            print(f"\nProcessing directory {dir_num}...")
            
            # Load the results
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Check if data is a list (single conversation) or dict (multiple conversations)
            if isinstance(data, list):
                data = {"conversation_0": data}
            elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                pass
            else:
                print(f"Warning: Unexpected data format in {dir_num}")
                continue
                
            # Flatten all items for processing
            all_items = []
            for conversation_id, items in data.items():
                for item in items:
                    all_items.append((conversation_id, item))
                    
            # Process items with ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_item = {
                    executor.submit(process_item, item): (conversation_id, item) 
                    for conversation_id, item in all_items
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item)):
                    conversation_id, original_item = future_to_item[future]
                    processed_result = future.result()
                    with results_lock:
                        all_results.append(processed_result)
    else:
        # Single file processing
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if data is a list (single conversation) or dict (multiple conversations)
        if isinstance(data, list):
            data = {"conversation_0": data}
        elif isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
            pass
        else:
            raise ValueError("Unexpected data format. Expected list or dict of lists.")

        # Flatten all items for processing
        all_items = []
        for conversation_id, items in data.items():
            for item in items:
                all_items.append((conversation_id, item))

        print(f"Processing {len(all_items)} items across {len(data)} conversations...")

        # Use ThreadPoolExecutor with specified workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_item = {
                executor.submit(process_item, item): (conversation_id, item) 
                for conversation_id, item in all_items
            }

            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item)):
                conversation_id, original_item = future_to_item[future]
                processed_result = future.result()
                with results_lock:
                    all_results.append(processed_result)

    # Calculate overall statistics
    all_bleu_scores = [item["bleu_score"] for item in all_results]
    all_f1_scores = [item["f1_score"] for item in all_results]
    all_llm_scores = [item["llm_score"] for item in all_results]

    # Overall statistics
    overall_stats = {
        "total_questions": len(all_results),
        "overall_avg_bleu_score": sum(all_bleu_scores) / len(all_bleu_scores) if all_bleu_scores else 0,
        "overall_avg_f1_score": sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0,
        "overall_avg_llm_score": sum(all_llm_scores) / len(all_llm_scores) if all_llm_scores else 0,
    }

    # Calculate category stats if categories exist
    category_stats = calculate_category_stats(all_results)

    # Prepare final output
    final_results = {
        "detailed_results": all_results,
        "overall_summary": overall_stats
    }
    
    # Add category stats only if they exist
    if category_stats is not None:
        final_results["summary_by_category"] = category_stats

    # Save results to JSON file
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {args.output_file}")
    print("\n=== Evaluation Summary ===")
    print(f"Total Questions: {overall_stats['total_questions']}")
    print(f"Overall Average BLEU Score: {overall_stats['overall_avg_bleu_score']:.4f}")
    print(f"Overall Average F1 Score: {overall_stats['overall_avg_f1_score']:.4f}")
    print(f"Overall Average LLM Score: {overall_stats['overall_avg_llm_score']:.4f}")

    # Print category-wise statistics if they exist
    if category_stats is not None:
        CATEGORY_LABEL_MAP = {
            "1": "Single Hop",
            "2": "Multi-Hop",
            "3": "Open Domain",
            "4": "Temporal",
            "5": "Adversarial"
        }

        print("\n=== Category-wise Statistics ===")
        for num_id, category_name in CATEGORY_LABEL_MAP.items():
            stats = category_stats.get(num_id, {
                "num_questions": 0,
                "avg_bleu_score": 0.0,
                "avg_f1_score": 0.0,
                "avg_llm_score": 0.0
            })
            print(f"\n{category_name}:")
            print(f"  Questions: {stats['num_questions']}")
            print(f"  Average BLEU Score: {stats['avg_bleu_score']:.4f}")
            print(f"  Average F1 Score: {stats['avg_f1_score']:.4f}")
            print(f"  Average LLM Score: {stats['avg_llm_score']:.4f}")


if __name__ == "__main__":
    main()