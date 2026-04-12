import argparse
import json
import multiprocessing as mp
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken
from llm_judge import evaluate_llm_judge
from tqdm import tqdm

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def iter_result_files(input_dir: Path, output_file: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.json")):
        if path.resolve() == output_file.resolve():
            continue
        yield path


def normalize_mapping(value: Any, key_field: str) -> Dict[str, Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        normalized: Dict[str, Dict[str, Any]] = {}
        for entry in value:
            if not isinstance(entry, dict):
                continue
            key_value = entry.get(key_field)
            if key_value is not None:
                normalized[str(key_value)] = entry
        return normalized
    return {}


def sum_timings(timings: Dict[str, Any]) -> Tuple[float, int, Dict[str, float], Dict[str, int]]:
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for key in ("add_chunk", "wrap_user_prompt", "answer"):
        values = timings.get(key, {})
        if isinstance(values, list):
            values = {str(idx): val for idx, val in enumerate(values, start=1)}
        if not isinstance(values, dict):
            values = {}
        category_total = 0.0
        category_count = 0
        for value in values.values():
            if isinstance(value, (int, float)):
                category_total += float(value)
                category_count += 1
        totals[key] = category_total
        counts[key] = category_count
    add_chunk_total = totals.get("add_chunk", 0.0)
    add_chunk_count = counts.get("add_chunk", 0)
    return add_chunk_total, add_chunk_count, totals, counts


def extract_credit_cost(response: Dict[str, Any]) -> float:
    stats = response.get("statistics")
    if isinstance(stats, dict):
        usage = stats.get("usage")
        if isinstance(usage, dict):
            cost = usage.get("credit_cost")
            if isinstance(cost, (int, float)):
                return float(cost)

    total = 0.0

    def walk(node: Any) -> None:
        nonlocal total
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "credit_cost" and isinstance(value, (int, float)):
                    total += float(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(response)
    return total


def sorted_record_items(records: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    def sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        key = item[0]
        try:
            return (0, int(key))
        except (TypeError, ValueError):
            return (1, str(key))

    return sorted(records.items(), key=sort_key)


def count_memory_tokens(data: Dict[str, Any], encoding) -> int:
    """Count tokens in memory data using tiktoken encoding.

    For memory types like episodic, semantic, etc., counts tokens from 'summary' and 'details'.
    For core memories, counts tokens from 'value'.
    """
    total_tokens = 0

    memories = data.get("memories", {})
    if not isinstance(memories, dict):
        return 0

    for memory_type, memory_data in memories.items():
        if not isinstance(memory_data, dict):
            continue

        items = memory_data.get("items", [])
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            # For most memory types, count summary and details
            if memory_type != "core":
                summary = item.get("summary", "")
                details = item.get("details", "")

                if isinstance(summary, str):
                    total_tokens += len(encoding.encode(summary))
                if isinstance(details, str):
                    total_tokens += len(encoding.encode(details))
            else:
                # For core memories, count the value field
                value = item.get("value", "")
                if isinstance(value, str):
                    total_tokens += len(encoding.encode(value))

    return total_tokens


def build_judge_tasks(
    records: Dict[str, Dict[str, Any]],
    base_index: int,
    cached_lookup: Dict[Tuple[Any, Any], Dict[str, Any]],
) -> Tuple[List[Optional[Dict[str, Any]]], List[Tuple[int, str, Any, Any, Any, Any, Any]]]:
    results: List[Optional[Dict[str, Any]]] = []
    tasks: List[Tuple[int, str, Any, Any, Any, Any, Any]] = []

    for local_idx, (_, record) in enumerate(sorted_record_items(records)):
        if not isinstance(record, dict):
            results.append(None)
            continue
        sample_id = record.get("sample_id")
        question_id = record.get("question_id", record.get("question_index"))
        cached = cached_lookup.get((sample_id, question_id))
        if cached is not None:
            results.append(cached)
            continue
        question = record.get("question")
        expected_answer = record.get("expected_answer")
        predicted_answer = record.get("predicted_answer")

        if expected_answer is None:
            results.append(
                {
                    "sample_id": record.get("sample_id"),
                    "question_index": record.get("question_index"),
                    "category": record.get("category"),
                    "label": None,
                    "score": None,
                    "question": question,
                    "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer,
                    "skipped_reason": "missing_expected_answer",
                }
            )
            continue

        if predicted_answer is None:
            results.append(
                {
                    "sample_id": record.get("sample_id"),
                    "question_index": record.get("question_index"),
                    "category": record.get("category"),
                    "label": None,
                    "score": None,
                    "question": question,
                    "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer,
                    "skipped_reason": "missing_predicted_answer",
                }
            )
            continue

        results.append(None)
        tasks.append(
            (
                base_index + local_idx,
                sample_id,
                record.get("question_index"),
                record.get("category"),
                question,
                expected_answer,
                predicted_answer,
            )
        )

    return results, tasks


def judge_task(task: Tuple[int, str, Any, Any, Any, Any, Any]) -> Tuple[int, int, str]:
    idx, _sample_id, _question_index, _category, question, expected_answer, predicted_answer = task
    score = evaluate_llm_judge(question, expected_answer, predicted_answer)
    label = "CORRECT" if score == 1 else "WRONG"
    return idx, score, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate eval results and compute LLM-judge accuracy/latency/cost."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to results folder (e.g., results/0124a).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output JSON file (default: <input_dir>/metrics.json).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_file = args.output_file or (input_dir / "metrics.json")

    cached_metrics = load_json(output_file) if output_file.exists() else None
    cached_judge_results = None
    if isinstance(cached_metrics, dict):
        cached_judge_results = cached_metrics.get("llm_judge_results")
        if not isinstance(cached_judge_results, list):
            cached_judge_results = None
    cached_lookup: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    if cached_judge_results is not None:
        for entry in cached_judge_results:
            if not isinstance(entry, dict):
                continue
            sample_id = entry.get("sample_id")
            question_id = entry.get("question_id", entry.get("question_index"))
            if sample_id is None or question_id is None:
                continue
            cached_lookup[(sample_id, question_id)] = entry

    total_latency = 0.0
    total_requests = 0
    total_answer_latency = 0.0
    total_answer_requests = 0
    latency_totals = {"add_chunk": 0.0, "wrap_user_prompt": 0.0, "answer": 0.0}
    latency_counts = {"add_chunk": 0, "wrap_user_prompt": 0, "answer": 0}
    total_cost = 0.0
    total_questions = 0
    total_correct = 0
    total_judged = 0
    judge_results: List[Optional[Dict[str, Any]]] = []
    judge_tasks: List[Tuple[int, str, Any, Any, Any, Any, Any]] = []

    # Initialize tiktoken encoding for gpt-4o-mini
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    total_memory_tokens = 0
    memory_file_count = 0

    sample_files = list(iter_result_files(input_dir, output_file))
    for path in tqdm(sample_files, total=len(sample_files), desc="Processing samples"):
        data = load_json(path)
        if data is None:
            continue

        timings = data.get("timings", {})
        if not isinstance(timings, dict):
            timings = {}
        latency, requests, totals, counts = sum_timings(timings)
        total_latency += latency
        total_requests += requests
        total_answer_latency += totals.get("answer", 0.0)
        total_answer_requests += counts.get("answer", 0)
        for key in latency_totals:
            latency_totals[key] += totals.get(key, 0.0)
            latency_counts[key] += counts.get(key, 0)

        responses = normalize_mapping(data.get("responses", {}), "chunk_index")
        for response in responses.values():
            if not isinstance(response, dict):
                continue
            payload = response.get("response")
            if isinstance(payload, dict):
                total_cost += extract_credit_cost(payload)

        records = normalize_mapping(data.get("records", {}), "question_index")
        total_questions += sum(
            1
            for record in records.values()
            if isinstance(record, dict) and record.get("expected_answer") is not None
        )
        results, tasks = build_judge_tasks(
            records,
            base_index=len(judge_results),
            cached_lookup=cached_lookup,
        )
        judge_results.extend(results)
        judge_tasks.extend(tasks)

    if judge_tasks:
        task_by_index = {task[0]: task for task in judge_tasks}
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=16) as pool:
            for idx, score, label in tqdm(
                pool.imap(judge_task, judge_tasks),
                total=len(judge_tasks),
                desc="Evaluating records",
            ):
                task = task_by_index[idx]
                _, sample_id, question_index, category, question, expected_answer, predicted_answer = task
                judge_results[idx] = {
                    "sample_id": sample_id,
                    "question_index": question_index,
                    "category": category,
                    "label": label,
                    "score": score,
                    "question": question,
                    "expected_answer": expected_answer,
                    "predicted_answer": predicted_answer,
                }
                if category != 5:
                    total_correct += score
                    total_judged += 1

    # Process memory files to count tokens
    memory_files = list(sorted(input_dir.glob("*_memories.json")))
    for path in tqdm(memory_files, total=len(memory_files), desc="Processing memory files"):
        data = load_json(path)
        if data is None:
            continue

        tokens = count_memory_tokens(data, encoding)
        total_memory_tokens += tokens
        memory_file_count += 1

    finalized_results = [result for result in judge_results if result is not None]
    total_correct = 0
    total_judged = 0
    for result in finalized_results:
        if isinstance(result, dict) and result.get("category") != 5:
            score = result.get("score")
            if isinstance(score, (int, float)):
                total_correct += int(score)
                total_judged += 1

    accuracy = (total_correct / total_judged) if total_judged else None
    average_latency = (total_latency / total_requests) if total_requests else None
    average_answer_latency = (
        total_answer_latency / total_answer_requests if total_answer_requests else None
    )

    category_metrics: Dict[str, Dict[str, Any]] = {}
    for result in finalized_results:
        if not isinstance(result, dict):
            continue
        category = result.get("category")
        if category is None or category == 5:
            continue
        key = str(category)
        entry = category_metrics.setdefault(
            key, {"category": category, "total_judged": 0, "total_correct": 0, "accuracy": None}
        )
        score = result.get("score")
        if isinstance(score, (int, float)):
            entry["total_judged"] += 1
            entry["total_correct"] += int(score)

    for entry in category_metrics.values():
        judged = entry.get("total_judged", 0)
        entry["accuracy"] = (entry["total_correct"] / judged) if judged else None

    average_memory_tokens = (total_memory_tokens / memory_file_count) if memory_file_count else None

    output = {
        "input_dir": str(input_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "total_questions": total_questions,
            "total_judged": total_judged,
            "total_correct": total_correct,
            "accuracy": accuracy,
            "total_requests": total_requests,
            "total_answer_requests": total_answer_requests,
            "average_latency_per_request_seconds": average_latency,
            "average_answer_latency_seconds": average_answer_latency,
            "total_latency_seconds": total_latency,
            "total_cost": total_cost,
            "total_memory_tokens": total_memory_tokens,
            "memory_file_count": memory_file_count,
            "average_memory_tokens": average_memory_tokens,
        },
        "accuracy_by_category": category_metrics,
        "latency_breakdown_seconds": latency_totals,
        "request_counts": latency_counts,
        "llm_judge_results": finalized_results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
