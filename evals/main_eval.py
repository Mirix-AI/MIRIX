import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

from mirix_memory_system import MirixMemorySystem
from task_agent import TaskAgent


def trigger_dream(
    base_url: str,
    client_id: str,
    user_id: str,
    version: str = "v1",
    model: str = "gpt-4.1-mini",
    raw_sessions: Optional[List[str]] = None,
    memory_types: Optional[List[str]] = None,
    temperature: Optional[float] = None,
) -> Dict:
    endpoint = "/memory/auto_dream_v2" if version == "v2" else "/memory/auto_dream"
    body: Dict = {"model": model}
    if raw_sessions:
        body["raw_sessions"] = raw_sessions
    if memory_types:
        body["memory_types"] = memory_types
    if temperature is not None:
        body["temperature"] = temperature
    try:
        resp = requests.post(
            f"{base_url}{endpoint}",
            headers={"Content-Type": "application/json", "X-Client-Id": client_id},
            params={"user_id": user_id},
            json=body,
            timeout=600,
        )
        if resp.status_code >= 400:
            return {"error": resp.status_code, "body": resp.text}
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


instructions = """Instructions:

1. Carefully analyze all utterances from both speakers.
2. The conversation has a timestamp, but the events mentioned in the conversation may have different timestamps. You have to extract the exact date of the mentioned events. Remember that "mentioned at" is not the same as "occurred at" so this has to be noted in the memories.
3. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
4. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp.
5. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
6. You are supposed to extract the event/fact/semantic knowledge from the conversation. For example, if the conversation happens at 2023 and the conversation says that "John went to India last year", then you should save the fact that "John went to India in 2022". Similarly for all other kinds of memories.
7. Make sure to extract the facts about the characters, such as their name, age, gender, occupation, hometown, etc."""

def load_locomo(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    for key in ("data", "items", "examples", "records"):
        if key in data and isinstance(data[key], list):
            return data[key]
    raise ValueError(f"Unsupported dataset format in {path}")


def iter_sessions(conversation: Dict) -> Iterable[Dict]:
    session_numbers = []
    for key, value in conversation.items():
        match = re.match(r"^session_(\d+)$", key)
        if match and isinstance(value, list):
            session_numbers.append(int(match.group(1)))
    for number in sorted(session_numbers):
        yield {
            "number": number,
            "date_time": conversation.get(f"session_{number}_date_time"),
            "turns": conversation.get(f"session_{number}", []),
        }


def format_session_chunk(session: Dict, date_time: str) -> str:

    header = f"Session {session['number']}"
    if session.get("date_time"):
        header += f" ({session['date_time']})"
    lines = [f"You have access to the conversation between two speakers. The conversation is timestamped at {date_time}.\n"]
    lines.append(instructions)
    lines.append(header)
    for turn in session.get("turns", []):
        speaker = turn.get("speaker", "").strip()
        text = turn.get("text", "").strip()
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

def load_sample_result(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def save_sample_result(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def normalize_sample_result(sample_result: Dict) -> Dict:
    timings = sample_result.setdefault(
        "timings",
        {"add_chunk": {}, "wrap_user_prompt": {}, "answer": {}},
    )
    for key in ("add_chunk", "wrap_user_prompt", "answer"):
        if isinstance(timings.get(key), list):
            timings[key] = {
                str(idx): value for idx, value in enumerate(timings[key], start=1)
            }
        elif not isinstance(timings.get(key), dict):
            timings[key] = {}

    responses = sample_result.setdefault("responses", {})
    if isinstance(responses, list):
        responses_dict: Dict[str, Dict] = {}
        for entry in responses:
            if isinstance(entry, dict):
                chunk_index = entry.get("chunk_index")
                if chunk_index is not None:
                    responses_dict[str(chunk_index)] = entry
        sample_result["responses"] = responses_dict
    elif not isinstance(responses, dict):
        sample_result["responses"] = {}

    records = sample_result.setdefault("records", {})
    if isinstance(records, list):
        records_dict: Dict[str, Dict] = {}
        for entry in records:
            if isinstance(entry, dict):
                qidx = entry.get("question_index")
                if qidx is not None:
                    records_dict[str(qidx)] = entry
        sample_result["records"] = records_dict
    elif not isinstance(records, dict):
        sample_result["records"] = {}

    return sample_result


def print_qa(qidx: int, question: str, expected: Optional[str], predicted: Optional[str]) -> None:
    print(f"[{qidx}] question: {question}")
    print(f"[{qidx}] expected_answer: {expected}")
    print(f"[{qidx}] predicted_answer: {predicted}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Mirix memory on LoCoMo.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/locomo10.json"),
        help="Path to LoCoMo dataset JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit number of questions per sample.",
    )
    parser.add_argument(
        "--run-llm",
        action="store_true",
        default=True,
        help="Call the LLM to answer questions.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results"),
        help="Output folder for per-sample JSON results.",
    )
    parser.add_argument(
        "--mirix_config_path",
        type=Path,
        default=None,
        help="Path to Mirix config file.",
    )
    parser.add_argument(
        "--autodream-every",
        type=int,
        default=None,
        help="Trigger auto-dream every N sessions (None = disabled).",
    )
    parser.add_argument(
        "--dream-version",
        choices=["v1", "v2"],
        default="v1",
        help="v1=CRUD-based auto_dream, v2=region-rewriting auto_dream_v2.",
    )
    parser.add_argument(
        "--dream-model",
        default="gpt-4.1-mini",
        help="LLM model used for dreaming.",
    )
    parser.add_argument(
        "--dream-temperature",
        type=float,
        default=None,
        help="Override dream LLM temperature (e.g. 0.0 for deterministic). Default = server default (0.7).",
    )
    parser.add_argument(
        "--cheat-first-dream",
        action="store_true",
        help=(
            "On the FIRST dream of each conversation, also send the raw session "
            "texts in the dream window to the dream agent (memories are usually "
            "sparse on the first dream, so the agent benefits from seeing source dialogue)."
        ),
    )
    parser.add_argument(
        "--dream-memory-types",
        type=str,
        default=None,
        help=(
            "Comma-separated list of memory types to dream on. "
            "Default = server side default (all 5 types). "
            "Example: 'episodic' or 'episodic,semantic'. "
            "Valid types: episodic, semantic, procedural, resource, knowledge_vault."
        ),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8531",
        help="MIRIX server base URL.",
    )
    args = parser.parse_args()

    items = load_locomo(args.data)
    if args.limit is not None:
        items = items[: args.limit]

    mirix_client_id = os.environ.get("MIRIX_CLIENT_ID", "mirix-eval-client")
    mirix_org_id = os.environ.get("MIRIX_ORG_ID", "mirix-eval-org")

    output_path = args.output_path
    output_path.mkdir(parents=True, exist_ok=True)

    user_id_prefix = os.environ.get("MIRIX_USER_ID_PREFIX", "")

    for item in items:
        sample_id = item.get("sample_id")
        if sample_id is None:
            continue
        effective_user_id = f"{user_id_prefix}{sample_id}" if user_id_prefix else sample_id
        sample_path = output_path / f"{sample_id}.json"

        task_agent = TaskAgent(mirix_config_path=str(args.mirix_config_path), client_id=mirix_client_id, org_id=mirix_org_id, user_id=effective_user_id) if args.run_llm else None

        sample_result = load_sample_result(sample_path)
        if sample_result is None:
            sample_result = {
                "sample_id": sample_id,
                "timings": {"add_chunk": {}, "wrap_user_prompt": {}, "answer": {}},
                "responses": {},
                "records": {},
            }

        sample_result.setdefault("sample_id", sample_id)
        sample_result = normalize_sample_result(sample_result)

        memory_system = MirixMemorySystem(user_id=effective_user_id,
                    mirix_config_path=str(args.mirix_config_path),
                    client=task_agent.mirix_client)

        conversation = item.get("conversation", {})
        dream_count = 0
        sessions = list(iter_sessions(conversation))
        pending_raw_sessions: List[str] = []  # accumulated raw text since last dream
        for idx, session in enumerate(sessions, start=1):
            idx_key = str(idx)
            if idx_key in sample_result["responses"]:
                # Session already loaded; still need to count for dream tracking
                if args.autodream_every and idx % args.autodream_every == 0:
                    dream_key = f"dream_{dream_count + 1}"
                    if dream_key in sample_result.get("dreams", {}):
                        dream_count += 1
                continue
            date_time_key = f"session_{idx}_date_time"
            date_time = conversation.get(date_time_key)
            if date_time is None:
                date_time = conversation.get(f"session_{idx + 1}_date_time")
            chunk = format_session_chunk(session, date_time=date_time)

            start = time.perf_counter()

            response = memory_system.add_chunk(chunk, raw_input=chunk)

            elapsed = time.perf_counter() - start

            sample_result["responses"][idx_key] = {
                "type": "add_chunk",
                "chunk_index": idx,
                "question_index": None,
                "response": response,
            }
            sample_result["timings"]["add_chunk"][idx_key] = elapsed
            pending_raw_sessions.append(chunk)

            is_last = (idx == len(sessions))
            if args.autodream_every and (idx % args.autodream_every == 0 or is_last):
                dream_count += 1
                dream_key = f"dream_{dream_count}"
                sample_result.setdefault("dreams", {})
                # Cheating: only the first dream of this conversation receives raw text.
                raw_for_dream = (
                    list(pending_raw_sessions)
                    if (args.cheat_first_dream and dream_count == 1)
                    else None
                )
                print(
                    f"  >> DREAM #{dream_count} after session {idx} (version={args.dream_version}"
                    f"{', cheat=raw' if raw_for_dream else ''})",
                    flush=True,
                )
                t0 = time.perf_counter()
                dream_memory_types = (
                    [t.strip() for t in args.dream_memory_types.split(",") if t.strip()]
                    if args.dream_memory_types
                    else None
                )
                dream_result = trigger_dream(
                    base_url=args.base_url,
                    client_id=mirix_client_id,
                    user_id=effective_user_id,
                    version=args.dream_version,
                    model=args.dream_model,
                    raw_sessions=raw_for_dream,
                    memory_types=dream_memory_types,
                    temperature=args.dream_temperature,
                )
                dream_elapsed = time.perf_counter() - t0
                print(f"     done in {dream_elapsed:.1f}s", flush=True)
                sample_result["dreams"][dream_key] = {
                    "after_session": idx,
                    "elapsed_s": round(dream_elapsed, 1),
                    "cheated": bool(raw_for_dream),
                    "result": dream_result,
                }
                pending_raw_sessions = []  # reset window

            save_sample_result(sample_path, sample_result)

        qa_list = item.get("qa", [])
        if args.max_questions is not None:
            qa_list = qa_list[: args.max_questions]

        for qidx, qa in enumerate(qa_list, start=1):
            qidx_key = str(qidx)
            if qidx_key in sample_result["records"]:
                record = sample_result["records"][qidx_key]
                print_qa(
                    qidx,
                    record.get("question", ""),
                    record.get("expected_answer"),
                    record.get("predicted_answer"),
                )
                continue
            question = qa.get("question", "")
            expected_answer = qa.get("answer")
            if expected_answer is None:
                record = {
                    "sample_id": sample_id,
                    "question_index": qidx,
                    "question": question,
                    "expected_answer": expected_answer,
                    "evidence": qa.get("evidence"),
                    "category": qa.get("category"),
                    "prompt": None,
                    "predicted_answer": None,
                    "messages": None,
                    "usage": None,
                    "usage_total": None,
                }
                sample_result["records"][qidx_key] = record
                save_sample_result(sample_path, sample_result)
                print_qa(qidx, question, expected_answer, None)
                continue
            start = time.perf_counter()
            input_messages = memory_system.wrap_user_prompt(question)
            sample_result["timings"]["wrap_user_prompt"][qidx_key] = (
                time.perf_counter() - start
            )
            retrieval_usage = getattr(memory_system, "_last_topic_extraction_usage", None)
            predicted = None
            message_trace = None
            usage_trace = None
            usage_total = None
            retrieved = None
            if task_agent:
                start = time.perf_counter()
                trace = task_agent.answer(input_messages, user_id=effective_user_id)
                predicted = trace.get("answer")
                message_trace = trace.get("messages")
                usage_trace = trace.get("usage")
                usage_total = trace.get("usage_total")
                retrieved = trace.get("retrieved")
                sample_result["timings"]["answer"][qidx_key] = (
                    time.perf_counter() - start
                )

            record = {
                "sample_id": sample_id,
                "question_index": qidx,
                "question": question,
                "expected_answer": expected_answer,
                "evidence": qa.get("evidence"),
                "category": qa.get("category"),
                "input_messages": input_messages,
                "predicted_answer": predicted,
                "messages": message_trace,
                "usage": usage_trace,
                "usage_total": usage_total,
                "retrieval_usage": retrieval_usage,
                "retrieved": retrieved,
                "retrieved_count": len(retrieved) if retrieved else 0,
            }
            sample_result["records"][qidx_key] = record
            save_sample_result(sample_path, sample_result)
            print_qa(qidx, question, expected_answer, predicted)

        # -- aggregate token usage across QA and dream calls --
        def _sum_usage(u) -> dict:
            """Accept a single usage dict or a list of usage dicts."""
            if isinstance(u, dict):
                return u
            if isinstance(u, list):
                out = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                for item in u:
                    if isinstance(item, dict):
                        out["prompt_tokens"]     += item.get("prompt_tokens", 0)
                        out["completion_tokens"] += item.get("completion_tokens", 0)
                        out["total_tokens"]      += item.get("total_tokens", 0)
                return out
            return {}

        qa_tokens = {"prompt": 0, "completion": 0, "total": 0}
        for rec in sample_result.get("records", {}).values():
            u = _sum_usage(rec.get("usage") or {})
            qa_tokens["prompt"]     += u.get("prompt_tokens", 0)
            qa_tokens["completion"] += u.get("completion_tokens", 0)
            qa_tokens["total"]      += u.get("total_tokens", 0)

        dream_tokens = {"prompt": 0, "completion": 0, "total": 0}
        for d in sample_result.get("dreams", {}).values():
            u = (d.get("result") or {}).get("usage") or {}
            dream_tokens["prompt"]     += u.get("prompt_tokens", 0)
            dream_tokens["completion"] += u.get("completion_tokens", 0)
            dream_tokens["total"]      += u.get("total_tokens", 0)

        ingestion_tokens = {"prompt": 0, "completion": 0, "total": 0}
        for r in sample_result.get("responses", {}).values():
            u = (r.get("response") or {}).get("usage") or {}
            ingestion_tokens["prompt"]     += u.get("prompt_tokens", 0)
            ingestion_tokens["completion"] += u.get("completion_tokens", 0)
            ingestion_tokens["total"]      += u.get("total_tokens", 0)

        # Server-side topic_extraction LLM cost (one call per QA question via
        # /memory/retrieve/conversation). Same overhead for baseline vs autodream
        # but needed for accurate absolute token-efficiency numbers.
        retrieval_tokens = {"prompt": 0, "completion": 0, "total": 0}
        for rec in sample_result.get("records", {}).values():
            u = rec.get("retrieval_usage") or {}
            retrieval_tokens["prompt"]     += u.get("prompt_tokens", 0)
            retrieval_tokens["completion"] += u.get("completion_tokens", 0)
            retrieval_tokens["total"]      += u.get("total_tokens", 0)

        sample_result["token_summary"] = {
            "ingestion": ingestion_tokens,
            "qa": qa_tokens,
            "retrieval": retrieval_tokens,
            "dream": dream_tokens,
            "grand_total": {
                "prompt":     ingestion_tokens["prompt"]     + qa_tokens["prompt"]     + retrieval_tokens["prompt"]     + dream_tokens["prompt"],
                "completion": ingestion_tokens["completion"] + qa_tokens["completion"] + retrieval_tokens["completion"] + dream_tokens["completion"],
                "total":      ingestion_tokens["total"]      + qa_tokens["total"]      + retrieval_tokens["total"]      + dream_tokens["total"],
            },
        }

        # -- retrieval coverage summary --
        questions_with_retrieval = 0
        total_retrieved = 0
        unique_ids = set()
        for rec in sample_result.get("records", {}).values():
            n = rec.get("retrieved_count", 0) or 0
            if n > 0:
                questions_with_retrieval += 1
                total_retrieved += n
            for item in (rec.get("retrieved") or []):
                rid = item.get("id") if isinstance(item, dict) else None
                if rid:
                    unique_ids.add(rid)
        n_q = len(sample_result.get("records", {}))
        sample_result["retrieval_summary"] = {
            "questions": n_q,
            "questions_with_retrieval": questions_with_retrieval,
            "questions_with_retrieval_ratio": (questions_with_retrieval / n_q) if n_q else 0.0,
            "total_retrieved": total_retrieved,
            "avg_retrieved_per_question": (total_retrieved / n_q) if n_q else 0.0,
            "unique_memory_ids_seen": len(unique_ids),
        }
        save_sample_result(sample_path, sample_result)

        try:
            all_memories = memory_system.list_all_memories(limit=0)
        except Exception as exc:
            all_memories = {
                "success": False,
                "error": str(exc),
                "user_id": sample_id,
            }

        memories_path = output_path / f"{sample_id}_memories.json"
        with memories_path.open("w", encoding="utf-8") as handle:
            json.dump(all_memories, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
