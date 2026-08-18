import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from mirix_memory_system import MirixMemorySystem
from task_agent import TaskAgent
from v720_policy import (
    format_visual_evidence,
    is_v720,
    mentioned_at_iso,
    needs_selective_ocr,
    relative_time_annotations,
)


instructions = """Instructions:

1. Carefully analyze all utterances from both speakers.
2. The conversation has a timestamp, but the events mentioned in the conversation may have different timestamps. You have to extract the exact date of the mentioned events. Remember that "mentioned at" is not the same as "occurred at" so this has to be noted in the memories.
3. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
4. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the conversation timestamp.
5. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
6. You are supposed to extract the event/fact/semantic knowledge from the conversation. For example, if the conversation happens at 2023 and the conversation says that "John went to India last year", then you should save the fact that "John went to India in 2022". Similarly for all other kinds of memories.
7. Make sure to extract the facts about the characters, such as their name, age, gender, occupation, hometown, etc."""


def dream_source_chunk_ids(after_idx: int, dream_every: int) -> list[int]:
    """Return zero-based source chunk ids for the dream ending at ``after_idx``."""
    if dream_every <= 0 or after_idx <= 0:
        return []
    batch_start = ((after_idx - 1) // dream_every) * dream_every + 1
    return list(range(batch_start - 1, after_idx))

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
    # MIRIX_WINDOW_TURNS: hand the extractor a few turns at a time instead of a whole
    # session.
    #
    # A session is thirty-odd turns and arrives in ONE add_chunk call, so the extractor
    # picks a handful of memories to stand for all of it, and what loses is the specific
    # noun. Traced on three of the 60 never-written errors:
    #     "Hey Jo, guess what I did? Dyed my hair last week"
    #         -> "Nate dyed his hair purple last week"      the nickname is gone
    #     "I'm reading 'The Lean Startup' hoping it'll give me tips for my biz"
    #         -> "Jon is wrapping up a business plan"       the title is gone
    # Measured over three conversations, 17-46% of the source's distinctive tokens (quoted
    # titles, proper nouns, numbers) never reach the store.
    #
    # Windows overlap by one turn so a fact stated across a turn boundary is not split, and
    # each window keeps its session number and date_time — the resolver needs the session
    # date to turn "next month" into a date, and occurred_at is 100% populated today.
    #
    # This was tried before and abandoned: evals/_chunking.py records fine-grained chunking
    # scoring 38/60 against ~30 canonical, dropped for leaderboard comparability rather than
    # for accuracy. Off by default for exactly that reason.
    window = int(os.environ.get("MIRIX_WINDOW_TURNS", "0") or 0)
    overlap = 1 if window > 1 else 0
    for number in sorted(session_numbers):
        turns = conversation.get(f"session_{number}", [])
        date_time = conversation.get(f"session_{number}_date_time")
        if window <= 0 or len(turns) <= window:
            yield {"number": number, "date_time": date_time, "turns": turns}
            continue
        start = 0
        while start < len(turns):
            yield {
                "number": number,
                "date_time": date_time,
                "turns": turns[start:start + window],
            }
            if start + window >= len(turns):
                break
            start += window - overlap


def format_session_chunk(
    session: Dict,
    date_time: str,
    *,
    v720: bool = False,
    visual_evidence: Optional[Dict[str, str]] = None,
) -> str:

    header = f"Session {session['number']}"
    if session.get("date_time"):
        header += f" ({session['date_time']})"
    lines = [f"You have access to the conversation between two speakers. The conversation is timestamped at {date_time}.\n"]
    lines.append(instructions)
    if v720:
        lines.append(
            "v7.20 temporal/provenance rules:\n"
            "- `mentioned_at` is the timestamp of this conversation session.\n"
            "- `event_time` is when a described event actually happened. Store it as the "
            "episodic occurred_at; never replace it with mentioned_at.\n"
            "- Preserve the speaker's exact relative-time phrase in details and also use "
            "the supplied event_time_hint when it is present.\n"
            "- BLIP and selective OCR/vision lines are first-class visual observations. "
            "Image retrieval metadata is only a search hint.\n"
            f"mentioned_at={mentioned_at_iso(date_time) or date_time}"
        )
    lines.append(header)
    for turn in session.get("turns", []):
        speaker = turn.get("speaker", "").strip()
        text = turn.get("text", "").strip()
        dia_id = str(turn.get("dia_id") or "").strip()
        label = f"{speaker} [{dia_id}]" if v720 and dia_id else speaker
        lines.append(f"{label}: {text}")
        if v720:
            for annotation in relative_time_annotations(text, date_time):
                lines.append(
                    "[Temporal evidence " + (dia_id or "unknown") + "]: "
                    f"relative_phrase={annotation['phrase']!r}; "
                    f"event_time_hint={annotation['event_time_hint']}; "
                    f"mentioned_at={mentioned_at_iso(date_time) or date_time}"
                )
            vision_text = (visual_evidence or {}).get(dia_id)
            lines.extend(format_visual_evidence(turn, vision_text))
    return "\n".join(lines)


def extract_selective_visual_evidence(turn: Dict, task_agent: Optional[TaskAgent]) -> str:
    """Use vision only for likely text-bearing images; BLIP covers all other images."""

    if os.environ.get("MIRIX_DISABLE_SELECTIVE_OCR") == "1":
        return ""
    if task_agent is None or not needs_selective_ocr(turn):
        return ""
    urls = turn.get("img_url") or []
    if isinstance(urls, str):
        urls = [urls]
    url = next((str(value) for value in urls if value), "")
    if not url:
        return ""
    model = os.environ.get("MIRIX_V720_VISION_MODEL", task_agent.model)
    try:
        response = task_agent.client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Inspect this conversation image as raw evidence. First transcribe "
                            "all visible text exactly (including title/sign wording). Then give "
                            "one precise factual description. Do not infer facts not visible."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
                ],
            }],
            max_completion_tokens=180,
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[main_eval] v7.20 selective OCR failed for {turn.get('dia_id')}: {exc}",
            flush=True,
        )
        return ""

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
        "--question-indices",
        type=str,
        default=None,
        help="Optional comma-separated, one-based question indices to evaluate.",
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
        default=Path("locomo_run"),
        help=(
            "Output sub-folder name. The path is resolved relative to "
            "<repo>/evals/results/locomo/, so passing 'foo' writes to "
            "evals/results/locomo/foo. Absolute paths are still honored "
            "but warned about, since they bypass the locomo namespace."
        ),
    )
    parser.add_argument(
        "--mirix_config_path",
        type=Path,
        default=None,
        help="Path to Mirix config file.",
    )
    args = parser.parse_args()

    items = load_locomo(args.data)
    if args.limit is not None:
        items = items[: args.limit]

    mirix_client_id = os.environ.get("MIRIX_CLIENT_ID", "mirix-eval-client")
    mirix_org_id = os.environ.get("MIRIX_ORG_ID", "mirix-eval-org")
    # Optional storage-only namespace for side-by-side graph evaluations.  The
    # public sample_id and result filenames remain unchanged, while PG/Neo4j
    # user ownership is isolated so a fresh run cannot contaminate an existing
    # benchmark graph with the same LoCoMo conversation ids.
    eval_user_prefix = os.environ.get("MIRIX_EVAL_USER_PREFIX", "")
    v720_enabled = is_v720(os.environ.get("MIRIX_GRAPH_VERSION"))
    if v720_enabled:
        graph_version = os.environ.get("MIRIX_GRAPH_VERSION", "v7.20")
        print(
            f"[main_eval] {graph_version} policy enabled: multimodal evidence + temporal provenance + compact QA retrieval",
            flush=True,
        )
    qa_only_existing_store = os.environ.get("MIRIX_QA_ONLY_EXISTING_STORE") == "1"
    if qa_only_existing_store:
        print(
            "[main_eval] QA-only mode: reusing existing PG/Neo4j state; ingest and Dream are skipped",
            flush=True,
        )

    # Force every main_eval run into the LoCoMo namespace so MAB and LoCoMo
    # outputs cannot bleed into each other. The user can still pass an
    # absolute path to break out (e.g. for one-off experiments), but a warning
    # makes the divergence explicit.
    locomo_root = Path(__file__).resolve().parent / "results" / "locomo"
    if args.output_path.is_absolute():
        print(
            f"[main_eval] WARNING: --output_path is absolute ({args.output_path}); "
            f"writing outside evals/results/locomo/ namespace.",
        )
        output_path = args.output_path
    else:
        output_path = locomo_root / args.output_path
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[main_eval] writing per-sample results to {output_path}")

    # Server-side token tracker is always-on (see mirix/database/token_tracker.py).
    # We just need to (a) reset before each sample's ingest, (b) snapshot after
    # ingest to get "build" tokens, (c) snapshot after QA to get "query" tokens.
    import httpx
    server_base = "http://127.0.0.1:8531"
    def _reset_tokens():
        try:
            httpx.post(f"{server_base}/debug/token_stats/reset", timeout=10)
        except Exception:
            pass
    def _snapshot_tokens():
        try:
            r = httpx.get(f"{server_base}/debug/token_stats", timeout=10)
            return r.json().get("stats", {})
        except Exception:
            return {}
    def _sum_tokens(stats):
        s = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
        for v in stats.values():
            for k in s: s[k] += v.get(k, 0)
        return s

    for item in items:
        sample_id = item.get("sample_id")
        if sample_id is None:
            continue
        storage_user_id = f"{eval_user_prefix}{sample_id}"
        sample_path = output_path / f"{sample_id}.json"

        task_agent = TaskAgent(
            mirix_config_path=str(args.mirix_config_path),
            client_id=mirix_client_id,
            org_id=mirix_org_id,
            user_id=storage_user_id,
            model=os.environ.get("MIRIX_QA_MODEL", "gpt-4.1-mini"),
        ) if args.run_llm else None

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
        if v720_enabled:
            sample_result.setdefault("visual_evidence", {})

        memory_system = MirixMemorySystem(user_id=storage_user_id,
                    mirix_config_path=str(args.mirix_config_path),
                    client=task_agent.mirix_client)

        # Reset server-side token counter so build_tokens reflects only this sample's ingest
        _reset_tokens()

        # Interleaved consolidation (MIRIX_DREAM_EVERY_N_CHUNKS=N): fire one
        # auto_dream cycle after every Nth ingested chunk, plus one final cycle
        # after the last chunk if the total is not a multiple of N. This models
        # the ONLINE periodic-reconsolidation design (consolidate while
        # ingesting), not a one-shot post-hoc dream on a finished store.
        dream_every = int(os.environ.get("MIRIX_DREAM_EVERY_N_CHUNKS", "0") or 0)

        def _fire_dream(after_idx: int) -> None:
            dream_key = f"dream_{after_idx}"
            if dream_key in sample_result["responses"]:
                return
            # Source metadata uses zero-based chunk ids.  The batch starts after the
            # previous N-boundary, so a 19-chunk run with N=5 yields [15,16,17,18]
            # for the final (four-chunk) dream instead of accidentally overlapping
            # chunk 15 again.
            source_chunk_ids = dream_source_chunk_ids(after_idx, dream_every)
            start = time.perf_counter()
            try:
                r = httpx.post(
                    f"{server_base}/memory/auto_dream",
                    params={"user_id": storage_user_id},
                    headers={"x-client-id": mirix_client_id, "x-org-id": mirix_org_id},
                    json={
                        "mode": os.environ.get("MIRIX_DREAM_MODE", "experience"),
                        # Eval AutoDream arms measure graph consolidation only. Keep
                        # PG rows byte-stable so retrieval changes can be attributed
                        # to the graph instead of an LLM rewrite of flat memories.
                        "graph_only": True,
                        # v7.14+ resolves the semantic-memory delta from exactly this
                        # batch. Older versions ignore the field and retain their
                        # historical full-graph behaviour.
                        "source_chunk_ids": source_chunk_ids,
                        # Hybrid Dream revisions keep intermediate cycles local and
                        # perform their one full semantic sweep only after ingest is
                        # complete.  Do not infer finality from a hard-coded chunk id.
                        "final_full_graph": after_idx == total_chunks,
                    },
                    timeout=3000,
                )
                payload = r.json()
            except Exception as exc:  # noqa: BLE001
                payload = {"error": str(exc)}
            elapsed = time.perf_counter() - start
            print(f"[main_eval] auto_dream after chunk {after_idx}: "
                  f"{elapsed:.0f}s {str(payload)[:160]}", flush=True)
            sample_result["responses"][dream_key] = {
                "type": "auto_dream",
                "chunk_index": after_idx,
                "source_chunk_ids": source_chunk_ids,
                "final_full_graph": after_idx == total_chunks,
                "question_index": None,
                "response": payload,
                "elapsed_seconds": elapsed,
            }
            save_sample_result(sample_path, sample_result)

        conversation = item.get("conversation", {})
        total_chunks = sum(1 for _ in iter_sessions(conversation))
        if qa_only_existing_store:
            for idx in range(1, total_chunks + 1):
                sample_result["responses"].setdefault(str(idx), {
                    "type": "qa_only_existing_store",
                    "chunk_index": idx,
                    "question_index": None,
                    "response": {"skipped": "existing_store"},
                })
        for idx, session in enumerate(iter_sessions(conversation), start=1):
            idx_key = str(idx)
            if idx_key in sample_result["responses"]:
                continue
            date_time_key = f"session_{idx}_date_time"
            date_time = conversation.get(date_time_key)
            if date_time is None:
                date_time = conversation.get(f"session_{idx + 1}_date_time")
            session_visual: Dict[str, str] = {}
            if v720_enabled:
                cached_visual = sample_result.setdefault("visual_evidence", {})
                for turn in session.get("turns", []):
                    dia_id = str(turn.get("dia_id") or "")
                    cached = cached_visual.get(dia_id)
                    if isinstance(cached, str) and cached:
                        session_visual[dia_id] = cached
                        continue
                    extracted = extract_selective_visual_evidence(turn, task_agent)
                    if extracted:
                        cached_visual[dia_id] = extracted
                        session_visual[dia_id] = extracted
                save_sample_result(sample_path, sample_result)

            chunk = format_session_chunk(
                session,
                date_time=date_time,
                v720=v720_enabled,
                visual_evidence=session_visual,
            )

            start = time.perf_counter()

            source_meta = None
            if v720_enabled:
                mentioned_at = mentioned_at_iso(date_time)
                source_meta = {
                    # ``occurred_at`` remains the compatibility ordering field for
                    # source provenance. It is the session/mention time here; the
                    # episodic row's occurred_at is independently extracted event_time.
                    "occurred_at": mentioned_at or str(date_time),
                    "mentioned_at": mentioned_at or str(date_time),
                    "temporal_role": "mentioned_at",
                    "session_id": session.get("number"),
                }
            response = memory_system.add_chunk(
                chunk,
                raw_input=chunk,
                source_meta=source_meta,
            )

            elapsed = time.perf_counter() - start

            sample_result["responses"][idx_key] = {
                "type": "add_chunk",
                "chunk_index": idx,
                "question_index": None,
                "response": response,
            }
            sample_result["timings"]["add_chunk"][idx_key] = elapsed
            save_sample_result(sample_path, sample_result)

            if dream_every and idx % dream_every == 0:
                _fire_dream(idx)

        if dream_every and total_chunks % dream_every != 0:
            # final consolidation so the tail chunks are dreamed before QA
            _fire_dream(total_chunks)

        # Snapshot build tokens (everything since reset, before any QA runs)
        build_stats = _snapshot_tokens()
        sample_result["token_stats"] = {"build_raw": build_stats, "build_sum": _sum_tokens(build_stats)}
        save_sample_result(sample_path, sample_result)

        qa_list = list(enumerate(item.get("qa", []), start=1))
        if args.max_questions is not None:
            qa_list = qa_list[: args.max_questions]
        if args.question_indices:
            selected = {
                int(value.strip())
                for value in args.question_indices.split(",")
                if value.strip()
            }
            qa_list = [(qidx, qa) for qidx, qa in qa_list if qidx in selected]

        for qidx, qa in qa_list:
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
            predicted = None
            message_trace = None
            usage_trace = None
            usage_total = None
            if task_agent:
                start = time.perf_counter()
                trace = task_agent.answer(input_messages, user_id=storage_user_id)
                predicted = trace.get("answer")
                message_trace = trace.get("messages")
                usage_trace = trace.get("usage")
                usage_total = trace.get("usage_total")
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
                # Pre-strip text, kept so a chain-of-thought run can be re-judged both
                # with and without its reasoning. The gap between those two scores is the
                # judge's length bias, which is otherwise inseparable from a real gain.
                "raw_answer": trace.get("raw_answer") if task_agent else None,
                "messages": message_trace,
                "usage": usage_trace,
                "usage_total": usage_total,
            }
            sample_result["records"][qidx_key] = record
            save_sample_result(sample_path, sample_result)
            print_qa(qidx, question, expected_answer, predicted)

        try:
            all_memories = memory_system.list_all_memories()
        except Exception as exc:
            all_memories = {
                "success": False,
                "error": str(exc),
                "user_id": storage_user_id,
            }

        memories_path = output_path / f"{sample_id}_memories.json"
        with memories_path.open("w", encoding="utf-8") as handle:
            json.dump(all_memories, handle, ensure_ascii=False, indent=2)

        # Snapshot post-QA total tokens. "query_tokens" is server-side retrieval
        # cost only (keyword extraction + LightRAG sub-calls). The actual QA
        # answer LLM call goes through task_agent (client-side OpenAI), tracked
        # separately in records[*].usage_total.
        post_qa_stats = _snapshot_tokens()
        post_qa_sum = _sum_tokens(post_qa_stats)
        build_sum = sample_result.get("token_stats", {}).get("build_sum", {})
        query_sum = {
            k: max(post_qa_sum.get(k, 0) - build_sum.get(k, 0), 0)
            for k in ("prompt", "completion", "total", "calls")
        }
        sample_result.setdefault("token_stats", {})
        sample_result["token_stats"]["query_raw"] = post_qa_stats
        sample_result["token_stats"]["query_sum"] = query_sum
        save_sample_result(sample_path, sample_result)


if __name__ == "__main__":
    main()
