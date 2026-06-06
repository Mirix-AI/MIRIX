"""
MemoryAgentBench-aligned LLM judge.

Port of ``llm_based_eval/longmem_qa_evaluate.py`` from
https://github.com/HUST-AI-HYZ/MemoryAgentBench so our scores are
directly comparable to the MAB leaderboard.

Differences vs ``llm_judge.py`` (our generic LoCoMo judge):
- Metric model is **gpt-4o** (not gpt-4o-mini).
- Prompts are **per question type** (single-session-*, multi-session,
  temporal-reasoning, knowledge-update, single-session-preference) with
  category-specific leniency (e.g. temporal allows off-by-one;
  knowledge-update accepts "old + new" responses; preference doesn't
  require hitting every rubric point).
- Scoring rule: ``label = 'yes' in response.lower()`` — the official
  binary rule, not a JSON-parsed CORRECT/WRONG label.

Abstention:
- ``longmem_eval.py`` now stores ``question_id`` (from HF
  ``metadata.question_ids``) on each record. ``organize_results.py``
  reads it and passes ``abstention='_abs' in question_id`` here. Old
  records without ``question_id`` quietly default to abstention=False.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Official MAB metric model (longmem_qa_evaluate.py: metric_model="gpt-4o").
DEFAULT_METRIC_MODEL = "gpt-4o"


_SINGLE_AND_MULTI = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. In addition, "
    "do not penalize off-by-one errors for the number of days. If the question asks for "
    "the number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response is still "
    "correct. \n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer.\n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired "
    "response. Otherwise, answer no. The model does not need to reflect all the points "
    "in the rubric. The response is correct as long as it recalls and utilizes the "
    "user's personal information correctly.\n\n"
    "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a "
    "model. Please answer yes if the model correctly identifies the question as "
    "unanswerable. The model could say that the information is incomplete, or some other "
    "information is given but the asked information is not.\n\n"
    "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


SUPPORTED_TASKS = frozenset(
    {
        "single-session-user",
        "single-session-assistant",
        "multi-session",
        "temporal-reasoning",
        "knowledge-update",
        "single-session-preference",
    }
)


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool = False,
) -> str:
    """Verbatim port of MemoryAgentBench's category routing."""
    if abstention:
        return _ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return _SINGLE_AND_MULTI.format(question, answer, response)
    if task == "temporal-reasoning":
        return _TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return _KNOWLEDGE_UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return _PREFERENCE.format(question, answer, response)
    raise NotImplementedError(f"Unknown MAB task category: {task!r}")


def evaluate_mab_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    question_type: str,
    abstention: bool = False,
    model: str = DEFAULT_METRIC_MODEL,
) -> int:
    """Return 1 if the MAB judge says yes, 0 otherwise.

    Mirrors the official scoring rule (`label = 'yes' in response.lower()`).
    """
    prompt = get_anscheck_prompt(
        question_type, question, gold_answer, generated_answer, abstention=abstention
    )
    completion = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=0,
        max_tokens=10,
    )
    text = (completion.choices[0].message.content or "").strip().lower()
    return 1 if "yes" in text else 0
