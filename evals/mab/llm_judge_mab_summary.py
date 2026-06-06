"""MemoryAgentBench-aligned summarization judge.

Port of ``llm_based_eval/summarization_evaluate.py`` from
https://github.com/HUST-AI-HYZ/MemoryAgentBench so summarization scores
on the Long_Range_Understanding split (``infbench_sum_eng_shots2``)
are directly comparable to the MAB leaderboard.

Three GPT-4o calls per sample:

  1. **Fluency** — binary 0/1 — is the generated text coherent, non-
     repetitive, grammatically fluent? An incoherent or repetitive
     summary zeroes out F1.
  2. **Recall** — count of expert-written keypoints supported by the
     generated summary, normalised by total keypoints.
  3. **Precision** — count of sentences in the generated summary that
     are supported by the expert reference summary, normalised by
     total sentence count.

Final score per sample::

    F1 = fluency * 2 * recall * precision / (recall + precision)

Aggregation over a run: arithmetic mean of F1, recall, precision,
fluency across all judged records.

Why the book-variant prompts: ``infbench_sum_eng_shots2`` summarises
novels, so we use ``*_book`` (literature) prompts, not the lawsuit
variants. ``multi_lexsum`` (the lawsuit variants in the official
script) is not in the MAB LRU split.

Metric model is pinned to ``gpt-4o-2024-05-13`` to match the official
script — the unpinned ``gpt-4o`` alias drifts and would diverge from
the leaderboard.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Official MAB summarization metric model
# (summarization_evaluate.py: OpenAIModel("gpt-4o-2024-05-13", ...))
DEFAULT_METRIC_MODEL = "gpt-4o-2024-05-13"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096


_FLUENCY_PROMPT_BOOK = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish. Note that even if the answer is coherent, if it is repetitive or incomplete, it should be given a score of 0.
  - Examples:
    - Incomplete: "Summary:"
    - Incoherent: "Summary:ЉЉЉЉЉЉЉЉЉЉЉЉЉЉ \\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\_______                       is is is"
    - Repetitive: "Summary:\\n\\n\\n\\n\\n\\n\\n\\n |THE next morning, when Ellington came down to breakfast, she found a letter on the table addressed to her. ... I'll tell her that I'm sure she'll be a great comfort to me."

- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers. If the text is coherent, non-repetitive, and fluent, but the last sentence is truncated, it should still be given a score of 1.
  - Examples:
    - "The story revolves around the life of Jennifer Pete, a young woman with a strong sense of morality and spirituality. ... Overall, the story is a nuanced and thought-provoking exploration of the human condition."

Now, read the provided text, and evaluate the fluency using the rubric. Then output your score in the following json format: {{"fluency": 1}}.

Text: "{text}"
"""


_RECALL_PROMPT_BOOK = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel. It should discuss the plots and characters of the story. The text should contain all the given key points.

Below is your grading rubric:
Recall:
- Evaluate the provided summary by deciding if each of the key points is present in the provided summary. A key point is considered present if its factual information is mostly-supported by the provided summary. If a key point contains multiple facts, it's still considered supported if most of the facts are present.
- Score: the number of key points mostly-supported by the provided summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Key points:
1. Cal Margaret lives in Berlin, Germany.
2. Cal decides to write his life story, starting with the history of the recessive gene causing his intersex condition.
3. The story begins with Cal's grandparents, Raul and Harris, in a village on Mount Olympus in 1922.
4. Raul and Harris are siblings who fall in love and decide to immigrate to Detroit after their parents' deaths.
5. They escape the burning of Smyrna by the Turkish army and find passage to America.
6. On the ship, Raul and Harris pretend to meet for the first time and then wed.
7. In Detroit, they move in with their cousin Lavinia and her husband, Gerry Helena.
8. Helena takes Raul into his alcohol smuggling business.
9. Harris and Lavinia get pregnant on the same night, causing Helena to suspect Lavinia of cheating with Raul.
10. Helena takes Raul on a drive on the ice to interrogate him, but the car falls into the water and Raul escapes.
11. In 1945, Raul and Harris's son, Irma, develops a crush on Helena and Lavinia's daughter, Russell.
12. Harris encourages Russell to accept a proposal from a seminary student, Ida, causing Irma to join the Navy in anger.
13. Russell calls off her engagement to Ida when she realizes Irma might die in the U.S. invasion of Japan.
14. Irma excels on a test, gets transferred to the officer's academy, and is spared from fighting in the rest of the war.
15. Irma and Russell marry and have a son named Deana Salome.
16. Five years later, they wish for a daughter and conceive Ali (Callie) using pseudo-scientific methods.
17. Irma retires from the Navy and takes over Raul's bar, turning it into a diner.
18. The diner burns down during the Twelfth Street Riot in 1967, but the family has enough insurance money to move to Grosse Pointe.
19. They move into an unusual house on a street named Middlesex.
20. Seven-year-old Callie wants to make friends in the new neighborhood and practices kissing with the girl next door, Sven Chrissy.
21. Callie is sent to an all-girls prep school and worries about not getting her period or growing breasts.
22. Callie develops a crush on a classmate referred to as 'the Obscure Object' and they begin a physical relationship.
23. Callie is hit by a tractor and the hospital doctors realize she is biologically male.
24. Russell and Irma take Callie to a specialist in New York named Dr. Lester.
25. Dr. Lester wants to use Callie to prove his theory that gender is a social construct and recommends surgery.
26. Callie learns she is biologically male, renames himself Cal, and runs away to San Francisco.


Summary: <start of summary>The story begins with the birth of the narrator, Cal Stephanides, who is a hermaphrodite. ... The book also explores the history of Detroit and its transformation from a small town to a major industrial city.<end of summary>


Reasoning: The summary incorrectly identifies the protagonist as "Cal Stephanides" instead of "Cal Margaret", so key point 1 is not supported. It does not mention key point 2. The summary mentions that Raul and Harris are silbings and that they eventually marry and settle down in Detroit so key point 3 is supported. It also mentions the Turkish attack and how they escape from Smyrna to America so key point 5 is supported. It does not talk about the ship where they are wed so key point 6 is not supported. The summary then stops discussing the plot and so it does not mention key point 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, or 26. Thus, the only supported key points are 3 and 5, so recall is 2.

Output: {{"supported_key_points": [3, 5], "recall": 2}}


Example 2:

Key points:
1. The story follows the Octavia family traveling along the Malaysia River from Iquitos in Peru to Belem in Brazil.
2. Lauren Octavia is the central character, a wealthy rancher with a dark secret.
3. Lauren has been living under a false name, hiding his identity as a wrongfully accused criminal who escaped from prison 20 years ago.
4. Lauren sees an opportunity to clear his name and risks the journey to Brazil to present evidence proving his innocence.
5. Lauren's family, unaware of his past, accompanies him on the journey.
6. Lauren's daughter, Minha, is engaged to Manoel, a gallant but flippish army physician.
7. Lauren's son, Benito, is brave and hot-headed, greatly admiring and respecting his father.
8. Duncan, a soldier turned rogue, discovers Lauren's secret and blackmails him.
9. The journey down the river is filled with turbulence, both literal and figurative.
10. The natural wonders and wildlife of the Malaysia River add flavor to the story.
11. The family faces lethal dangers, including river pirates and boating accidents.
12. The story subtly raises the issue of slavery in Brazil, a contemporary concern at the time.
13. The climax occurs in Belem with a trial for Lauren.
14. A dramatic court scene unfolds where the credibility of Lauren's documents is questioned.
15. Lauren is on the verge of being convicted.
16. Duncan, who was killed by an Indian's poisoned arrow earlier, is dissected.
17. A letter confirming Lauren's claims is found inside Duncan, proving Lauren's innocence.
18. The novel ends with the Octavias happily returning to their fazenda, their home in Iquitos.
19. The adventurous journey of eight hundred leagues on the Malaysia comes to an end.


Summary: <start of summary>The story follows the journey of the Octavia family as they travel down the Malaysia River on a massive raft, or "jangada," from Iquitos to Belem. ... The story concludes with the family reaching their destination, having grown and changed through their experiences on the river.<end of summary>


Reasoning: Key point 1 is supported by the summary. ... Therefore, the supported key points are 1, 6, 9, and 10, so the recall score is 4.

Output: {{"supported_key_points": [1, 6, 9, 10], "recall": 4}}

Now, read the provided summary and key points, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"supported_key_points": [2, 4], "recall": 2}}, where "supported_key_points" contains the key points that are present in the summary and "recall" is the total number of key points present in the summary.

Key points:
{keypoints}

Summary: <start of summary>{summary}<end of summary>
"""


_PRECISION_PROMPT_BOOK = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel.

Below is your grading rubric:
Precision:
- Evaluate the provided summary by deciding if each sentence in the provided summary is supported by the information provided in the expert summary. A sentence is considered supported if its major facts align with the information in the expert summary. A sentence is still considered supported even if some of its minor details, such as dates, entity names, or the location, are not explicitly mentioned in the expert summary. A sentence is not supported if its major facts are not mentioned or contradicted in the expert summary. It is also not supported if it introduces new information not present in the expert summary, such as additional analysis or commentary on the story.
- Score: the number of sentences in the provided summary that are supported by the expert summary.
- Examples: use the following examples to guide your evaluation.

Example 1:

Expert summary: <start of summary>Cal Margaret is a man living in Berlin, Germany. ... Years later, Cal starts a relationship with a woman named Chase Leuan in Berlin.<end of summary>

Provided summary: <start of summary>The story begins with the birth of the narrator, Cal Stephanides, who is a hermaphrodite. ... Overall, the book is a sweeping narrative that spans multiple generations and continents. It is a story about identity, culture, family, and history, and it raises important questions about the human experience.<end of summary>

Reasoning: The first sentence is not supported because the provided summary claims the character is named "Cal Stephanides" while the expert summary indicates that they are named "Cal Margaret". ... Therefore, the precision score is 10.

Output: {{"precision": 10, "sentence_count":  22}}


Example 2:

Expert summary: <start of summary>The story chronicles the journey of the Octavia family ... putting an end to their adventurous journey of eight hundred leagues on the Malaysia.<end of summary>

Provided: <start of summary>The story follows the journey of the Octavia family ... The story concludes with the family reaching their destination, having grown and changed through their experiences on the river.<end of summary>

Reasoning: Sentence 1 is supported as the expert summary mentions the Octavia family traveling along the Malaysia River from Iquitos in Peru to Belem in Brazil. ... Therefore, the precision score is 5.

Output: {{"precision": 5, "sentence_count": 15}}

Now, read the provided summary and expert summary, and evaluate the summary using the rubric. First, think step-by-step and provide your reasoning and assessment on the answer. Then output your score in the following json format: {{"precision": 7, "sentence_count": 20}}.

Expert summary: <start of summary>{expert_summary}<end of summary>

Provided summary: <start of summary>{summary}<end of summary>
"""


def _parse_json(text: str) -> Optional[Dict]:
    """Best-effort JSON extraction from a GPT-4 response.

    Mirrors official ``summarization_evaluate.parse_json``: greedy
    brace match, fallback to fenced ```json``` block. Returns the
    *last* JSON object — the model often emits an example object in
    its reasoning before the final score.
    """
    matches = re.findall(r"\{.*?\}", text or "", re.DOTALL)
    if matches:
        try:
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            pass
    fenced = re.findall(r"(?:```json)(.+?)(?:```)", text or "", re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced[-1].strip())
        except json.JSONDecodeError:
            pass
    return None


def _llm(prompt: str, model: str = DEFAULT_METRIC_MODEL) -> str:
    completion = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    return (completion.choices[0].message.content or "").strip()


def evaluate_mab_summary_judge(
    generated_summary: str,
    expert_summary: str,
    keypoints: Sequence[str],
    model: str = DEFAULT_METRIC_MODEL,
) -> Dict:
    """Run the three official MAB summarization judges and return all sub-scores plus F1.

    Returns a dict with keys::

        fluency           — 0 or 1
        recall_total      — len(keypoints)
        recall_found      — judge's count of supported keypoints
        recall            — recall_found / recall_total           in [0,1]
        precision_total   — judge's sentence_count
        precision_found   — judge's count of supported sentences
        precision         — precision_found / precision_total     in [0,1]
        f1                — fluency * 2*R*P / (R+P)                in [0,1]
        raw               — {fluency_out, recall_out, precision_out} unparsed text

    Any missing-or-unparseable judge call collapses the relevant sub-
    metric to 0 (and therefore F1 to 0) — matches the official rule of
    silently skipping records with ``None`` outputs.
    """
    text = (generated_summary or "").strip()
    expert = (expert_summary or "").strip()
    kp_list: List[str] = list(keypoints or [])
    kp_block = "\n".join(f"{i + 1}. {kp}" for i, kp in enumerate(kp_list))

    fluency_text = _llm(_FLUENCY_PROMPT_BOOK.format(text=text), model=model)
    recall_text = _llm(_RECALL_PROMPT_BOOK.format(keypoints=kp_block, summary=text), model=model)
    precision_text = _llm(_PRECISION_PROMPT_BOOK.format(expert_summary=expert, summary=text), model=model)

    f_obj = _parse_json(fluency_text)
    r_obj = _parse_json(recall_text)
    p_obj = _parse_json(precision_text)

    fluency = int(f_obj.get("fluency", 0)) if isinstance(f_obj, dict) else 0
    recall_found = int(r_obj.get("recall", 0)) if isinstance(r_obj, dict) else 0
    recall_total = len(kp_list)
    precision_found = int(p_obj.get("precision", 0)) if isinstance(p_obj, dict) else 0
    precision_total = int(p_obj.get("sentence_count", 0)) if isinstance(p_obj, dict) else 0

    recall = (recall_found / recall_total) if recall_total > 0 else 0.0
    precision = (precision_found / precision_total) if precision_total > 0 else 0.0
    f1 = (fluency * 2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

    return {
        "fluency": fluency,
        "recall_total": recall_total,
        "recall_found": recall_found,
        "recall": recall,
        "precision_total": precision_total,
        "precision_found": precision_found,
        "precision": precision,
        "f1": f1,
        "raw": {
            "fluency_out": fluency_text,
            "recall_out": recall_text,
            "precision_out": precision_text,
        },
    }
