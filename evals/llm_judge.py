import hashlib
import argparse
import json
from collections import defaultdict

import numpy as np
import openai
from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a ’gold’ (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(question, gold_answer, generated_answer):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": ACCURACY_PROMPT.format(
                    question=question, gold_answer=gold_answer, generated_answer=generated_answer
                ),
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    label = json.loads(response.choices[0].message.content)["label"]
    return 1 if label == "CORRECT" else 0


# def main():
#     """Main function to evaluate RAG results using LLM judge."""
#     parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
#     parser.add_argument(
#         "--input_file",
#         type=str,
#         default="results/default_run_v4_k30_new_graph.json",
#         help="Path to the input dataset file",
#     )

#     args = parser.parse_args()

#     dataset_path = args.input_file
#     output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

#     with open(dataset_path, "r") as f:
#         data = json.load(f)

#     LLM_JUDGE = defaultdict(list)
#     RESULTS = defaultdict(list)

#     index = 0
#     for k, v in data.items():
#         for x in v:
#             question = x["question"]
#             gold_answer = x["answer"]
#             generated_answer = x["response"]
#             category = x["category"]

#             # Skip category 5
#             if int(category) == 5:
#                 continue

#             # Evaluate the answer
#             label = evaluate_llm_judge(question, gold_answer, generated_answer)
#             LLM_JUDGE[category].append(label)

#             # Store the results
#             RESULTS[index].append(
#                 {
#                     "question": question,
#                     "gt_answer": gold_answer,
#                     "response": generated_answer,
#                     "category": category,
#                     "llm_label": label,
#                 }
#             )

#             # Save intermediate results
#             with open(output_path, "w") as f:
#                 json.dump(RESULTS, f, indent=4)

#             # Print current accuracy for all categories
#             print("All categories accuracy:")
#             for cat, results in LLM_JUDGE.items():
#                 if results:  # Only print if there are results for this category
#                     print(f"  Category {cat}: {np.mean(results):.4f} " f"({sum(results)}/{len(results)})")
#             print("------------------------------------------")
#         index += 1

#     # Save final results
#     with open(output_path, "w") as f:
#         json.dump(RESULTS, f, indent=4)

#     # Print final summary
#     print("PATH: ", dataset_path)
#     print("------------------------------------------")
#     for k, v in LLM_JUDGE.items():
#         print(k, np.mean(v))


# if __name__ == "__main__":
#     main()

# One canonical prompt, asserted. Four files in this repo carried their own copy of the
# LoCoMo accuracy prompt and one of them used U+2019 apostrophes where the others used
# ASCII; the difference was measured at 3 questions, all on the error side. A grader that
# silently differs between two comparisons makes those comparisons incomparable, so any
# edit to the text below must update this digest deliberately.
PROMPT_SHA = "e994418538d665fa1321f1a38dfeeac3e56a0c7616ec2ba93347be00eb76d2c8"


def assert_canonical_prompt() -> None:
    """Callers importing ACCURACY_PROMPT should not have to trust that it is unmodified."""
    got = hashlib.sha256(ACCURACY_PROMPT.encode("utf-8")).hexdigest()
    if got != PROMPT_SHA:
        raise AssertionError(
            f"llm_judge.ACCURACY_PROMPT changed: {got[:16]} != {PROMPT_SHA[:16]}. "
            "Scores graded before and after this edit are not comparable."
        )
