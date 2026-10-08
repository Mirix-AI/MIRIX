# Copyright 2026 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import gzip
import pickle
import json
import random
import argparse
from functools import partial
import time

from prompts.memory_instruction import SUCCESSFUL_SI, FAILED_SI, AWM_INSTRUCTION, AWM_EXAMPLE
from utils.clients import CLIENT_DICT


def _extract_think_from_output(ai: dict) -> str:
    """Extract think text from agent info, falling back to chat_messages."""
    think = ai.get("think", "")
    if think:
        return think
    # Model may put reasoning outside <think> tags — extract text before <action>
    msgs = ai.get("chat_messages", [])
    if len(msgs) >= 3:
        output = str(msgs[2])
        action_idx = output.find("<action>")
        if action_idx > 0:
            raw = output[:action_idx].strip()
            raw = re.sub(r"</?think>", "", raw).strip()
            if raw:
                return raw
    return think


def extract_think_and_action(folder: str) -> tuple[list[str], list[str]]:
    """Extract think/action pairs from step pkl files."""
    step_files = sorted(
        [f for f in os.listdir(folder) if re.match(r"step_\d+\.pkl\.gz", f)],
        key=lambda f: int(re.findall(r"\d+", f)[0])
    )
    think_list = []
    action_list = []
    for f in step_files:
        try:
            with gzip.open(os.path.join(folder, f), 'rb') as fh:
                data = pickle.load(fh)
            ai = data.agent_info
            think = _extract_think_from_output(ai)
            action = ai.get("action", "")
            if not action:  # skip empty action steps
                continue
            think_list.append(think)
            action_list.append(action)
        except Exception:
            continue
    return think_list, action_list

def format_trajectory(think_list: list[str], action_list: list[list[str]]) -> str:
    trajectory = []
    for t, a in zip(think_list, action_list):
        # acts = '\n'.join(a)
        acts = a
        trajectory.append(f"<think>\n{t}\n</think>\n<action>\n{acts}\n</action>")
    return '\n\n'.join(trajectory)

def random_group_sample(d: dict, n) -> list:
    """Randomly sample n groups from the dictionary."""
    return [ex for v in d.values() for ex in random.sample(v, min(n, len(v)))]


def format_examples(examples: list[dict], flag=False) -> str:
    """Format examples to the prompt."""
    formatted_examples = []
    for ex in examples:
        trajectory = format_trajectory(ex["think_list"], ex["action_list"])
        formatted_examples.append(f"Query: {ex['query']}\nTrajectory:\n{trajectory}")
    # return '\n\n'.join(["## Concrete Examples"] + formatted_examples + ["## Summary Workflow"])
    if flag:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is CORRECT."] + ["## Updated Memory"])
    else:
        return '\n\n'.join(["## Query and Trajectory Generated Using Previous Memory"] + formatted_examples + ["## Correctness Signal"]+ ["The result is INCORRECT."] + ["## Updated Memory"])


def get_info(f: str, status: str = None) -> dict:
        
    # get query -> task objective
    task_id = f.split('/')[-1].split("_")[0].split(".")[1]
    config_path = os.path.join("config_files", f"{task_id}.json")
    config = json.load(open(config_path))
    query = config["intent"]

    template_id = config["intent_template_id"]  # for deduplication

    # parse trajectory from step pkl files
    think_list, action_list = extract_think_and_action(f)

    # add to template dict
    if status == 'success':
        wdict = {"query": query, "template_id": template_id, "think_list": think_list, "action_list": action_list, "status": "success"}
    elif status == 'fail':
        wdict = {"query": query, "template_id": template_id, "think_list": think_list, "action_list": action_list, "status": "fail"}

    return wdict

def main():
    # collect result directories, e.g., ["results/webarena.0", ...]
    args.result_dir = args.result_dir.split()

    cur_task = os.path.join(args.result_dir[0], args.task)

    # correctness signals for trajectories
    if args.criteria == "gt":
        reward = json.load(open(os.path.join(cur_task, "summary_info.json")))["cum_reward"]
    elif args.criteria == "autoeval":
        reward = json.load(open(os.path.join(cur_task, f"{args.model}_autoeval.json")))[0]["rm"]
    else:
        raise ValueError(f"Invalid criteria: {args.criteria}.")

    if reward == 1:
        status = "success"
    else:
        status = "fail"

    ex = get_info(cur_task, status)

    # Define the LLM client based on the model choice
    llm_client = CLIENT_DICT[args.model](model_name=args.model)

    # memory extraction based on the trajectory and user queries
    trajectory = format_trajectory(ex["think_list"], ex["action_list"])
    trajectory = f"**Query:** {ex['query']}\n\n**Trajectory:**\n{trajectory}"

    # Load autoeval thoughts if available, and append to trajectory so the
    # memory LLM knows why the task succeeded or failed.
    autoeval_thoughts = ""
    if args.criteria == "autoeval":
        autoeval_path = os.path.join(cur_task, f"{args.model}_autoeval.json")
        try:
            autoeval_data = json.load(open(autoeval_path))
            if isinstance(autoeval_data, list) and autoeval_data:
                autoeval_thoughts = autoeval_data[0].get("thoughts", "")
            elif isinstance(autoeval_data, dict):
                autoeval_thoughts = autoeval_data.get("thoughts", "")
        except Exception:
            pass

    if args.memory_mode == "reasoningbank":
        if autoeval_thoughts:
            status_label = "succeeded" if ex['status'] == 'success' else "failed"
            trajectory += f"\n\nThe task {status_label} because: {autoeval_thoughts}"
        if ex['status'] == 'success':
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=SUCCESSFUL_SI, temperature=1.0)
        else:
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=FAILED_SI, temperature=1.0)

    elif args.memory_mode == "awm":
        if ex['status'] == 'success':
            generated_memory_item, _ = llm_client.one_step_chat(trajectory, system_msg=AWM_INSTRUCTION + AWM_EXAMPLE, temperature=0.7)

    elif args.memory_mode == "synapse":
        if ex['status'] == 'success':
            generated_memory_item = trajectory

    # write memory to jsonl file 
    with open(args.output_path, 'a') as f:
        f.write(json.dumps({
            "task_id": args.task.split(".")[-1],
            "query": ex["query"],
            "think_list": ex["think_list"],
            "action_list": ex["action_list"],
            "status": ex["status"],
            "memory_items": generated_memory_item.split("\n\n"),
            "template_id": ex["template_id"]
        }) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="results_base_new",
                        help="Path to the result directory. Support multiple directories separated by space.")
    parser.add_argument("--output_path", type=str, default=None, required=True,
                        help="Path to the output file.")
    parser.add_argument("--criteria", type=str, default="autoeval", choices=["gt", "autoeval"])
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        choices=["gpt-3.5", "gpt-4", "gpt-4o", "gemini-2.5-flash"])
    parser.add_argument("--task", type=str, default="webarena.47")
    parser.add_argument("--memory_mode", type=str, default="reasoningbank")
    args = parser.parse_args()

    main()