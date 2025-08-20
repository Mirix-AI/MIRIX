from dotenv import load_dotenv
load_dotenv()
import os
import json
import time
import argparse
import numpy as np
import subprocess
import tempfile
from tqdm import tqdm
from log_utils import tee_parent_logs, run_subprocess_with_logs, prepare_logs_paths
from conversation_creator import ConversationCreator

## CONSTANTS for chunk size moved to constants.py to avoid circular imports
## python main.py --agent_name mirix --dataset LOCOMO --config_path ../mirix/configs/mirix_azure_example.yaml
## python main.py --agent_name mirix --dataset MemoryAgentBench --config_path ../mirix/configs/mirix_azure_example.yaml --num_exp 2
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Memory Illustration")
    parser.add_argument("--agent_name", type=str, choices=['gpt-long-context', 'mirix', 'siglip', 'gemini-long-context'])
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['LOCOMO', 'ScreenshotVQA', 'MemoryAgentBench'])
    parser.add_argument("--num_exp", type=int, default=5)
    parser.add_argument("--load_db_from", type=str, default=None)
    parser.add_argument("--num_images_to_accumulate", default=None, type=int)
    parser.add_argument("--global_idx", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini", help="Model name to use for gpt-long-context agent")
    parser.add_argument("--config_path", type=str, default=None, help="Config file path for mirix agent")
    parser.add_argument("--force_answer_question", action="store_true", default=False)
    # for MemoryAgentBench / , "eventqa_full"
    parser.add_argument("--sub_datasets", nargs='+', type=str, default=["longmemeval_s*"], help="Sub-datasets to run")
    
    return parser.parse_args()

def run_subprocess_interactive(args, global_idx, logs=None):
    """
    Run the run_instance.py script using subprocess with interactive capability.
    """
    # Build command arguments
    cmd = [
        'python', 'run_instance.py',
        '--agent_name', args.agent_name,
        '--dataset', args.dataset,
        '--global_idx', str(global_idx),
        '--num_exp', str(args.num_exp),
        '--sub_datasets', *args.sub_datasets
    ]
    
    # Add optional arguments
    if args.model_name:
        cmd.extend(['--model_name', args.model_name])
    if args.config_path:
        cmd.extend(['--config_path', args.config_path])
    if args.force_answer_question:
        cmd.append('--force_answer_question')
    
    try:
        print(f"Running subprocess for global_idx {global_idx}")
        parent_log_path = logs.parent if logs is not None else None
        if parent_log_path:
            try:
                with open(parent_log_path, 'a', encoding='utf-8') as plog:
                    plog.write(f"[main] Starting subprocess for global_idx {global_idx}: {' '.join(cmd)}\n")
            except Exception:
                pass

        result = run_subprocess_with_logs(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout_log_path=(logs.child if logs is not None else None),
            stderr_log_path=None,
            combine_streams=True,
        )

        print(f"Subprocess completed successfully for global_idx {global_idx}")
        if parent_log_path:
            try:
                with open(parent_log_path, 'a', encoding='utf-8') as plog:
                    plog.write(f"[main] Subprocess completed for global_idx {global_idx} (rc=0)\n")
            except Exception:
                pass
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed for global_idx {global_idx} with return code {e.returncode}")
        if parent_log_path:
            try:
                with open(parent_log_path, 'a', encoding='utf-8') as plog:
                    plog.write(f"[main] Subprocess failed for global_idx {global_idx} (rc={e.returncode})\n")
            except Exception:
                pass
        raise

def main():
    
    # parse arguments
    args = parse_args()
    
    # initialize run count
    conversation_creator = ConversationCreator(args.dataset, args.num_exp, args.sub_datasets)
    dataset_length = conversation_creator.get_dataset_length()

    for global_idx in tqdm(range(dataset_length), desc="Running subprocesses", unit="item"):
        
        if args.global_idx is not None and global_idx != args.global_idx:
            continue
        
        logs = prepare_logs_paths(args, global_idx)
        with tee_parent_logs(logs.parent):
            try:
                run_subprocess_interactive(
                    args,
                    global_idx,
                    logs=logs,
                )
            except Exception as e:
                # Log the error and continue with the next item
                try:
                    with open(logs.parent, 'a', encoding='utf-8') as plog:
                        plog.write(f"[main] Error during run for global_idx {global_idx}: {repr(e)}\n")
                except Exception:
                    pass
                print(f"Encountered error for global_idx {global_idx}: {e}. Continuing to next.")
                continue

if __name__ == '__main__':
    main()
