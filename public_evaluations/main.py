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
from conversation_creator import ConversationCreator
from constants import CHUNK_SIZE_MEMORY_AGENT_BENCH

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
    parser.add_argument("--sub_datasets", nargs='+', type=str, default=["longmemeval_s*", "eventqa_full"], help="Sub-datasets to run")
    
    return parser.parse_args()

def run_subprocess_interactive(args, global_idx):
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
        # Run the subprocess without capturing output (allows interactive debugging)
        print(f"Running subprocess for global_idx {global_idx}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), 
                              check=True)  # No capture_output=True
        
        print(f"Subprocess completed successfully for global_idx {global_idx}")
            
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed for global_idx {global_idx} with return code {e.returncode}")
        raise

def main():
    
    # parse arguments
    args = parse_args()
    
    # initialize conversation creator
    dataset_length = args.num_exp * len(args.sub_datasets)

    for global_idx in tqdm(range(dataset_length), desc="Running subprocesses", unit="item"):
        
        if args.global_idx is not None and global_idx != args.global_idx:
            continue
        
        run_subprocess_interactive(args, global_idx)

if __name__ == '__main__':
    main()
