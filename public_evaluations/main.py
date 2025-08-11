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


## python main.py --agent_name mirix --dataset MemoryAgentBench
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Modal Memory Illustration")
    parser.add_argument("--agent_name", type=str, choices=['gpt-long-context', 'mirix', 'siglip', 'gemini-long-context'])
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['LOCOMO', 'ScreenshotVQA', 'MemoryAgentBench'])
    parser.add_argument("--num_exp", type=int, default=100)
    parser.add_argument("--load_db_from", type=str, default=None)
    parser.add_argument("--num_images_to_accumulate", default=None, type=int)
    parser.add_argument("--global_idx", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini", help="Model name to use for gpt-long-context agent")
    parser.add_argument("--config_path", type=str, default=None, help="Config file path for mirix agent")
    parser.add_argument("--force_answer_question", action="store_true", default=False)
    # for MemoryAgentBench
    parser.add_argument("--chunk_size", type=int, default=4096)

    return parser.parse_args()

def run_with_chunks_and_questions_subprocess(args, global_idx, chunks, queries_and_answers):
    """
    Run the extracted function using subprocess to isolate memory and processes.
    """
    # Create temporary files for chunks and queries
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as chunks_file:
        json.dump(chunks, chunks_file, indent=2)
        chunks_filepath = chunks_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as queries_file:
        json.dump(queries_and_answers, queries_file, indent=2)
        queries_filepath = queries_file.name
    
    try:
        # Build command arguments
        cmd = [
            'python', 'run_instance.py',
            '--agent_name', args.agent_name,
            '--dataset', args.dataset,
            '--global_idx', str(global_idx),
            '--chunks_file', chunks_filepath,
            '--queries_file', queries_filepath,
            '--chunk_size', str(args.chunk_size)
        ]
        
        # Add optional arguments
        if args.model_name:
            cmd.extend(['--model_name', args.model_name])
        if args.config_path:
            cmd.extend(['--config_path', args.config_path])
        if args.force_answer_question:
            cmd.append('--force_answer_question')
        
        # Run the subprocess
        print(f"Running subprocess for global_idx {global_idx}")
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), 
                              capture_output=True, text=True, check=True)
        
        print(f"Subprocess completed successfully for global_idx {global_idx}")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed for global_idx {global_idx} with return code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise
    
    finally:
        # Clean up temporary files
        try:
            os.unlink(chunks_filepath)
            os.unlink(queries_filepath)
        except OSError:
            pass

def main():
    
    args = parse_args()
    conversation_creator = ConversationCreator(args.dataset, args.num_exp, args.chunk_size)

    if args.agent_name == 'gpt-long-context':
        with_instructions = False
    else: 
        with_instructions = True

    all_chunks = conversation_creator.chunks(with_instructions=with_instructions)
    all_queries_and_answers = conversation_creator.get_query_and_answer()

    for global_idx, (chunks, queries_and_answers) in enumerate(zip(all_chunks, all_queries_and_answers)):
        
        if args.global_idx is not None and global_idx != args.global_idx:
            continue
        
        run_with_chunks_and_questions_subprocess(args, global_idx, chunks, queries_and_answers)

if __name__ == '__main__':
    main()
