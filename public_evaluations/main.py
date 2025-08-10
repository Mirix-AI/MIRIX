from dotenv import load_dotenv
load_dotenv()
import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from conversation_creator import ConversationCreator
from agent import AgentWrapper

## rm -r ~/.mirix/sqlite.db
## conda activate mirix_eval
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
    return parser.parse_args()

def run_with_chunks_and_questions(
        args,
        global_idx,
        chunks, 
        queries_and_answers):

    # make out_dir with the model name
    out_dir = f"./results/{args.agent_name}_{args.dataset}-model{args.model_name}/"
    if args.agent_name == 'gpt-long-context' or args.agent_name == 'gemini-long-context':
        out_dir = f"./results/{args.agent_name}_{args.dataset}-{args.model_name}/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_dir = out_dir + f"{global_idx}"


    # if out_dir exists, load the agent from it
    if os.path.exists(out_dir):
        agent = AgentWrapper(args.agent_name, load_agent_from=out_dir, model_name=args.model_name, config_path=args.config_path)
    # create an agent
    else:
        if args.agent_name == 'mirix':
            if os.path.exists(os.path.expanduser(f"~/.mirix/sqlite.db")):
                # need to delete the existing db
                os.system(f"rm -rf ~/.mirix/sqlite.db")
            
        agent = AgentWrapper(args.agent_name, model_name=args.model_name, config_path=args.config_path)


    # load the current step & chunks for continuing memory accumulation
    if os.path.exists(f"{out_dir}/current_step.txt"):
        with open(f"{out_dir}/current_step.txt", "rb") as f:
            current_step = int(f.read().decode())
    else:
        current_step = -1

    if os.path.exists(f"{out_dir}/chunks.json"):
        with open(f"{out_dir}/chunks.json", "r") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    for idx, next_chunk in tqdm(enumerate(chunks), total=len(chunks)):

        if idx <= current_step or args.force_answer_question:
            continue

        if args.dataset == 'ScreenshotVQA':
            image_uris, timestamp = [x[0] for x in next_chunk], [x[1] for x in next_chunk]
            response = agent.send_message(message=None, 
                                          image_uris=image_uris, 
                                          memorizing=True,
                                          timestamp=timestamp)
            existing_chunks.append({
                'image_uri': image_uris,
                'response': response
            })
        else:
            prompt = next_chunk
            response = agent.send_message(prompt, memorizing=True)

            existing_chunks.append({
                'message': prompt,
                'response': response
            })

        # save the chunks and current step in chunking
        # TODO: if args.agent_name == 'mirix':
        agent.save_agent(out_dir)

        with open(f"{out_dir}/chunks.json", "w") as f:
            json.dump(existing_chunks, f, indent=2)

        with open(f"{out_dir}/current_step.txt", "wb") as f:
            f.write(str(idx).encode())


    # save the agent
    agent.save_agent(out_dir)
    agent.prepare_before_asking_questions()


    # load the results to continue from last breakpoint
    if os.path.exists(f"{out_dir}/results.json"):
        existing_results = json.load(open(f"{out_dir}/results.json", "r"))
    else:
        existing_results = []
    
    existing_results = [x for x in existing_results if x['response'] != 'ERROR']

    all_questions = [x['question'] for x in existing_results]


    # QA loop
    for item in queries_and_answers:

        question_text = item[1]

        if question_text in all_questions:
            item_idx = all_questions.index(question_text)
            if 'metadata' not in existing_results[item_idx]:
                existing_results[item_idx]['metadata'] = item[3] if len(item) > 3 else None
                with open(f"{out_dir}/results.json", "w") as f:
                    json.dump(existing_results, f, indent=2)
            continue
        print("Question [{} / {}]: ".format(len(existing_results), len(queries_and_answers)), question_text)

        response = agent.send_message(question_text, memorizing=False)

        existing_results.append(
            {
                'question': question_text,
                'response': response,
                'answer': item[2],
                'metadata': item[3] if len(item) > 3 else None
            }
        )

        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(existing_results, f, indent=2)
        
        agent = AgentWrapper(args.agent_name, load_agent_from=out_dir, model_name=args.model_name, config_path=args.config_path)

def main():
    
    args = parse_args()
    conversation_creator = ConversationCreator(args.dataset, args.num_exp)

    if args.agent_name == 'gpt-long-context':
        with_instructions = False
    else: 
        with_instructions = True

    all_chunks = conversation_creator.chunks(with_instructions=with_instructions)
    all_queries_and_answers = conversation_creator.get_query_and_answer()

    for global_idx, (chunks, queries_and_answers) in enumerate(zip(all_chunks, all_queries_and_answers)):
        
        if args.global_idx is not None and global_idx != args.global_idx:
            continue
        
        run_with_chunks_and_questions(args, global_idx, chunks, queries_and_answers)

if __name__ == '__main__':
    main()
