import os
import json
import argparse
from subprocess import Popen

def main():
    # collect examples
    config_files = [
        os.path.join("config_files", f) for f in os.listdir("config_files")
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    config_files = sorted(config_files, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    config_list = [json.load(open(f)) for f in config_files]
    config_flags = [config["sites"][0] == args.website for config in config_list]
    task_ids = [config["task_id"] for config, flag in zip(config_list, config_flags) if flag]

    if args.end_index == None: args.end_index = len(task_ids)
    for tid in task_ids[args.start_index: args.end_index]:
        # step 1: run inference
        process = Popen([
            "python", "run.py", 
            "--task", f"webarena.{tid}",
            "--workflow_path", f"workflow/{args.website}.txt"
        ])
        process.wait()

        # step 2: run evaluation
        process = Popen([
            "python", "-m", "autoeval.evaluate_trajectory",
            "--result_dir", f"results/webarena.{tid}"
        ])
        process.wait()

        # step 3: update workflow
        process = Popen([
            "python", "induce_prompt.py",
            "--result_dir", "results",
            "--output_path", f"workflow/{args.website}.txt",
        ])
        process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", type=str, required=True,
                        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    args = parser.parse_args()

    main()
