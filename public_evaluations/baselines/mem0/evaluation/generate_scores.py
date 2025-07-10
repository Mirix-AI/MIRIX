import json
import os
import pandas as pd

# Get all JSON files from eval_results directory
eval_results_dir = "eval_results"
json_files = [f for f in os.listdir(eval_results_dir) if f.endswith('.json')]

print(f"Found {len(json_files)} JSON files to process:")
for file in json_files:
    print(f"  - {file}")
print("\n" + "="*60 + "\n")

# Process each file
for json_file in json_files:
    print(f"Processing: {json_file}")
    print("-" * 40)
    
    # Load the evaluation metrics data
    file_path = os.path.join(eval_results_dir, json_file)
    with open(file_path, "r") as f:
        data = json.load(f)

    # Flatten the data into a list of question items
    all_items = []
    for key in data:
        all_items.extend(data[key])

    # Convert to DataFrame
    df = pd.DataFrame(all_items)

    # Convert category to numeric type
    df["category"] = pd.to_numeric(df["category"])

    # Calculate mean scores by category
    result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    # Add count of questions per category
    result["count"] = df.groupby("category").size()

    # Print the results
    print("Mean Scores Per Category:")
    print(result)

    # Calculate overall means
    overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    print("\nOverall Mean Scores:")
    print(overall_means)
    print("\n" + "="*60 + "\n")
