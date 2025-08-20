# MIRIX Evaluation Code
> [!WARNING]
> If you are using MIRIX desktop app and have your own memory built using sqlite, please do not run the evaluation and do not run the code `rm ~/.mirix/sqlite.db`. It is the safest to run the experiments on a separate server.
> 
> If you really need to run on your local machine, please make sure either of the following requirements are met:  
> (1) You are using `postgresql` for your desktop application. Ensure that the `.env` file in the `public_evaluations` folder does not contain the PostgreSQL URL. 
> (2) Follow the instructions in [@https://docs.mirix.io/advanced/backup-restore/](https://docs.mirix.io/advanced/backup-restore/) to backup your current memory. 


## LOCOMO Experiments

### MIRIX
Download the dataset from the following link (source: [Mem0](https://github.com/mem0ai/mem0/tree/main/evaluation)):

[locomo.json](https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-)

Create the results directory by running `mkdir results`. The required file structure should be:
```
main.py
agent.py
conversation_creator.py
README.md
data/
    locomo10.json
results/
.env
```

Save your `OPENAI_API_KEY` in the `.env` file:
```
OPENAI_API_KEY=sk-xxxxxx
```
Clear the existing memory states:
```
rm -r ~/.mirix/sqlite.db
```
Generate the responses by running:
```
python main.py --agent_name mirix --dataset LOCOMO
```
Generate evaluation scores using `evals.py`:
```
python evals.py --input_file results/mirix_LOCOMO --output_file results/mirix_LOCOMO/evaluation_metrics.json
```

> **Note**: This evaluation uses `gpt-4.1-mini` instead of `gemini-2.5-flash` (used in the main branch) to ensure fair comparison. The `search_method` is set to `embedding` with OpenAI's `text-embed-3-small` as the embedding model. For LOCOMO, `text-embed-3-small` demonstrates slightly better performance compared to `bm25` search. 


### Baselines

**Zep**: Following their [Official Implementation](https://github.com/getzep/zep-papers), experiments were re-run using `gpt-4.1-mini` with results uploaded to `public_evaluations/baselines/zep-papers/kg_architecture_agent_memory/locomo_eval/data/zep_locomo_grades_with_categories.json`. 

**Mem0 and LangMem**: Following the implementation in [Mem0](https://github.com/mem0ai/mem0/tree/main/evaluation), the environment variable `model` was changed to `gpt-4.1-mini` and all experiments were re-run as instructed on their webpage. 

