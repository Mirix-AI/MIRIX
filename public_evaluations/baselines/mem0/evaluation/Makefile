# Run the experiments
run-mem0-add:
	python run_experiments.py --technique_type mem0 --method add

run-mem0-search:
	python run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30

run-mem0-plus-add:
	python run_experiments.py --technique_type mem0 --method add --is_graph

run-mem0-plus-search:
	python run_experiments.py --technique_type mem0 --method search --is_graph --output_folder results/ --top_k 30

run-rag:
	python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 1 --output_folder results/

run-full-context:
	python run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/

run-langmem:
	python run_experiments.py --technique_type langmem --output_folder results/

run-zep-add:
	python run_experiments.py --technique_type zep --method add --output_folder results/ --rate_limit_delay 2.0 --max_retries 5

run-zep-add-slow:
	python run_experiments.py --technique_type zep --method add --output_folder results/ --rate_limit_delay 3.0 --max_retries 8

run-zep-search:
	python run_experiments.py --technique_type zep --method search --output_folder results/ --rate_limit_delay 2.0 --max_retries 8

# Enhanced zep-search with better error handling and longer delays
run-zep-search-robust:
	python run_experiments.py --technique_type zep --method search --output_folder results/ --rate_limit_delay 2.0 --max_retries 8

# Resume zep-search from existing results (alias for robust version since resume is built-in now)
run-zep-search-resume:
	python run_experiments.py --technique_type zep --method search --output_folder results/ --rate_limit_delay 2.0 --max_retries 8

run-openai:
	python run_experiments.py --technique_type openai --output_folder results/

# Check progress of zep search experiments
check-zep-progress:
	python check_zep_progress.py
