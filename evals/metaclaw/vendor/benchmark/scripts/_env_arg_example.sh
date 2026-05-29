#!/usr/bin/env bash
# _env_arg_example.sh - Environment variable template for MetaClaw benchmark scripts.
#
# Copy this file, fill in your values, and source it before running any script:
#
#   cp benchmark/scripts/_env_arg_example.sh ~/.metaclaw_bench_env.sh
#   # Edit ~/.metaclaw_bench_env.sh
#   source ~/.metaclaw_bench_env.sh
#
# See docs/scripts.md for a full variable reference.

# ---------------------------------------------------------------------------
# Required: LLM API credentials
# ---------------------------------------------------------------------------

# Base URL of the OpenAI-compatible API endpoint.
# Examples:
#   https://api.openai.com/v1
#   http://127.0.0.1:30000/v1  (local MetaClaw proxy)
export BENCHMARK_BASE_URL=https://api.openai.com/v1

# API key for the above endpoint.
export BENCHMARK_API_KEY=<your-api-key>

# Model ID as expected by the API server.
# This is injected into the openclaw agent config as the primary model.
# Examples: gpt-4o, Kimi-K2.5, claude-opus-4-6
export BENCHMARK_MODEL=<your-model-id>

# ---------------------------------------------------------------------------
# Required for RL scripts (rl_only_run.py, rl_run.py, etc.)
# ---------------------------------------------------------------------------

# API key for the Tinker RL fine-tuning service.
export TINKER_KEY=<your-tinker-key>

# Model ID used by the RL fine-tuning step.
export TINKER_MODEL=<your-tinker-model>

# Process Reward Model ID used for RL scoring.
export PRM_MODEL=<your-prm-model>

# ---------------------------------------------------------------------------
# Optional overrides
# ---------------------------------------------------------------------------

# Absolute path to the MetaClaw project root.
# Defaults to the grandparent of the scripts/ directory when unset.
# export METACLAW_ROOT=~/workspace/MetaClaw

# Explicit path to metaclaw-bench binary.  Defaults to 'metaclaw-bench' on PATH.
# export METACLAW_BENCH_BIN=/path/to/metaclaw-bench

# Explicit path to metaclaw binary (used by proxy_run.py).  Defaults to 'metaclaw' on PATH.
# export METACLAW_BIN=/path/to/metaclaw

# Shell script to source for loading credentials.  When set, the runner scripts
# will source this file instead of relying on the current shell environment.
# export METACLAW_API_KEY_SCRIPT=/path/to/load_keys.sh

# Directory containing pre-built skills (required for skills-based scripts).
# export METACLAW_SKILLS_DIR=~/workspace/metaclaw-data/skills

# Override the benchmark input file used by all runner scripts.
# Useful for switching between the full dataset and a smaller subset without
# editing individual scripts.  Must be an absolute path to an all_tests*.json file.
# Defaults to <METACLAW_ROOT>/benchmark/data/metaclaw-bench/all_tests_metaclaw.json
# (baseline_run.py defaults to all_tests.json in the same directory).
# export METACLAW_BENCH_INPUT=/path/to/benchmark/data/metaclaw-bench-small/all_tests_metaclaw.json

# ---------------------------------------------------------------------------
# Sanity check (optional — uncomment to print loaded values)
# ---------------------------------------------------------------------------
# echo "BENCHMARK_BASE_URL : ${BENCHMARK_BASE_URL}"
# echo "BENCHMARK_API_KEY  : ***"
# echo "BENCHMARK_MODEL    : ${BENCHMARK_MODEL}"
# echo "TINKER_MODEL       : ${TINKER_MODEL}"
# echo "PRM_MODEL          : ${PRM_MODEL}"
# echo "METACLAW_ROOT      : ${METACLAW_ROOT:-<auto-detect>}"
