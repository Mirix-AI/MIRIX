import os
import sys
import contextlib
import subprocess

from conversation_creator import ConversationCreator
from datetime import datetime
from constants import CHUNK_SIZE_MEMORY_AGENT_BENCH


def compute_run_out_dir(args, global_idx):
    """Compute absolute output directory for a specific run to place logs alongside results.

    This mirrors the directory structure in `run_instance.py` so logs align with results.
    """
    # Determine subset name and chunk size
    if args.dataset == 'MemoryAgentBench':
        conversation_creator = ConversationCreator(args.dataset, args.num_exp, args.sub_datasets)
        all_queries_and_answers = conversation_creator.get_query_and_answer()
        if global_idx >= len(all_queries_and_answers):
            raise ValueError(
                f"global_idx {global_idx} out of range for queries/answers (n={len(all_queries_and_answers)})"
            )
        queries_and_answers = all_queries_and_answers[global_idx]
        subset_name = queries_and_answers[0][3]
        chunk_size = CHUNK_SIZE_MEMORY_AGENT_BENCH[subset_name]
    else:
        subset_name = "None"
        chunk_size = "None"

    # Parent folder name matches run_instance.py
    if args.agent_name == 'gpt-long-context' or args.agent_name == 'gemini-long-context':
        parent_folder = f"{args.agent_name}_{args.dataset}-{args.model_name}"
    else:
        parent_folder = f"{args.agent_name}_{args.dataset}-model{args.model_name}"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_out_dir = os.path.join(
        base_dir,
        "logs",
        parent_folder,
        f"{global_idx}_subset{subset_name}_cksize{chunk_size}"
    )

    # Ensure the directory exists so logs can be created before child writes results
    os.makedirs(abs_out_dir, exist_ok=True)
    return abs_out_dir


def setup_logs_for_run(args, global_idx):
    """Ensure logs directory exists for this run and return log file paths.

    Returns (out_dir_abs, logs_dir, parent_log_path, child_stdout_path, child_stderr_path)
    """
    out_dir_abs = compute_run_out_dir(args, global_idx)
    logs_dir = os.path.join(out_dir_abs, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    parent_log_path = os.path.join(logs_dir, "parent.log")
    child_stdout_path = os.path.join(logs_dir, "child_stdout.log")
    child_stderr_path = os.path.join(logs_dir, "child_stderr.log")

    return out_dir_abs, logs_dir, parent_log_path, child_stdout_path, child_stderr_path


class LogsPaths:
    def __init__(self, parent_log_path, child_log_path):
        self.parent = parent_log_path
        self.child = child_log_path


def prepare_logs_paths(args, global_idx):
    """Prepare and return simplified logs paths object with combined child log.

    Returns LogsPaths with .parent and .child attributes.
    """
    _, _, parent_log_path, child_stdout_path, _ = setup_logs_for_run(args, global_idx)
    logs_dir = os.path.dirname(child_stdout_path)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    child_log_path = os.path.join(logs_dir, f"child_{ts}.log")
    parent_log_path = os.path.join(os.path.dirname(parent_log_path), f"parent_{ts}.log")
    # Ensure parent and child files' directories exist
    os.makedirs(os.path.dirname(parent_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(child_log_path), exist_ok=True)
    return LogsPaths(parent_log_path, child_log_path)


class _Tee:
    def __init__(self, primary_stream, secondary_stream):
        self.primary_stream = primary_stream
        self.secondary_stream = secondary_stream
    def write(self, data):
        try:
            self.primary_stream.write(data)
        except Exception:
            pass
        try:
            self.secondary_stream.write(data)
        except Exception:
            pass
    def flush(self):
        try:
            self.primary_stream.flush()
        except Exception:
            pass
        try:
            self.secondary_stream.flush()
        except Exception:
            pass
    def isatty(self):
        try:
            return self.primary_stream.isatty()
        except Exception:
            return False
    def fileno(self):
        return self.primary_stream.fileno()


@contextlib.contextmanager
def tee_parent_logs(parent_log_path):
    """Context manager to tee the parent process stdout/stderr to a log file."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # Ensure parent log file directory exists
    os.makedirs(os.path.dirname(parent_log_path), exist_ok=True)
    with open(parent_log_path, 'a', encoding='utf-8') as parent_fh:
        sys.stdout = _Tee(original_stdout, parent_fh)
        sys.stderr = _Tee(original_stderr, parent_fh)
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_subprocess_with_logs(cmd, cwd, stdout_log_path, stderr_log_path, combine_streams=False):
    """Run a subprocess redirecting stdout/stderr to the given log files.

    If combine_streams is True, stderr is redirected to stdout and only stdout_log_path is used.
    """
    # Ensure directories exist
    if stdout_log_path:
        os.makedirs(os.path.dirname(stdout_log_path), exist_ok=True)
    if not combine_streams and stderr_log_path:
        os.makedirs(os.path.dirname(stderr_log_path), exist_ok=True)

    stdout_fh = open(stdout_log_path if stdout_log_path else os.devnull, 'a', encoding='utf-8')
    stderr_fh = None
    try:
        if combine_streams:
            return subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                stdout=stdout_fh,
                stderr=subprocess.STDOUT,
            )
        else:
            stderr_fh = open(stderr_log_path if stderr_log_path else os.devnull, 'a', encoding='utf-8')
            return subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                stdout=stdout_fh,
                stderr=stderr_fh,
            )
    finally:
        try:
            stdout_fh.close()
        except Exception:
            pass
        if stderr_fh is not None:
            try:
                stderr_fh.close()
            except Exception:
                pass


