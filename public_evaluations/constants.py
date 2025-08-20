"""Shared constants for public_evaluations.

This module centralizes configuration values used across multiple modules to
avoid circular imports (e.g., between main.py and conversation_creator.py).
"""

# CONSTANTS for chunk size used by MemoryAgentBench sub-datasets
CHUNK_SIZE_MEMORY_AGENT_BENCH = {
    # AR
    'ruler_qa1_197K': 4096, #512,
    'ruler_qa2_421K': 4096, #512,
    'longmemeval_s*': 4096, #512,
    'eventqa_full': 4096,
    # ICL
    'icl_banking77_5900shot_balance': 4096,
    'icl_clinic150_7050shot_balance': 4096,
    'recsys_redial_full': 4096,
    # CR
    'factconsolidation_mh_262k': 4096, #512,
    'factconsolidation_sh_262k': 4096, #512,
}


