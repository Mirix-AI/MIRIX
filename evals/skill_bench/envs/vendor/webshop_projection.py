# Vendored from verl-agent (Apache-2.0 License)
# Original: agent_system/environments/env_package/webshop/projection.py
#
# Copied from langfengQ/verl-agent 20bd331bdbc9026a5668e11362178e10ab7400c8
# (sha256 0808d4643459acf2a4c496eab3fc8735025c440f28cc517066355a1f5fbc048f; byte-identical
# in aiming-lab/SkillRL 4d984d4). Copyright 2025 Nanyang Technological University (NTU),
# Singapore and the verl-agent (GiGPO) team.
# Modified by skill_bench: reformatted with ruff (quote style, line breaks), the bare
# ``except:`` written as ``except Exception:`` and type hints use built-in generics. The
# logic is unchanged; tests compare it with the original on a corpus of responses. See
# NOTICE in this directory.
#
# skill_bench calls it only on a response that holds an <action>...</action> pair: the
# fallback of line "actions[i] = actions[i][-20:]" (a reply without the tags sends its last
# 20 characters to the environment) is never reached (envs/webshop.py).

import re


def webshop_projection(actions: list[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        <think>some reasoning...</think><action>up/down/left/right/still</action>
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-20:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = (
                actions[i][start_idx + len(start_tag) : end_idx].strip().lower()
            )

            actions[i] = extracted_action
            valids[i] = 1

        except Exception:
            # randomly choose an action from the action list if illegal
            actions[i] = actions[i][-20:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r"[一-鿿]", original_str):
            valids[i] = 0

    return actions, valids
