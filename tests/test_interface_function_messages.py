"""How the interfaces render an agent's function messages.

The agent reported a failed validation as ``"Validation Error: ..."``, a prefix
no interface recognizes: ``CLIInterface`` printed "did not recognize function
message" to stdout (a bare ``print``) and ``QueuingInterface`` queued it as a
generic message instead of an error. The agent now sends ``"Error: "`` (tested
on the agent in ``test_tool_call_plan.py``); an unrecognized message never goes
to stdout. Offline and pure.
"""

from mirix.interface import CLIInterface, QueuingInterface
from mirix.schemas.message import Message


def test_the_cli_renders_an_error_without_printing_to_stdout(capsys):
    CLIInterface.function_message("Error: Validation error: 'name' cannot be empty.")
    assert "did not recognize" not in capsys.readouterr().out


def test_the_cli_logs_an_unrecognized_message_instead_of_printing_it(capsys):
    CLIInterface.function_message("Validation Error: 'name' cannot be empty.")
    assert capsys.readouterr().out == ""


def test_the_queue_marks_a_validation_error_as_an_error():
    interface = QueuingInterface(debug=False)
    msg_obj = Message.dict_to_message(
        agent_id="agent-1",
        model="fake-model",
        openai_message_dict={
            "role": "tool",
            "name": "skill_create",
            "content": "x",
            "tool_call_id": "c1",
        },
    )
    interface.function_message(
        "Error: Validation error: 'name' cannot be empty.", msg_obj=msg_obj
    )
    queued = interface.buffer.get_nowait()
    message = queued.get("message_api") if isinstance(queued, dict) else queued
    assert message["status"] == "error"
    assert message["function_return"] == "Validation error: 'name' cannot be empty."
