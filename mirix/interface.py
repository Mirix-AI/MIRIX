import re
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Literal

from colorama import Fore, Style, init
import queue
import asyncio

from mirix.constants import CLI_WARNING_PREFIX, ASSISTANT_MESSAGE_CLI_SYMBOL, INNER_THOUGHTS_CLI_SYMBOL
from mirix.schemas.message import Message
from mirix.utils import json_loads, printd, is_utc_datetime

init(autoreset=True)

# DEBUG = True  # puts full message outputs in the terminal
DEBUG = False  # only dumps important messages in the terminal

STRIP_UI = False


class AgentInterface(ABC):
    """Interfaces handle Mirix-related events (observer pattern)

    The 'msg' args provides the scoped message, and the optional Message arg can provide additional metadata.
    """

    @abstractmethod
    def user_message(self, msg: str, msg_obj: Optional[Message] = None):
        """Mirix receives a user message"""
        raise NotImplementedError

    @abstractmethod
    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None):
        """Mirix generates some internal monologue"""
        raise NotImplementedError

    @abstractmethod
    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None):
        """Mirix uses send_message"""
        raise NotImplementedError

    @abstractmethod
    def function_message(self, msg: str, msg_obj: Optional[Message] = None):
        """Mirix calls a function"""
        raise NotImplementedError

class CLIInterface(AgentInterface):
    """Basic interface for dumping agent events to the command-line"""

    @staticmethod
    def important_message(msg: str):
        fstr = f"{Fore.MAGENTA}{Style.BRIGHT}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def warning_message(msg: str):
        fstr = f"{Fore.RED}{Style.BRIGHT}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        else:
            print(fstr.format(msg=msg))

    @staticmethod
    def internal_monologue(msg: str, msg_obj: Optional[Message] = None):
        # ANSI escape code for italic is '\x1B[3m'
        fstr = f"\x1B[3m{Fore.LIGHTBLACK_EX}{INNER_THOUGHTS_CLI_SYMBOL} {{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def assistant_message(msg: str, msg_obj: Optional[Message] = None):
        fstr = f"{Fore.YELLOW}{Style.BRIGHT}{ASSISTANT_MESSAGE_CLI_SYMBOL} {Fore.YELLOW}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def memory_message(msg: str, msg_obj: Optional[Message] = None):
        fstr = f"{Fore.LIGHTMAGENTA_EX}{Style.BRIGHT}🧠 {Fore.LIGHTMAGENTA_EX}{{msg}}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def system_message(msg: str, msg_obj: Optional[Message] = None):
        fstr = f"{Fore.MAGENTA}{Style.BRIGHT}🖥️ [system] {Fore.MAGENTA}{msg}{Style.RESET_ALL}"
        if STRIP_UI:
            fstr = "{msg}"
        print(fstr.format(msg=msg))

    @staticmethod
    def user_message(msg: str, msg_obj: Optional[Message] = None, raw: bool = False, dump: bool = False, debug: bool = DEBUG):
        def print_user_message(icon, msg, printf=print):
            if STRIP_UI:
                printf(f"{icon} {msg}")
            else:
                printf(f"{Fore.GREEN}{Style.BRIGHT}{icon} {Fore.GREEN}{msg}{Style.RESET_ALL}")

        def printd_user_message(icon, msg):
            return print_user_message(icon, msg)

        if not (raw or dump or debug):
            # we do not want to repeat the message in normal use
            return

        if isinstance(msg, str):
            if raw:
                printd_user_message("🧑", msg)
                return
            else:
                try:
                    msg_json = json_loads(msg)
                except:
                    printd(f"{CLI_WARNING_PREFIX}failed to parse user message into json")
                    printd_user_message("🧑", msg)
                    return
        if msg_json["type"] == "user_message":
            if dump:
                print_user_message("🧑", msg_json["message"])
                return
            msg_json.pop("type")
            printd_user_message("🧑", msg_json)
        elif msg_json["type"] == "contine_chaining":
            if debug:
                msg_json.pop("type")
                printd_user_message("💓", msg_json)
            elif dump:
                print_user_message("💓", msg_json)
                return

        elif msg_json["type"] == "system_message":
            msg_json.pop("type")
            printd_user_message("🖥️", msg_json)
        else:
            printd_user_message("🧑", msg_json)

    @staticmethod
    def function_message(msg: str, msg_obj: Optional[Message] = None, debug: bool = DEBUG):
        def print_function_message(icon, msg, color=Fore.RED, printf=print):
            if STRIP_UI:
                printf(f"⚡{icon} [function] {msg}")
            else:
                printf(f"{color}{Style.BRIGHT}⚡{icon} [function] {color}{msg}{Style.RESET_ALL}")

        def printd_function_message(icon, msg, color=Fore.RED):
            return print_function_message(icon, msg, color, printf=(print if debug else printd))

        if isinstance(msg, dict):
            printd_function_message("", msg)
            return

        if msg.startswith("Success"):
            printd_function_message("🟢", msg)
        elif msg.startswith("Error: "):
            printd_function_message("🔴", msg)
        elif msg.startswith("Ran "):
            # NOTE: ignore 'ran' messages that come post-execution
            return
        elif msg.startswith("Running "):
            if debug:
                printd_function_message("", msg)
            else:
                match = re.search(r"Running (\w+)\((.*)\)", msg)
                if match:
                    function_name = match.group(1)
                    function_args = match.group(2)
                    if function_name in ["archival_memory_insert", "archival_memory_search", "core_memory_append"]:
                        if function_name in ["archival_memory_insert", "core_memory_append"]:
                            print_function_message("🧠", f"updating memory with {function_name}")
                        elif function_name == "archival_memory_search":
                            print_function_message("🧠", f"searching memory with {function_name}")
                        try:
                            msg_dict = eval(function_args)
                            if function_name == "archival_memory_search":
                                output = f'\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}'
                                if STRIP_UI:
                                    print(output)
                                else:
                                    print(f"{Fore.RED}{output}{Style.RESET_ALL}")
                            elif function_name == "archival_memory_insert":
                                output = f'\t→ {msg_dict["content"]}'
                                if STRIP_UI:
                                    print(output)
                                else:
                                    print(f"{Style.BRIGHT}{Fore.RED}{output}{Style.RESET_ALL}")
                            else:
                                if STRIP_UI:
                                    print(f'\t {msg_dict["old_content"]}\n\t→ {msg_dict["new_content"]}')
                                else:
                                    print(
                                        f'{Style.BRIGHT}\t{Fore.RED} {msg_dict["old_content"]}\n\t{Fore.GREEN}→ {msg_dict["new_content"]}{Style.RESET_ALL}'
                                    )
                        except Exception as e:
                            printd(str(e))
                            printd(msg_dict)
                    elif function_name in ["conversation_search", "conversation_search_date"]:
                        print_function_message("🧠", f"searching memory with {function_name}")
                        try:
                            msg_dict = eval(function_args)
                            output = f'\tquery: {msg_dict["query"]}, page: {msg_dict["page"]}'
                            if STRIP_UI:
                                print(output)
                            else:
                                print(f"{Fore.RED}{output}{Style.RESET_ALL}")
                        except Exception as e:
                            printd(str(e))
                            printd(msg_dict)
                else:
                    printd(f"{CLI_WARNING_PREFIX}did not recognize function message")
                    printd_function_message("", msg)
        else:
            try:
                msg_dict = json_loads(msg)
                if "status" in msg_dict and msg_dict["status"] == "OK":
                    printd_function_message("", str(msg), color=Fore.GREEN)
                else:
                    printd_function_message("", str(msg), color=Fore.RED)
            except Exception:
                print(f"{CLI_WARNING_PREFIX}did not recognize function message {type(msg)} {msg}")
                printd_function_message("", msg)

    @staticmethod
    def print_messages(message_sequence: List[Message], dump=False):
        # rewrite to dict format
        message_sequence = [msg.to_openai_dict() for msg in message_sequence]

        idx = len(message_sequence)
        for msg in message_sequence:
            if dump:
                print(f"[{idx}] ", end="")
                idx -= 1
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                CLIInterface.system_message(content)
            elif role == "assistant":
                # Differentiate between internal monologue, function calls, and messages
                if msg.get("function_call"):
                    if content is not None:
                        CLIInterface.internal_monologue(content)
                    # I think the next one is not up to date
                    # function_message(msg["function_call"])
                    args = json_loads(msg["function_call"].get("arguments"))
                    CLIInterface.assistant_message(args.get("message"))
                    # assistant_message(content)
                elif msg.get("tool_calls"):
                    if content is not None:
                        CLIInterface.internal_monologue(content)
                    function_obj = msg["tool_calls"][0].get("function")
                    if function_obj:
                        args = json_loads(function_obj.get("arguments"))
                        CLIInterface.assistant_message(args.get("message"))
                else:
                    CLIInterface.internal_monologue(content)
            elif role == "user":
                CLIInterface.user_message(content, dump=dump)
            elif role == "function":
                CLIInterface.function_message(content, debug=dump)
            elif role == "tool":
                CLIInterface.function_message(content, debug=dump)
            else:
                print(f"Unknown role: {content}")

    @staticmethod
    def print_messages_simple(message_sequence: List[Message]):
        # rewrite to dict format
        message_sequence = [msg.to_openai_dict() for msg in message_sequence]

        for msg in message_sequence:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                CLIInterface.system_message(content)
            elif role == "assistant":
                CLIInterface.assistant_message(content)
            elif role == "user":
                CLIInterface.user_message(content, raw=True)
            else:
                print(f"Unknown role: {content}")

    @staticmethod
    def print_messages_raw(message_sequence: List[Message]):
        # rewrite to dict format
        message_sequence = [msg.to_openai_dict() for msg in message_sequence]

        for msg in message_sequence:
            print(msg)

    @staticmethod
    def step_yield():
        pass

    @staticmethod
    def step_complete():
        pass


class QueuingInterface(AgentInterface):
    """Messages are queued inside an internal buffer and manually flushed"""

    def __init__(self, debug=True):
        self.buffer = queue.Queue()
        self.debug = debug

    def _queue_push(self, message_api: Union[str, dict], message_obj: Union[Message, None]):
        """Wrapper around self.buffer.queue.put() that ensures the types are safe

        Data will be in the format: {
            "message_obj": ...
            "message_string": ...
        }
        """

        # Check the string first

        if isinstance(message_api, str):
            # check that it's the stop word
            if message_api == "STOP":
                assert message_obj is None
                self.buffer.put(
                    {
                        "message_api": message_api,
                        "message_obj": None,
                    }
                )
            else:
                raise ValueError(f"Unrecognized string pushed to buffer: {message_api}")

        elif isinstance(message_api, dict):
            # check if it's the error message style
            if len(message_api.keys()) == 1 and "internal_error" in message_api:
                assert message_obj is None
                self.buffer.put(
                    {
                        "message_api": message_api,
                        "message_obj": None,
                    }
                )
            else:
                assert message_obj is not None, message_api
                self.buffer.put(
                    {
                        "message_api": message_api,
                        "message_obj": message_obj,
                    }
                )

        else:
            raise ValueError(f"Unrecognized type pushed to buffer: {type(message_api)}")

    def to_list(self, style: Literal["obj", "api"] = "obj"):
        """Convert queue to a list (empties it out at the same time)"""
        items = []
        while not self.buffer.empty():
            try:
                # items.append(self.buffer.get_nowait())
                item_to_push = self.buffer.get_nowait()
                if style == "obj":
                    if item_to_push["message_obj"] is not None:
                        items.append(item_to_push["message_obj"])
                elif style == "api":
                    items.append(item_to_push["message_api"])
                else:
                    raise ValueError(style)
            except queue.Empty:
                break
        if len(items) > 1 and items[-1] == "STOP":
            items.pop()

        # If the style is "obj", then we need to deduplicate any messages
        # Filter down items for duplicates based on item.id
        if style == "obj":
            seen_ids = set()
            unique_items = []
            for item in reversed(items):
                if item.id not in seen_ids:
                    seen_ids.add(item.id)
                    unique_items.append(item)
            items = list(reversed(unique_items))

        return items

    def clear(self):
        """Clear all messages from the queue."""
        with self.buffer.mutex:
            # Empty the queue
            self.buffer.queue.clear()

    async def message_generator(self, style: Literal["obj", "api"] = "obj"):
        while True:
            if not self.buffer.empty():
                message = self.buffer.get()
                message_obj = message["message_obj"]
                message_api = message["message_api"]

                if message_api == "STOP":
                    break

                # yield message
                if style == "obj":
                    yield message_obj
                elif style == "api":
                    yield message_api
                else:
                    raise ValueError(style)

            else:
                await asyncio.sleep(0.1)  # Small sleep to prevent a busy loop

    def step_yield(self):
        """Enqueue a special stop message"""
        self._queue_push(message_api="STOP", message_obj=None)

    @staticmethod
    def step_complete():
        pass

    def error(self, error: str):
        """Enqueue a special stop message"""
        self._queue_push(message_api={"internal_error": error}, message_obj=None)
        self._queue_push(message_api="STOP", message_obj=None)

    def user_message(self, msg: str, msg_obj: Optional[Message] = None):
        """Handle reception of a user message"""
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"
        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())

    def internal_monologue(self, msg: str, msg_obj: Optional[Message] = None) -> None:
        """Handle the agent's internal monologue"""
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"
        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())

        new_message = {"internal_monologue": msg}

        # add extra metadata
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()

        self._queue_push(message_api=new_message, message_obj=msg_obj)

    def assistant_message(self, msg: str, msg_obj: Optional[Message] = None) -> None:
        """Handle the agent sending a message"""
        # assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"

        if self.debug:
            print(msg)
            if msg_obj is not None:
                print(vars(msg_obj))
                print(msg_obj.created_at.isoformat())

        new_message = {"assistant_message": msg}

        # add extra metadata
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()
        else:
            new_message["id"] = self.buffer.queue[-1]["message_api"]["id"]
            # assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = self.buffer.queue[-1]["message_api"]["date"]

            msg_obj = self.buffer.queue[-1]["message_obj"]

        self._queue_push(message_api=new_message, message_obj=msg_obj)

    def function_message(self, msg: str, msg_obj: Optional[Message] = None, include_ran_messages: bool = False) -> None:
        """Handle the agent calling a function"""
        # TODO handle 'function' messages that indicate the start of a function call
        assert msg_obj is not None, "QueuingInterface requires msg_obj references for metadata"

        if self.debug:
            print(msg)
            print(vars(msg_obj))
            print(msg_obj.created_at.isoformat())

        if msg.startswith("Running "):
            msg = msg.replace("Running ", "")
            new_message = {"function_call": msg}

        elif msg.startswith("Ran "):
            if not include_ran_messages:
                return
            msg = msg.replace("Ran ", "Function call returned: ")
            new_message = {"function_call": msg}

        elif msg.startswith("Success: "):
            msg = msg.replace("Success: ", "")
            new_message = {"function_return": msg, "status": "success"}

        elif msg.startswith("Error: "):
            msg = msg.replace("Error: ", "", 1)
            new_message = {"function_return": msg, "status": "error"}

        else:
            # NOTE: generic, should not happen
            new_message = {"function_message": msg}

        # add extra metadata
        if msg_obj is not None:
            new_message["id"] = str(msg_obj.id)
            assert is_utc_datetime(msg_obj.created_at), msg_obj.created_at
            new_message["date"] = msg_obj.created_at.isoformat()

        self._queue_push(message_api=new_message, message_obj=msg_obj)

