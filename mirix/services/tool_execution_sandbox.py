import asyncio
import ast
import base64
import os
import pickle
import sys
import tempfile
import traceback
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

from mirix.log import get_logger
from mirix.observability.context import get_trace_context
from mirix.observability.langfuse_client import get_langfuse_client

if TYPE_CHECKING:
    try:
        from e2b_code_interpreter import AsyncSandbox, Execution
    except ImportError:
        Execution = Any  # type: ignore
        AsyncSandbox = Any  # type: ignore
from mirix.schemas.agent import AgentState
from mirix.schemas.client import Client
from mirix.schemas.sandbox_config import SandboxConfig, SandboxRunResult, SandboxType
from mirix.schemas.tool import Tool
from mirix.services.tool_manager import ToolManager
from mirix.settings import tool_settings
from mirix.utils import get_friendly_error_msg

logger = get_logger(__name__)


class ToolExecutionSandbox:
    METADATA_CONFIG_STATE_KEY = "config_state"
    REQUIREMENT_TXT_NAME = "requirements.txt"

    # For generating long, random marker hashes
    NAMESPACE = uuid.NAMESPACE_DNS
    LOCAL_SANDBOX_RESULT_START_MARKER = str(uuid.uuid5(NAMESPACE, "local-sandbox-result-start-marker"))
    LOCAL_SANDBOX_RESULT_END_MARKER = str(uuid.uuid5(NAMESPACE, "local-sandbox-result-end-marker"))

    # This is the variable name in the auto-generated code that contains the function results
    # We make this a long random string to avoid collisions with any variables in the user's code
    LOCAL_SANDBOX_RESULT_VAR_NAME = "result_ZQqiequkcFwRwwGQMqkt"

    def __init__(
        self,
        tool_name: str,
        args: dict,
        actor: Client,
        force_recreate=True,
        tool_object: Optional[Tool] = None,
    ):
        self.tool_name = tool_name
        self.args = args
        self.actor = actor
        self.tool = tool_object
        self.force_recreate = force_recreate

    async def _ensure_tool(self) -> None:
        """Resolve the tool object asynchronously if not provided at init."""
        if self.tool is None:
            self.tool = await ToolManager().get_tool_by_name(tool_name=self.tool_name, actor=self.actor)
            if not self.tool:
                raise ValueError(
                    f"Agent attempted to invoke tool {self.tool_name} that does not exist for organization {self.actor.organization_id}"
                )

    async def run(
        self,
        agent_state: Optional[AgentState] = None,
        additional_env_vars: Optional[Dict] = None,
    ) -> SandboxRunResult:
        """
        Run the tool in a sandbox environment.

        Args:
            agent_state (Optional[AgentState]): The state of the agent invoking the tool
            additional_env_vars (Optional[Dict]): Environment variables to inject into the sandbox

        Returns:
            SandboxRunResult containing tool result and agent state
        """
        sandbox_type = "e2b" if tool_settings.e2b_api_key else "local"

        langfuse = get_langfuse_client()
        trace_context = get_trace_context() if langfuse else {}
        trace_id = trace_context.get("trace_id") if trace_context else None
        parent_span_id = trace_context.get("observation_id") if trace_context else None

        async def _execute_tool() -> SandboxRunResult:
            if tool_settings.e2b_api_key:
                logger.debug("Using e2b sandbox to execute %s", self.tool_name)
                return await self.run_e2b_sandbox(agent_state=agent_state, additional_env_vars=additional_env_vars)
            else:
                logger.debug("Using local sandbox to execute %s", self.tool_name)
                return await self.run_local_dir_sandbox(agent_state=agent_state, additional_env_vars=additional_env_vars)

        if langfuse and trace_id:
            from typing import cast

            from langfuse.types import TraceContext

            trace_context_dict: dict = {"trace_id": trace_id}
            if parent_span_id:
                trace_context_dict["parent_span_id"] = parent_span_id

            args_for_trace = {key: str(value) for key, value in self.args.items()}

            try:
                with langfuse.start_as_current_observation(
                    name=f"tool_execution: {self.tool_name}",
                    as_type="tool",
                    trace_context=cast(TraceContext, trace_context_dict),
                    input={"tool_name": self.tool_name, "args": args_for_trace},
                    metadata={
                        "sandbox_type": sandbox_type,
                        "tool_name": self.tool_name,
                    },
                ) as span:
                    result = await _execute_tool()

                    span.update(
                        output={
                            "status": result.status,
                            "has_stdout": bool(result.stdout),
                            "has_stderr": bool(result.stderr),
                        },
                        metadata={
                            "sandbox_type": sandbox_type,
                            "tool_name": self.tool_name,
                            "status": result.status,
                        },
                        level="ERROR" if result.status == "error" else "DEFAULT",
                    )
            except Exception as e:
                logger.debug(f"Langfuse tool execution trace failed: {e}")
                result = await _execute_tool()
        else:
            result = await _execute_tool()

        logger.debug(f"Executed tool '{self.tool_name}', logging output from tool run: \n")
        for log_line in (result.stdout or []) + (result.stderr or []):
            logger.debug("%s", log_line)
        logger.debug("Ending output log from tool run.")

        return result

    async def run_local_dir_sandbox(
        self,
        agent_state: Optional[AgentState] = None,
        additional_env_vars: Optional[Dict] = None,
    ) -> SandboxRunResult:
        sbx_config = self.sandbox_config_manager.get_or_create_default_sandbox_config(
            sandbox_type=SandboxType.LOCAL, actor=self.actor
        )
        local_configs = sbx_config.get_local_config()

        env = os.environ.copy()
        env_vars = self.sandbox_config_manager.get_sandbox_env_vars_as_dict(
            sandbox_config_id=sbx_config.id, actor=self.actor, limit=100
        )
        env.update(env_vars)

        if agent_state:
            env.update(agent_state.get_agent_env_vars_as_dict())

        if additional_env_vars:
            env.update(additional_env_vars)

        if not os.path.exists(local_configs.sandbox_dir) or not os.path.isdir(local_configs.sandbox_dir):
            logger.warning(f"Sandbox directory does not exist, creating: {local_configs.sandbox_dir}")
            os.makedirs(local_configs.sandbox_dir)

        with tempfile.NamedTemporaryFile(
            mode="w", dir=local_configs.sandbox_dir, suffix=".py", delete=False
        ) as temp_file:
            code = self.generate_execution_script(
                agent_state=agent_state, wrap_print_with_markers=True
            )
            temp_file.write(code)
            temp_file.flush()
            temp_file_path = temp_file.name

        try:
            if local_configs.use_venv:
                return await self.run_local_dir_sandbox_venv(sbx_config, env, temp_file_path)
            else:
                return await self.run_local_dir_sandbox_runpy(sbx_config, env, temp_file_path)
        except Exception as e:
            logger.error(f"Executing tool {self.tool_name} has an unexpected error: {e}")
            logger.error(f"Logging out tool {self.tool_name} auto-generated code for debugging: \n\n{code}")
            raise e
        finally:
            os.remove(temp_file_path)

    async def run_local_dir_sandbox_venv(
        self, sbx_config: SandboxConfig, env: Dict[str, str], temp_file_path: str
    ) -> SandboxRunResult:
        local_configs = sbx_config.get_local_config()
        venv_path = os.path.join(local_configs.sandbox_dir, local_configs.venv_name)

        if not os.path.isdir(venv_path):
            logger.warning(f"Virtual environment directory does not exist at: {venv_path}, creating one now...")
            await self.create_venv_for_local_sandbox(
                sandbox_dir_path=local_configs.sandbox_dir, venv_path=venv_path, env=env
            )

        python_executable = os.path.join(venv_path, "bin", "python3")
        if not os.path.isfile(python_executable):
            raise FileNotFoundError(f"Python executable not found in virtual environment: {python_executable}")

        env["VIRTUAL_ENV"] = venv_path
        env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
        env["PYTHONWARNINGS"] = "ignore"

        try:
            process = await asyncio.create_subprocess_exec(
                os.path.join(venv_path, "bin", "python3"),
                temp_file_path,
                env=env,
                cwd=local_configs.sandbox_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=60
            )
            stdout_text = stdout_bytes.decode() if stdout_bytes else ""
            stderr_text = stderr_bytes.decode() if stderr_bytes else ""

            if process.returncode != 0:
                logger.error("Executing tool %s failed with return code %d", self.tool_name, process.returncode)
                func_return = get_friendly_error_msg(
                    function_name=self.tool_name,
                    exception_name="SubprocessError",
                    exception_message=f"Process exited with code {process.returncode}: {stderr_text}",
                )
                return SandboxRunResult(
                    func_return=func_return,
                    agent_state=None,
                    stdout=[stdout_text] if stdout_text else [],
                    stderr=[stderr_text] if stderr_text else [],
                    status="error",
                    sandbox_config_fingerprint=sbx_config.fingerprint(),
                )

            func_result, stdout_parsed = self.parse_out_function_results_markers(stdout_text)
            func_return, agent_state = self.parse_best_effort(func_result)
            return SandboxRunResult(
                func_return=func_return,
                agent_state=agent_state,
                stdout=[stdout_parsed] if stdout_parsed else [],
                stderr=[stderr_text] if stderr_text else [],
                status="success",
                sandbox_config_fingerprint=sbx_config.fingerprint(),
            )

        except asyncio.TimeoutError:
            raise TimeoutError(f"Executing tool {self.tool_name} has timed out.")

        except Exception as e:
            logger.error(f"Executing tool {self.tool_name} has an unexpected error: {e}")
            raise e

    async def run_local_dir_sandbox_runpy(
        self, sbx_config: SandboxConfig, env: Dict[str, str], temp_file_path: str
    ) -> SandboxRunResult:
        """Run tool script via async subprocess (no thread pool, no GIL contention)."""
        cwd = os.path.dirname(temp_file_path)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_file_path,
                env=env,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=60
            )
            stdout_text = stdout_bytes.decode() if stdout_bytes else ""
            stderr_text = stderr_bytes.decode() if stderr_bytes else ""

            if process.returncode != 0:
                logger.error(
                    "Executing tool %s failed with return code %d",
                    self.tool_name, process.returncode,
                )
                func_return = get_friendly_error_msg(
                    function_name=self.tool_name,
                    exception_name="SubprocessError",
                    exception_message=(
                        f"Process exited with code {process.returncode}: "
                        f"{stderr_text}"
                    ),
                )
                return SandboxRunResult(
                    func_return=func_return,
                    agent_state=None,
                    stdout=[stdout_text] if stdout_text else [],
                    stderr=[stderr_text] if stderr_text else [],
                    status="error",
                    sandbox_config_fingerprint=sbx_config.fingerprint(),
                )

            func_result, stdout_parsed = (
                self.parse_out_function_results_markers(stdout_text)
            )
            func_return, agent_state = self.parse_best_effort(func_result)
            return SandboxRunResult(
                func_return=func_return,
                agent_state=agent_state,
                stdout=[stdout_parsed] if stdout_parsed else [],
                stderr=[stderr_text] if stderr_text else [],
                status="success",
                sandbox_config_fingerprint=sbx_config.fingerprint(),
            )

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Executing tool {self.tool_name} has timed out."
            )

        except Exception as e:
            logger.error(
                f"Executing tool {self.tool_name} has an unexpected "
                f"error: {e}"
            )
            raise e

    def parse_out_function_results_markers(self, text: str):
        if self.LOCAL_SANDBOX_RESULT_START_MARKER not in text:
            return "", text
        marker_len = len(self.LOCAL_SANDBOX_RESULT_START_MARKER)
        start_index = text.index(self.LOCAL_SANDBOX_RESULT_START_MARKER) + marker_len
        end_index = text.index(self.LOCAL_SANDBOX_RESULT_END_MARKER)
        return (
            text[start_index:end_index],
            text[: start_index - marker_len] + text[end_index + +marker_len :],
        )

    async def create_venv_for_local_sandbox(self, sandbox_dir_path: str, venv_path: str, env: Dict[str, str]):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "venv", "--with-pip", venv_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"venv creation failed: {stderr.decode() if stderr else ''}"
            )

        pip_path = os.path.join(venv_path, "bin", "pip")
        try:
            logger.info("Upgrading pip in the virtual environment...")
            process = await asyncio.create_subprocess_exec(
                pip_path, "install", "--upgrade", "pip",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"pip upgrade failed: {stderr.decode() if stderr else ''}")

            requirements_txt_path = os.path.join(sandbox_dir_path, self.REQUIREMENT_TXT_NAME)
            if os.path.isfile(requirements_txt_path):
                logger.info(f"Installing packages from requirements file: {requirements_txt_path}")
                process = await asyncio.create_subprocess_exec(
                    pip_path, "install", "-r", requirements_txt_path,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(f"pip install failed: {stderr.decode() if stderr else ''}")
                logger.info("Successfully installed packages from requirements.txt")
            else:
                logger.warning(
                    "No requirements.txt file provided or the file does not exist. Skipping package installation."
                )

        except RuntimeError:
            raise
        except Exception as e:
            logger.error("Error while setting up the virtual environment: %s", e)
            raise RuntimeError(f"Failed to set up the virtual environment: {e}")

    # e2b sandbox specific functions

    async def run_e2b_sandbox(
        self,
        agent_state: Optional[AgentState] = None,
        additional_env_vars: Optional[Dict] = None,
    ) -> SandboxRunResult:
        sbx_config = self.sandbox_config_manager.get_or_create_default_sandbox_config(
            sandbox_type=SandboxType.E2B, actor=self.actor
        )
        sbx = await self.get_running_e2b_sandbox_with_same_state(sbx_config)
        if not sbx or self.force_recreate:
            if not sbx:
                logger.info(f"No running e2b sandbox found with the same state: {sbx_config}")
            else:
                logger.info("Force recreated e2b sandbox with state: %s", sbx_config)
            sbx = await self.create_e2b_sandbox_with_metadata_hash(sandbox_config=sbx_config)

        logger.info("E2B Sandbox configurations: %s", sbx_config)
        logger.info("E2B Sandbox ID: %s", sbx.sandbox_id)

        await sbx.set_timeout(sbx_config.get_e2b_config().timeout)

        env_vars = self.sandbox_config_manager.get_sandbox_env_vars_as_dict(
            sandbox_config_id=sbx_config.id, actor=self.actor, limit=100
        )
        if agent_state:
            env_vars.update(agent_state.get_agent_env_vars_as_dict())

        if additional_env_vars:
            env_vars.update(additional_env_vars)
        code = self.generate_execution_script(agent_state=agent_state)
        execution = await sbx.run_code(code, envs=env_vars)

        if execution.results:
            func_return, agent_state = self.parse_best_effort(execution.results[0].text)
        elif execution.error:
            logger.error(
                f"Executing tool {self.tool_name} raised a {execution.error.name} with message: \n{execution.error.value}"
            )
            logger.error("Traceback from e2b sandbox: \n%s", execution.error.traceback)
            func_return = get_friendly_error_msg(
                function_name=self.tool_name,
                exception_name=execution.error.name,
                exception_message=execution.error.value,
            )
            execution.logs.stderr.append(execution.error.traceback)
        else:
            raise ValueError(f"Tool {self.tool_name} returned execution with None")

        return SandboxRunResult(
            func_return=func_return,
            agent_state=agent_state,
            stdout=execution.logs.stdout,
            stderr=execution.logs.stderr,
            status="error" if execution.error else "success",
            sandbox_config_fingerprint=sbx_config.fingerprint(),
        )

    def parse_exception_from_e2b_execution(self, e2b_execution: "Execution") -> Exception:
        builtins_dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        exception_class = builtins_dict.get(e2b_execution.error.name, Exception)
        return exception_class(e2b_execution.error.value)

    async def get_running_e2b_sandbox_with_same_state(self, sandbox_config: SandboxConfig) -> Optional["AsyncSandbox"]:
        from e2b_code_interpreter import AsyncSandbox

        running_sandboxes = await self.list_running_e2b_sandboxes()

        state_hash = sandbox_config.fingerprint()
        for sandbox in running_sandboxes:
            if (
                self.METADATA_CONFIG_STATE_KEY in sandbox.metadata
                and sandbox.metadata[self.METADATA_CONFIG_STATE_KEY] == state_hash
            ):
                return await AsyncSandbox.connect(sandbox.sandbox_id)

        return None

    async def create_e2b_sandbox_with_metadata_hash(self, sandbox_config: SandboxConfig) -> "AsyncSandbox":
        from e2b_code_interpreter import AsyncSandbox

        state_hash = sandbox_config.fingerprint()
        e2b_config = sandbox_config.get_e2b_config()
        if e2b_config.template:
            sbx = await AsyncSandbox.create(
                sandbox_config.get_e2b_config().template,
                metadata={self.METADATA_CONFIG_STATE_KEY: state_hash},
            )
        else:
            sbx = await AsyncSandbox.create(
                metadata={self.METADATA_CONFIG_STATE_KEY: state_hash},
                **e2b_config.model_dump(exclude={"pip_requirements"}),
            )

        if e2b_config.pip_requirements:
            for package in e2b_config.pip_requirements:
                await sbx.commands.run(f"pip install {package}")
        return sbx

    async def list_running_e2b_sandboxes(self):
        from e2b_code_interpreter import AsyncSandbox

        return await AsyncSandbox.list()

    # general utility functions

    def parse_best_effort(self, text: str) -> Any:
        if not text:
            return None, None
        result = pickle.loads(base64.b64decode(text))
        agent_state = None
        if result["agent_state"] is not None:
            agent_state = result["agent_state"]
        return result["results"], agent_state

    def parse_function_arguments(self, source_code: str, tool_name: str):
        """Get arguments of a function from its source code"""
        tree = ast.parse(source_code)
        args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == tool_name:
                for arg in node.args.args:
                    args.append(arg.arg)
        return args

    def generate_execution_script(self, agent_state: AgentState, wrap_print_with_markers: bool = False) -> str:
        """
        Generate code to run inside of execution sandbox.
        Passes into a serialized agent state into the code, to be accessed by the tool.

        Args:
            agent_state (AgentState): The agent state
            wrap_print_with_markers (bool): If true, we wrap the final statement with a `print` and wrap with special markers

        Returns:
            code (str): The generated code strong
        """
        # dump JSON representation of agent state to re-load
        code = "from typing import *\n"
        code += "import pickle\n"
        code += "import sys\n"
        code += "import base64\n"

        # Load the agent state data into the program
        if agent_state:
            code += "import mirix\n"
            code += "from mirix import * \n"
            import pickle

            agent_state_pickle = pickle.dumps(agent_state)
            code += f"agent_state = pickle.loads({agent_state_pickle})\n"
        else:
            # agent state is None
            code += "agent_state = None\n"

        for param in self.args:
            code += self.initialize_param(param, self.args[param])

        if "agent_state" in self.parse_function_arguments(self.tool.source_code, self.tool.name):
            inject_agent_state = True
        else:
            inject_agent_state = False

        code += "\n" + self.tool.source_code + "\n"

        # TODO: handle wrapped print

        code += (
            self.LOCAL_SANDBOX_RESULT_VAR_NAME
            + ' = {"results": '
            + self.invoke_function_call(inject_agent_state=inject_agent_state)
            + ', "agent_state": agent_state}\n'
        )
        code += f"{self.LOCAL_SANDBOX_RESULT_VAR_NAME} = base64.b64encode(pickle.dumps({self.LOCAL_SANDBOX_RESULT_VAR_NAME})).decode('utf-8')\n"

        if wrap_print_with_markers:
            code += f"sys.stdout.write('{self.LOCAL_SANDBOX_RESULT_START_MARKER}')\n"
            code += f"sys.stdout.write(str({self.LOCAL_SANDBOX_RESULT_VAR_NAME}))\n"
            code += f"sys.stdout.write('{self.LOCAL_SANDBOX_RESULT_END_MARKER}')\n"
        else:
            code += f"{self.LOCAL_SANDBOX_RESULT_VAR_NAME}\n"

        return code

    def _convert_param_to_value(self, param_type: str, raw_value: str) -> str:
        if param_type == "string":
            value = "pickle.loads(" + str(pickle.dumps(raw_value)) + ")"

        elif param_type == "integer" or param_type == "boolean" or param_type == "number":
            value = raw_value

        elif param_type == "array":
            value = raw_value

        elif param_type == "object":
            value = raw_value

        else:
            raise TypeError(f"Unsupported type: {param_type}, raw_value={raw_value}")
        return str(value)

    def initialize_param(self, name: str, raw_value: str) -> str:
        params = self.tool.json_schema["parameters"]["properties"]
        spec = params.get(name)
        if spec is None:
            # ignore extra params (like 'self') for now
            return ""

        param_type = spec.get("type")
        if param_type is None and spec.get("parameters"):
            param_type = spec["parameters"].get("type")

        value = self._convert_param_to_value(param_type, raw_value)

        return name + " = " + value + "\n"

    def invoke_function_call(self, inject_agent_state: bool) -> str:
        """
        Generate the code string to call the function.

        Args:
            inject_agent_state (bool): Whether to inject the axgent's state as an input into the tool

        Returns:
            str: Generated code string for calling the tool
        """
        kwargs = []
        for name in self.args:
            if name in self.tool.json_schema["parameters"]["properties"]:
                kwargs.append(name)

        param_list = [f"{arg}={arg}" for arg in kwargs]
        if inject_agent_state:
            param_list.append("agent_state=agent_state")
        params = ", ".join(param_list)
        # if "agent_state" in kwargs:
        #    params += ", agent_state=agent_state"
        # TODO: fix to figure out when to insert agent state or not
        # params += "agent_state=agent_state"

        func_call_str = self.tool.name + "(" + params + ")"
        return func_call_str
