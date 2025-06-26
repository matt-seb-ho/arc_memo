import builtins
import io
import logging
import os
import signal
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Any, Literal

import psutil
from pebble import ProcessExpired, ProcessPool
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Disallow imports that can escape the sandbox
DISALLOWED_IMPORTS = ("os", "sys", "subprocess", "multiprocessing", "pathlib")


@dataclass
class ExecutionResult:
    """
    A generic result for sandboxed code execution.

    Attributes:
        status: "ok", "timeout", or "error".
        output: The value of `return_var_name` from the executed code, or None.
        error: An error message if status is "error", else None.
    """

    status: Literal["ok", "timeout", "error"]
    output: Any = None
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


def multi_execute(
    code_list: list[str],
    return_var_name: str,
    timeout: float = 1,
    max_workers: int = 8,
    tqdm_kwargs: dict | None = None,
) -> list[ExecutionResult | None]:
    """
    Execute each code snippet in its own process via pebble, returning a list of ExecutionResult.

    code_list: list of source-code strings.
    return_var_name: variable name to extract from each execution.
    timeout: per-task execution timeout in seconds.
    max_workers: max concurrent processes.
    """
    results = [None] * len(code_list)
    tqdm_kwargs = tqdm_kwargs or {}
    # submit all tasks to the pool
    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(
            _worker, code_list, [return_var_name] * len(code_list), timeout=timeout
        )
        iterator = future.result()

        # iterate in submission order
        for i in tqdm(range(len(code_list)), **tqdm_kwargs):
            try:
                results[i] = next(iterator)
            except ProcessExpired as error:
                logger.debug(
                    f"Task {i} expired after {timeout}s, exit code {error.exitcode}"
                )
                results[i] = ExecutionResult(status="timeout", error="timeout")
            except Exception as error:
                logger.debug(f"Task {i} failed: {error}")
                results[i] = ExecutionResult(status="error", error=str(error))
    # ensure no stray processes remain
    terminate_all_processes()

    return results


def _safe_import(name: str, _globals=None, _locals=None, fromlist=(), level=0):
    """
    Restrict imports in the sandbox to prevent unsafe operations.
    """
    if name in DISALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' not allowed")
    return __import__(name, _globals, _locals, fromlist, level)


def _worker(
    source: str,
    return_var_name: str = "output_grid",
) -> ExecutionResult:
    """
    Execute source code in a restricted global context and return the named variable.
    Raises RuntimeError if the variable is missing.
    """
    # Copy the builtins namespace to a mutable dict
    safe_builtins = builtins.__dict__.copy()
    # Remove dangerous builtins
    for name in ("exit", "quit"):
        safe_builtins.pop(name, None)
    # Override import to restrict modules
    safe_builtins["__import__"] = _safe_import

    buf_out = io.StringIO()
    buf_err = io.StringIO()

    ctx = {"__builtins__": safe_builtins}
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(source, ctx)
        exec(source, ctx)
        if return_var_name not in ctx:
            raise RuntimeError(f"'{return_var_name}' not found in execution context")
        return ExecutionResult(
            status="ok",
            output=ctx[return_var_name],
            error=None,
            stdout=buf_out.getvalue(),
            stderr=buf_err.getvalue(),
        )
    except Exception as e:
        return ExecutionResult(
            status="error",
            output=None,
            error=str(e),
            stdout=buf_out.getvalue(),
            stderr=buf_err.getvalue(),
        )


def kill_process(pid: int, grace_period: float = 1.0) -> None:
    """Try SIGTERM, then SIGKILL after a grace period."""
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(grace_period)
        if psutil.pid_exists(pid):
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.debug(f"kill_process error: {e}")


def terminate_all_processes() -> None:
    """Kill any child processes left behind by executor workers."""
    parent = psutil.Process()
    for child in parent.children(recursive=True):
        kill_process(child.pid)


if __name__ == "__main__":
    # Example usage
    sample_code = [
        "x = 1\ny = 2\nresult = x + y",
    ]
    results = multi_execute(sample_code, "result", timeout=2)
    for r in results:
        print(r)
