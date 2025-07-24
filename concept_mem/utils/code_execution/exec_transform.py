from dataclasses import dataclass
from typing import Optional

import numpy as np

from concept_mem.utils.code_execution.exec import ExecutionResult, multi_execute

EXECUTE_TRANSFORM_TEMPLATE = """\
import numpy as np

{input_grid_definition}

{transform_function_definition}

output_grid = {function_name}(input_grid)
"""


@dataclass
class TransformFunctionResult(ExecutionResult):
    """
    A slightly specialized result for sandboxed solution code execution.
    - retains fields "status", "error", and "output"
    - except "output" is always a numpy array if not None
    """

    output: Optional[np.ndarray]


def execute_transforms(
    transform_functions: str | list[str],
    input_grids: list[np.ndarray],
    timeout: float = 2.0,
    function_name: str = "main",
    max_workers: int = 8,
    library: str = None,
) -> list[TransformFunctionResult]:
    """
    Execute a list of transform functions in parallel and return the results.
    Each function is expected to take an input grid and return a transformed grid.
    Optionally, prepend a library of helper routines to each function.
    """
    # prepare function calls
    # - if `transform_functions` is a string,
    #   assume we want to evaluate the same function for all inputs
    if isinstance(transform_functions, str):
        transform_functions = [transform_functions] * len(input_grids)
    sources = []
    for src, input_grid in zip(transform_functions, input_grids):
        code = EXECUTE_TRANSFORM_TEMPLATE.format(
            input_grid_definition=build_input_grid_code(input_grid),
            transform_function_definition=src,
            function_name=function_name,
        )
        if library:
            code = library + "\n\n" + code
        sources.append(code)
    # execute functions
    exec_results = multi_execute(
        code_list=sources,
        return_var_name="output_grid",
        timeout=timeout,
        max_workers=max_workers,
        tqdm_kwargs={"leave": False},
    )
    # post-process: (check output type, shape, value range)
    transform_results = []
    for r in exec_results:
        if r.status == "ok":
            if not isinstance(r.output, np.ndarray):
                tfr = TransformFunctionResult(
                    status="error",
                    output=r.output,
                    error=f"return type error: function returned {type(r.output)}",
                )
            elif len(r.output.shape) != 2:
                tfr = TransformFunctionResult(
                    status="error",
                    output=r.output,
                    error=f"return shape error: function returned {r.output.shape}",
                )
            elif not np.all((0 <= r.output) & (r.output <= 9)):
                tfr = TransformFunctionResult(
                    status="error",
                    output=r.output,
                    error="return value range error: function returned values outside [0, 9]",
                )
            else:
                # all checks passed
                tfr = TransformFunctionResult(status="ok", output=r.output, error=None)
            transform_results.append(tfr)
        else:
            # propagate error
            tfr = TransformFunctionResult(status=r.status, output=None, error=r.error)
            transform_results.append(tfr)
    return transform_results


def load_starter_library(path: str = None) -> str:
    """
    Load the full source code of starter.py as a string for use as a library.
    If path is None, use the default location in the repo.
    """
    import pathlib
    if path is None:
        path = pathlib.Path(__file__).parent / "starter.py"
    else:
        path = pathlib.Path(path)
    return path.read_text()


def build_input_grid_code(arr: np.ndarray) -> str:
    """
    Given any 2D array-like, cast to int and return a string
    containing executable Python code that reconstructs it as:
      input_grid = np.array([...], dtype=int)
    """
    a = np.array(arr, dtype=int)
    inner = np.array2string(a, separator=", ")
    code = f"input_grid = np.array({inner}, dtype=int)"
    return code
