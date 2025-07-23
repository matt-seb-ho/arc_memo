from typing import Literal

import numpy as np
import pandas as pd

from concept_mem.constants import NO_CODE_BLOCK_MESSAGE
from concept_mem.data.arc_agi import IOPair, Problem
from concept_mem.evaluation.solution_tree import (
    IOPairExecutionResult,
    SolutionStep,
    SolutionTree,
)
from concept_mem.utils import extract_python_block
from concept_mem.utils.code_execution.exec_transform import execute_transforms

# -----------------------------------------------------------------------------
# evaluating individual puzzles
# -----------------------------------------------------------------------------


def evaluate_solution_on_io_pairs(
    code: str,
    io_pairs: list[IOPair],
    is_train: bool = True,
) -> list[IOPairExecutionResult]:
    """returns a tuple containing code extraction status and execution results."""
    # prepare code solution and inputs
    input_grids = [pair.x for pair in io_pairs]
    # execute transforms and collect results
    transform_results = execute_transforms(
        transform_functions=code,
        input_grids=input_grids,
        timeout=2.0,
        function_name="transform",
        max_workers=1,
    )
    output_list = []
    for i, tr in enumerate(transform_results):
        # try to coerce output to numpy array
        if isinstance(tr.output, list):
            tr.output = np.array(tr.output)
        # handle output errors
        # - if status is not ok or incorrectly typed,
        #   try to save output as a string for code refinement feedback
        # - mark this pair's outcome as incorrect
        if tr.status != "ok" or not isinstance(tr.output, np.ndarray):
            # try to conserve output as a string for code refinement feedback
            try:
                output_grid = str(tr.output)
            finally:
                output_grid = None
            binary_score = False
        else:
            output_grid = tr.output
            binary_score = np.array_equal(tr.output, io_pairs[i].y)
        # add to output list
        output_list.append(
            IOPairExecutionResult(
                is_train=is_train,
                pair_idx=i,
                output=output_grid,
                correct=binary_score,
                error=tr.error,
                stdout=tr.stdout,
            )
        )
    return output_list


def parse_code_solution(
    completion: str | None,
) -> tuple[str | None, str | None]:
    if completion is None:
        return None, "null completion."
    if completion == "":
        return None, "empty completion."
    code = extract_python_block(completion)
    if code is None:
        return None, NO_CODE_BLOCK_MESSAGE
    return code, None


def score_problem_attempt(
    problem: Problem,
    attempt: SolutionStep,
) -> None:
    code, parsing_error = parse_code_solution(attempt.completion)
    if parsing_error:
        attempt.parsing_error = parsing_error
        attempt.validated = True
        return
    attempt.train_results = evaluate_solution_on_io_pairs(
        code=code,
        io_pairs=problem.train_pairs,
        is_train=True,
    )
    attempt.test_results = evaluate_solution_on_io_pairs(
        code=code,
        io_pairs=problem.test_pairs,
        is_train=False,
    )
    attempt.validated = True


# -----------------------------------------------------------------------------
# scoring and aggregation for puzzle set
# -----------------------------------------------------------------------------

"""Utility helpers for flattening `SolutionTree` objects into a pandas `DataFrame`
- one (puzzle, branch, thread, step, IO-pair) per row
- mix-and-match filters (branch selections, thread budgets, step budgets)
- apply different per-puzzle scoring rules with a couple of `groupby` calls.
- all filters are **composable** because they are just boolean masks on the df
"""

STEP_TAGS = [
    "puzzle_id",
    "branch_id",
    "thread_id",
    "step_idx",
]

# ----------------------------------------------------------------------------
# flattening helpers


def _iter_rows(
    solution_trees: dict[str, SolutionTree],
) -> tuple[list[dict], list[dict]]:
    """Yield per-case and per-step row dicts for normalized table output."""
    case_rows = []
    step_rows = []
    for puzzle_id, tree in solution_trees.items():
        for branch_id, branch in tree.prompt_branches.items():
            for thread_id, thread in branch.threads.items():
                for step in thread.steps:
                    step_key = {
                        "puzzle_id": puzzle_id,
                        "branch_id": branch_id,
                        "thread_id": thread_id,
                        "step_idx": step.step_idx,
                    }
                    step_rows.append(
                        {
                            **step_key,
                            "validated": step.validated,
                            "parsing_error": step.parsing_error,
                            "completion": step.completion,
                        }
                    )
                    for res in step.train_results + step.test_results:
                        case_rows.append(
                            {
                                **step_key,
                                "is_train": res.is_train,
                                "case_idx": res.pair_idx,
                                "correct": res.correct,
                            }
                        )
    return case_rows, step_rows


def flatten_solution_trees(
    solution_trees: dict[str, SolutionTree],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert nested `SolutionTree` mapping into normalized step_df and case_df."""
    case_rows, step_rows = _iter_rows(solution_trees)
    if not case_rows:
        return pd.DataFrame(), pd.DataFrame()
    case_df = pd.DataFrame(case_rows)
    case_df["correct"] = case_df["correct"].astype(bool)
    case_df["is_train"] = case_df["is_train"].astype(bool)
    step_df = pd.DataFrame(step_rows)
    return case_df, step_df


# ----------------------------------------------------------------------------
# scoring logic


def _keep_last_step(df: pd.DataFrame) -> pd.DataFrame:
    grouping_cols = ["puzzle_id", "branch_id", "thread_id"]
    max_step = df.groupby(grouping_cols, sort=False)["step_idx"].transform("max")
    return df[df["step_idx"] == max_step]


def _official_per_pair(group: pd.DataFrame, attempts_allowed: int | None) -> bool:
    if attempts_allowed is not None:
        head = group.sort_values("step_idx").head(attempts_allowed)
        return head["correct"].any()
    return group["correct"].any()


def official_score_per_puzzle(
    df: pd.DataFrame,
    attempts_allowed: int | None = None,
    step_selection: Literal["all", "last"] = "all",
) -> pd.DataFrame:
    if df.empty:
        return pd.Series(dtype=float)
    if step_selection == "last":
        df = _keep_last_step(df)

    test_df = df[~df["is_train"]]
    solved_per_pair = (
        test_df.groupby(["puzzle_id", "case_idx"], sort=False)
        .apply(lambda g: _official_per_pair(g, attempts_allowed), include_groups=False)
        .reset_index(name="solved")
    )

    per_puzzle = solved_per_pair.groupby("puzzle_id", sort=False)["solved"].mean()
    return per_puzzle


def official_score(
    df: pd.DataFrame,
    attempts_allowed: int | None = None,
    step_selection: Literal["all", "last"] = "all",
) -> float:
    official_score_per_puzzle_series = official_score_per_puzzle(
        df,
        attempts_allowed=attempts_allowed,
        step_selection=step_selection,
    )
    if official_score_per_puzzle_series.empty:
        return 0.0
    return official_score_per_puzzle_series.sum()


def strict_score_per_step(
    df: pd.DataFrame,
    *,
    include_train: bool = False,
    step_selection: Literal["all", "last"] = "last",
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)

    if not include_train:
        df = df[~df["is_train"]]
    if step_selection == "last":
        df = _keep_last_step(df)

    step_correct = df.groupby(STEP_TAGS, sort=False)["correct"].all()
    return step_correct


def strict_score_per_puzzle(
    df: pd.DataFrame,
    *,
    include_train: bool = False,
    step_selection: Literal["all", "last"] = "all",
) -> pd.Series:
    step_correct = strict_score_per_step(
        df,
        include_train=include_train,
        step_selection=step_selection,
    )
    if step_correct.empty:
        return pd.Series(dtype=int)
    per_puzzle = step_correct.groupby("puzzle_id", sort=False).any()
    return per_puzzle


def strict_score(
    df: pd.DataFrame,
    *,
    include_train: bool = False,
    step_selection: Literal["all", "last"] = "all",
) -> int:
    strict_score_per_puzzle_series = strict_score_per_puzzle(
        df,
        include_train=include_train,
        step_selection=step_selection,
    )
    if strict_score_per_puzzle_series.empty:
        return 0
    return strict_score_per_puzzle_series.sum()
