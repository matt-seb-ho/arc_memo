from typing import Literal

import numpy as np
import pandas as pd

from concept_mem.evaluation.solution_tree import (
    IOPairExecutionResult,
    SolutionStep,
    SolutionTree,
)
from concept_mem.types import IOPair, Problem
from concept_mem.utils import extract_code_block_contents
from concept_mem.utils.code_execution.exec_transform import execute_transforms

# -----------------------------------------------------------------------------
# evaluating individual puzzles
# -----------------------------------------------------------------------------


def evaluate_solution_on_io_pairs(
    code: str,
    io_pairs: list[IOPair],
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
        max_workers=8,
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
                output_grid=output_grid,
                correct=binary_score,
                error=tr.error,
                stdout=tr.stdout,
            )
        )
    return output_list


def score_problem_attempt(
    problem: Problem,
    attempt: SolutionStep,
) -> None:
    code = extract_code_block_contents(attempt.completion)
    if code is None:
        attempt.code_extracted = False
        return
    attempt.code_extracted = True
    attempt.train_results = evaluate_solution_on_io_pairs(
        code=code,
        io_pairs=problem.train_pairs,
    )
    attempt.test_results = evaluate_solution_on_io_pairs(
        code=code,
        io_pairs=problem.test_pairs,
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


# -----------------------------------------------------------------------------
# flattening helpers


def _iter_rows(solution_trees: dict[str, SolutionTree]):
    """Yield dict‑rows for every IO‑pair in every step."""

    for puzzle_id, tree in solution_trees.items():
        for branch_id, branch in tree.prompt_branches.items():
            for thread_id, thread in branch.threads.items():
                for step in thread.steps:
                    for res in step.train_results + step.test_results:
                        yield {
                            "puzzle_id": puzzle_id,
                            "branch_id": branch_id,
                            "prompt_id": step.prompt_id,
                            "thread_id": thread_id,
                            "step_idx": step.step_idx,
                            "is_train": res.is_train,
                            "pair_idx": res.pair_idx,
                            "correct": res.correct,
                            "validated": step.validated,
                            "code_extracted": step.code_extracted,
                        }


def flatten_solution_trees(solution_trees: dict[str, SolutionTree]) -> pd.DataFrame:
    """Convert nested `SolutionTree` mapping to a DataFrame."""

    rows = list(_iter_rows(solution_trees))
    if not rows:
        return pd.DataFrame()  # empty input guard

    df = pd.DataFrame(rows)
    # Enforce dtypes that play nicely with boolean math later
    df["correct"] = df["correct"].astype(bool)
    df["is_train"] = df["is_train"].astype(bool)
    return df


# -----------------------------------------------------------------------------
# scoring back‑ends


def _keep_last_step(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose `step_idx` equals the *maximum* for their thread."""
    last_idx = df.groupby(["puzzle_id", "branch_id", "thread_id"], sort=False)[
        "step_idx"
    ].transform("max")
    return df[df["step_idx"] == last_idx]


def _official_per_pair(group: pd.DataFrame, attempts_allowed: int | None) -> bool:
    """Return *True* if at least one of the first *k* attempts is correct."""

    # preserve chronological ordering (step_idx) to approximate "first k model
    # interactions" semantics; you can swap this to another secondary key (e.g.
    # log‑prob) if you have it.
    if attempts_allowed is not None:
        head = group.sort_values("step_idx").head(attempts_allowed)
    return head["correct"].any()


def official_score(
    df: pd.DataFrame,
    attempts_allowed: int = 2,
    step_selection: Literal["all", "last"] = "all",
) -> float:
    """Compute the *official* score

    \sum_puzzles \frac{1}{N_p} \sum_{i=1}^{N_p} s_i

    where N_p is the number of IO‑pairs in puzzle p, and s_i is the score for the ith puzzle
    (mean fraction of correctly solved test IO‑pairs per puzzle, then summed across puzzles).
    """

    if df.empty:
        return 0.0
    if step_selection == "last":
        df = _keep_last_step(df)

    test_df = df[~df["is_train"]]
    solved_per_pair = (
        test_df.groupby(["puzzle_id", "pair_idx"], sort=False)
        .apply(lambda g: _official_per_pair(g, attempts_allowed))
        .reset_index(name="solved")
    )

    # fraction of solved pairs per puzzle
    per_puzzle = solved_per_pair.groupby("puzzle_id", sort=False)["solved"].mean()
    return per_puzzle.sum()


def strict_score(
    df: pd.DataFrame,
    *,
    include_train: bool = False,
    step_selection: Literal["all", "last"] = "all",  # re‑use for consistency
) -> int:
    """Return the number of puzzles that are *strict‑correct* under the rules.

    - a code solution is strictly correct if it passes all test IO-pairs
    (and optionally all train IO-pairs).
    """

    if df.empty:
        return 0

    if not include_train:
        df = df[~df["is_train"]]
    if step_selection == "last":
        df = _keep_last_step(df)

    # step‑level strict correctness
    step_tags = [
        "puzzle_id",
        "branch_id",
        "thread_id",
        "step_idx",
    ]
    step_correct = df.groupby(step_tags, sort=False)["correct"].all()
    puzzle_strict_correct = step_correct.groupby("puzzle_id", sort=False).all()
    return int(puzzle_strict_correct.sum())
