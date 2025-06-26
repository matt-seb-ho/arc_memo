import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from concept_mem.constants import NO_CODE_BLOCK_MESSAGE
from concept_mem.types import Problem
from concept_mem.utils import write_json
from concept_mem.utils.code_execution.exec_transform import execute_transforms

logger = logging.getLogger(__name__)


def extract_python_code_block(text: str) -> Optional[str]:
    """
    Extract the first Python code block (```python ... ```).
    If none, fall back to the first unlabeled code block (``` ... ```).
    Returns the code inside the fences, or None if no code block is found.
    """
    # Combined pattern: optional “python” label after the opening backticks
    pattern = re.compile(r"```(?:python)?\s*?\n([\s\S]*?)\n```", re.IGNORECASE)

    match = pattern.search(text)
    if match:
        return match.group(1)

    logger.debug("No markdown code block found in the response.")
    return None


def swap_main_with_transform(code: str) -> str:
    main_str = "def main("
    transform_str = "def transform("
    if transform_str not in code:
        return code.replace(main_str, transform_str)
    return code


def skip_top_level_prints(code: str) -> str:
    # filter out code lines with 0 indentation print statements
    code_lines = code.split("\n")
    code_lines = [line for line in code_lines if not line.startswith("print(")]
    return "\n".join(code_lines)


def post_process_code(code: str) -> str:
    code = swap_main_with_transform(code)
    code = skip_top_level_prints(code)
    return code


def parse_code_solution(
    completion: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if completion is None:
        return None, "null completion."
    if completion == "":
        return None, "empty completion."
    code = extract_python_code_block(completion)
    if code is None:
        return None, NO_CODE_BLOCK_MESSAGE
    post_processed_code = post_process_code(code)
    return post_processed_code, None


def score_transform_function(
    problem: Problem, code: str, split: str = "test", timeout: float = 2.0
) -> tuple[float, list[bool], list[np.ndarray | str | None]]:
    correct = []
    return_output_grids = []
    problems_to_evaluate = (
        problem.train_pairs if split == "train" else problem.test_pairs
    )
    input_grids = [pair.x for pair in problems_to_evaluate]
    exec_results = execute_transforms(
        transform_functions=[code] * len(input_grids),
        input_grids=input_grids,
        timeout=timeout,
        function_name="transform",
        max_workers=8,
    )
    for i, exec_result in enumerate(exec_results):
        if exec_result.status == "ok":
            output_grid = exec_result.output
            expected_output_grid = problems_to_evaluate[i].y
            correct.append(np.array_equal(output_grid, expected_output_grid))
            return_output_grids.append(exec_result.output)
        else:
            logger.debug(f"\t[-] function error: {exec_result.error}")
            correct.append(False)
            return_output_grids.append(exec_result.error)
    total_score = sum(correct) / len(correct) if len(correct) > 0 else 0
    return total_score, correct, return_output_grids


def score_completions(
    problems: dict[str, Problem],
    completions: dict[str, list[str | dict]],
    output_dir: Path,
    attempt: int = 1,
) -> dict:
    results = {}
    validation_errors = []
    total_test_score = 0.0
    fully_correct_count = 0
    problems_with_extracted_solution = 0
    for problem_id, prompt_completions in tqdm(
        completions.items(),
        total=len(completions),
        desc="scoring completions",
    ):
        problem = problems[problem_id]
        sample_results = []
        per_test_case = [False] * len(problem.test_pairs)
        has_extracted_solution = False  # meaning a markdown python code block was found
        has_fully_correct_solution = False  # meaning all test AND train examples pass
        if len(prompt_completions) == 0:
            logger.debug(f"No completions for problem {problem_id}, skipping")
            validation_errors.append(
                {
                    "uid": problem.uid,
                    "error": "empty completion list",
                    "completion_idx": None,
                }
            )
        for i, completion in enumerate(prompt_completions):
            if isinstance(completion, dict):
                sample_idx = completion["idx"]
                attempt = completion["attempt"]
                completion = completion["completion"]
            else:
                sample_idx = i
                attempt = 1
            parsed_code, parsing_message = parse_code_solution(completion)
            if parsing_message:
                validation_errors.append(
                    {
                        "uid": problem.uid,
                        "error": parsing_message,
                        "completion_idx": i,
                    }
                )
                sample_results.append(
                    {
                        "completion": completion,
                        "sample_idx": sample_idx,
                        "attempt": attempt,
                        "test_score": 0.0,
                        "train_scores": None,
                        "test_scores": None,
                        "train_output_grids": None,
                        "test_output_grids": None,
                    }
                )
                continue
            # run validation on reference examples
            ref_score, ref_correct, ref_output_grids = score_transform_function(
                problem=problem,
                code=parsed_code,
                split="train",
            )
            # run validation on test case(s)
            test_score, test_correct, test_output_grids = score_transform_function(
                problem=problem,
                code=parsed_code,
                split="test",
            )
            # update problem-wise results
            has_extracted_solution = True
            has_fully_correct_solution = has_fully_correct_solution or (
                ref_score == 1.0 and test_score == 1.0
            )
            for j, test_case_correct in enumerate(test_correct):
                per_test_case[j] = per_test_case[j] or test_case_correct
            # reformat output grids for JSON serialization
            sample_results.append(
                {
                    "completion": completion,
                    "sample_idx": sample_idx,
                    "attempt": attempt,
                    "test_score": test_score,
                    "train_scores": ref_correct,
                    "test_scores": test_correct,
                    "train_output_grids": _convert_grids_to_list(ref_output_grids),
                    "test_output_grids": _convert_grids_to_list(test_output_grids),
                }
            )
        problem_ensemble_score = (
            (sum(per_test_case) / len(per_test_case)) if per_test_case else 0
        )
        results[problem_id] = {
            "sample_results": sample_results,
            "ensemble_test_score": problem_ensemble_score,
            "has_fully_correct_solution": has_fully_correct_solution,
            "has_extracted_solution": has_extracted_solution,
        }
        total_test_score += problem_ensemble_score
        if has_fully_correct_solution:
            fully_correct_count += 1
        if has_extracted_solution:
            problems_with_extracted_solution += 1

    # write results to files
    output_file = output_dir / f"eval_output_att{attempt}.json"
    write_json(results, output_file, indent=True)

    # report and write validation errors to file
    validation_error_file = output_dir / f"validation_errors_att{attempt}.json"
    write_json(validation_errors, validation_error_file, indent=True)
    validation_error_counts = Counter([e["error"] for e in validation_errors])
    logger.info(
        f"Validation error counts: {validation_error_counts}, total: {len(validation_errors)}"
    )

    # report and write summary stats to file
    if len(problems) == 0:
        adjusted_pass_rate = 0
        adjusted_avg_test_score = 0
    else:
        adjusted_pass_rate = fully_correct_count / problems_with_extracted_solution
        adjusted_avg_test_score = total_test_score / problems_with_extracted_solution
    summary_stats = {
        "attempt": attempt,
        "total_test_score": total_test_score,
        "total_problems": len(completions),
        "fully_correct": fully_correct_count,
        "problems_with_extracted_solution": problems_with_extracted_solution,
        "full_pass_rate": fully_correct_count / len(problems),
        "avg_test_score": total_test_score / len(problems),
        "adjusted_pass_rate": adjusted_pass_rate,
        "adjusted_avg_test_score": adjusted_avg_test_score,
    }
    summary_file = Path(output_dir) / f"summary_stats_att{attempt}.json"
    write_json(summary_stats, summary_file, indent=True)
    logger.info(f"Summary Stats: {json.dumps(summary_stats, indent=2)}")
    logger.info(f"Wrote [res, val_errors, stats] to {output_dir}")
    return results


def _convert_grids_to_list(grids: list[np.ndarray | None]) -> list[list | None]:
    """
    Convert a list of numpy arrays to a list of lists for JSON serialization.
    """
    return [grid.tolist() if isinstance(grid, np.ndarray) else grid for grid in grids]
