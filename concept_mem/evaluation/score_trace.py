import json
import logging
from dataclasses import asdict
from pathlib import Path

from concept_mem.evaluation.solution_trace_types import (
    PromptVariant,
    PuzzleTrace,
)
from concept_mem.utils import write_json

logger = logging.getLogger(__name__)


# target dict
# {
#     "puzzle_id": {
#         "ensembled_scores": list[bool] | None,
#         "has_fully_correct_solution": bool,
#         "prompt_variants": {
#             "prompt_id": {
#                 "ensembled_scores": list[bool] | None,
#                 "has_fully_correct_solution": bool,
#                 "paths": {
#                     "path_id": [last RefinementPass info]
#                 }
#             }
#         }
#     }
# }


def score_variant(variant: PromptVariant) -> dict:
    path_entries = {}
    ensembled_scores = None
    has_fully_correct_solution = False
    for path_id, path in variant.paths.items():
        last_pass = path.last()
        path_entries[str(path_id)] = last_pass.get_json_dict()
        if len(last_pass.test_scores) == 0:
            continue
        elif ensembled_scores is None:
            ensembled_scores = last_pass.test_scores.copy()
        else:
            ensembled_scores = [
                a or b for a, b in zip(ensembled_scores, last_pass.test_scores)
            ]
        if all(last_pass.train_scores) and all(last_pass.test_scores):
            has_fully_correct_solution = True
    return {
        "ensembled_scores": ensembled_scores,
        "paths": path_entries,
        "has_fully_correct_solution": has_fully_correct_solution,
    }


def score_trace(trace: PuzzleTrace) -> dict:
    pv_entries = {}
    ensembled_scores = None
    has_fully_correct_solution = False
    for prompt_id, variant in trace.prompt_variants.items():
        scores = score_variant(variant)
        pv_entries[prompt_id] = scores
        if scores["ensembled_scores"] is None:
            continue
        elif ensembled_scores is None:
            ensembled_scores = scores["ensembled_scores"]
        else:
            ensembled_scores = [
                a or b for a, b in zip(ensembled_scores, scores["ensembled_scores"])
            ]
        if scores["has_fully_correct_solution"]:
            has_fully_correct_solution = True
    return {
        "ensembled_scores": ensembled_scores,
        "prompt_variants": pv_entries,
        "has_fully_correct_solution": has_fully_correct_solution,
    }


def score_traces(
    traces: dict[str, PuzzleTrace],
    pass_num: int,
    pass_dir: Path,
) -> None:
    score_artifact = {}
    total_test_score = 0
    fully_correct = 0
    extracted = 0

    for puzzle_id, trace in traces.items():
        trace_score = score_trace(trace)
        score_artifact[puzzle_id] = trace_score
        if trace_score["ensembled_scores"]:
            total_test_score += sum(trace_score["ensembled_scores"]) / len(
                trace_score["ensembled_scores"]
            )
            extracted += 1
        if trace_score["has_fully_correct_solution"]:
            fully_correct += 1

    # ---------------------------------------------------------------
    n_probs = len(traces) or 1
    summary = {
        "pass": pass_num,
        "total_test_score": total_test_score,
        "total_problems": n_probs,
        "fully_correct": fully_correct,
        "problems_with_extracted_solution": extracted,
        "full_pass_rate": fully_correct / n_probs,
        "avg_test_score": total_test_score / n_probs,
        "adjusted_pass_rate": fully_correct / extracted if extracted else 0,
        "adjusted_avg_test_score": total_test_score / extracted if extracted else 0,
    }

    # write artefacts -------------------------------------------------
    write_json(score_artifact, pass_dir / "results.json")
    write_json(summary, pass_dir / "summary_stats.json", indent=True)
    logger.info("Score Summary: %s", json.dumps(summary, indent=2))
