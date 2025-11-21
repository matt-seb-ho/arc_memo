#!/usr/bin/env python3
"""
Compute pass@1 and pass@2 metrics for all validation runs in `aime_val/`.

The script expects each run directory to contain:
- `solutions.json`: JSON mapping problem_id -> extracted answer string
- `full_responses.json`: JSON mapping problem_id -> list of raw model responses

It reads the ground-truth answers from `data/aime/validation.json`, compares
each run's outputs, and writes a summary JSON file named `pass_metrics.json`
in the current directory (`experiments_label_guided/`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
AIME_VAL_DIR = BASE_DIR / "aime_val"
GT_PATH = BASE_DIR.parent / "data" / "aime" / "validation.json"
OUTPUT_PATH = BASE_DIR / "pass_metrics.json"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunMetrics:
    total: int
    pass1: int
    pass2: int

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "pass@1": {
                "correct": self.pass1,
                "rate": round(self.pass1 / self.total, 4),
            },
            "pass@2": {
                "correct": self.pass2,
                "rate": round(self.pass2 / self.total, 4),
            },
        }


def load_ground_truth(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    problems = data["problems"] if isinstance(data, dict) else data
    gt: Dict[str, str] = {}
    for entry in problems:
        pid = str(entry.get("id", "")).strip()
        ans = str(entry.get("answer", "")).strip()
        if pid:
            gt[pid] = ans
    return gt


_BOXED_RE = re.compile(r"\\boxed\{(\d{1,3})\}")
_FINAL_ANSWER_RE = re.compile(r"[Ff]inal\s+[Aa]nswer\s*:\s*(\d+)")
_ANSWER_RE = re.compile(r"[Aa]nswer\s*:\s*(\d+)")
_THE_ANSWER_IS_RE = re.compile(r"[Tt]he\s+answer\s+is\s+(\d+)")
_NUMBER_RE = re.compile(r"\b(\d{1,3})\b")


def extract_answer(response: str) -> Optional[str]:
    if not response:
        return None
    text = response.strip()

    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1]

    for pattern in (_FINAL_ANSWER_RE, _ANSWER_RE, _THE_ANSWER_IS_RE):
        match = pattern.search(text)
        if match:
            return match.group(1)

    tail = text[-500:]
    numbers = _NUMBER_RE.findall(tail)
    for num in reversed(numbers):
        value = int(num)
        if 0 <= value <= 999:
            return num
    return None


def compute_metrics_for_run(run_dir: Path, gt: Dict[str, str]) -> Optional[RunMetrics]:
    solutions_path = run_dir / "solutions.json"
    responses_path = run_dir / "full_responses.json"
    if not solutions_path.exists() or not responses_path.exists():
        return None

    with solutions_path.open("r", encoding="utf-8") as f:
        solutions = json.load(f)
    with responses_path.open("r", encoding="utf-8") as f:
        full_responses = json.load(f)

    total = len(gt)
    pass1 = sum(
        1 for pid, answer in gt.items() if solutions.get(pid, "").strip() == answer
    )

    pass2 = 0
    for pid, answer in gt.items():
        responses: Iterable[str] = full_responses.get(pid) or []
        success = False
        for attempt_idx, response in enumerate(responses):
            extracted = extract_answer(response)
            if extracted and extracted == answer and attempt_idx < 2:
                success = True
                break
        if success:
            pass2 += 1

    return RunMetrics(total=total, pass1=pass1, pass2=pass2)


def main() -> None:
    if not AIME_VAL_DIR.exists():
        raise SystemExit(f"Run directory not found: {AIME_VAL_DIR}")

    ground_truth = load_ground_truth(GT_PATH)
    results: Dict[str, dict] = {}

    for run_dir in sorted(AIME_VAL_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics = compute_metrics_for_run(run_dir, ground_truth)
        if metrics:
            results[run_dir.name] = metrics.to_dict()

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote metrics for {len(results)} runs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


