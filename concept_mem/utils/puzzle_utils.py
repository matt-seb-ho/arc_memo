import logging
from functools import cache
from typing import Optional

import arckit
import numpy as np
from datasets import Dataset, load_dataset

from concept_mem.constants import (
    BARC_DATASET_ID,
    BARC_SEED_UIDS_PATH,
    BARC_SEEDS_PATH,
    DEFAULT_CODE,
    EXCLUDED_CONCEPTS,
    REPO_ROOT,
    URL_TEMPLATE,
)
from concept_mem.types import IOPair, Problem
from concept_mem.utils.common import read_json

# set up logging
logger = logging.getLogger(__name__)


@cache
def load_barc() -> Dataset:
    dataset = load_dataset(BARC_DATASET_ID, split="train")
    return dataset


def convert_problem_to_task(
    p: Problem, id_: str, dataset: Optional[str] = None
) -> arckit.data.Task:
    # convert from list[IOPair] to list[dict]
    train_pairs = [_io_pair_to_dict(pair) for pair in p.train_pairs]
    test_pairs = [_io_pair_to_dict(pair) for pair in p.test_pairs]
    task = arckit.data.Task(
        id=id_,
        train=train_pairs,
        test=test_pairs,
        dataset=dataset,
    )
    return task


def _io_pair_to_dict(pair: IOPair) -> dict:
    # convert to list[list[int]]
    return {"input": pair.x.astype(int).tolist(), "output": pair.y.astype(int).tolist()}


def extract_code_from_seed(content: str) -> str:
    assert (
        "# ============= remove below this point for prompting =============" in content
    )
    content = content.split(
        "# ============= remove below this point for prompting ============="
    )[0].strip()
    content = content.split("def generate_input")[0].strip()
    content = content.replace("def main(", "def transform(")
    return content


def extract_code_from_source(code: str, idx: Optional[int] = None) -> str:
    # preprocess solution code
    if "def generate_input" in code:
        code = code.split("def generate_input")[0].strip()
    else:
        logger.debug(
            f"generate_input function not found in code for BARC problem {idx}"
        )
    if "def main(" in code:
        code = code.replace("def main(", "def transform(")
    else:
        logger.warning(
            f"main function not found in code for BARC problem {idx}, using default (empty) code"
        )
        code = DEFAULT_CODE
    return code


def get_problem_from_barc(
    dataset: Dataset,
    idx: int,
    uid: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Problem]:
    r = dataset[idx]

    # get input-output pairs
    all_pairs = []
    for example in r["examples"]:
        input_grid = np.array(example[0])
        output_grid = np.array(example[1])
        all_pairs.append(IOPair(input_grid, output_grid))

    # get code
    code = extract_code_from_source(r["source"], idx)

    # use first 3 pairs as train pairs and 4th pair as test pair
    problem = Problem(
        uid=uid,
        filename=filename,
        code=code,
        train_pairs=all_pairs[0:3],
        test_pairs=[all_pairs[3]],
    )
    return problem


@cache
def load_arc_data(split: str = "train") -> dict[str, Problem]:
    if split == "train":
        from arc import train_problems

        problem_list = train_problems
    elif split == "validation":
        from arc import validation_problems

        problem_list = validation_problems
    elif split == "barc_seeds":
        from arc import train_problems

        seed_uids = set(read_json(BARC_SEED_UIDS_PATH))
        problem_list = []
        for problem in train_problems:
            if problem.uid not in seed_uids:
                continue
            seed_code_path = BARC_SEEDS_PATH / f"{problem.uid}.py"
            if seed_code_path.exists():
                problem.code = extract_code_from_seed(seed_code_path.read_text())
            problem_list.append(problem)
    elif split == "val100":
        from arc import validation_problems

        full_problem_list = validation_problems
        uids = set(read_json(REPO_ROOT / "data/testbeds/validation_n100_uids.json"))
        problem_list = [problem for problem in full_problem_list if problem.uid in uids]
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'validation'")
    return {problem.uid: problem for problem in problem_list}


def get_arc_problem_by_id(puzzle_id: str) -> tuple[Problem | None, str | None]:
    """
    Get an ARC problem by its UID.
    Returns None if the problem does not exist.
    """
    for split in ["train", "validation"]:
        problems = load_arc_data(split)
        if puzzle_id in problems:
            return problems[puzzle_id], split
    return None, None


@cache
def load_barc_seeds():
    from arc import train_problems

    seeds = []
    seed_uids = set(read_json(BARC_SEED_UIDS_PATH))
    for problem in train_problems:
        if problem.uid in seed_uids:
            # barc_seeds[problem.uid] = problem
            source_path = BARC_SEEDS_PATH / f"{problem.uid}.py"
            if source_path.exists():
                problem.code = extract_code_from_seed(source_path.read_text())
            seeds.append(problem)
    return seeds


def load_arc_problems(uids: list[str]) -> dict[str, Problem]:
    return {uid: Problem(seed_id=uid) for uid in uids}


def extract_barc_concepts(code: str, exclude: set[str] = EXCLUDED_CONCEPTS) -> str:
    code_lines = code.split("\n")
    concept_list = _get_concepts_from_lines(code_lines)
    concept_list = [c for c in concept_list if c.lower() not in exclude]
    return ", ".join(concept_list)


def _get_concepts_from_lines(lines):
    concepts = []
    for i, line in enumerate(lines):
        if "# concepts:" in line:
            in_line_concepts = lines[i][12:]
            if in_line_concepts.strip() != "":
                concepts.extend(lines[i][12:].split(","))
            while (
                i + 1 < len(lines)
                and lines[i + 1].startswith("# ")
                and not lines[i + 1].startswith("# description:")
            ):
                concepts.extend(lines[i + 1][2:].split(","))
                i += 1
            concepts = [c.strip() for c in concepts]
            break
    if concepts == []:
        for i, line in enumerate(lines):
            if "concepts:" in line.lower():
                in_line_concepts = lines[i][12:]
                if in_line_concepts.strip() != "":
                    concepts.extend(lines[i][12:].split(","))
                while (
                    i + 1 < len(lines)
                    and lines[i + 1].startswith("# ")
                    and not lines[i + 1].lower().startswith("description:")
                ):
                    concepts.extend(lines[i + 1][2:].split(","))
                    i += 1
                concepts = [c.strip() for c in concepts]
                break
    return concepts


def extract_barc_seed_comment_sections(solution: str) -> dict[str, str]:
    """
    Scan through `lines` until the first function definition, pulling out
    any comment‑blocks labelled "# concepts:" and "# description:".

    Returns a dict with keys 'concepts' and/or 'description' if found.
    """
    lines = solution.splitlines()
    sections = {"concepts": [], "description": []}
    current = None

    for line in lines:
        stripped = line.strip()
        # stop once we hit the function definition
        if stripped.startswith("def "):
            break

        # is it a comment?
        if stripped.startswith("#"):
            # drop leading '#' and whitespace
            content = stripped[1:].strip()

            # section headers?
            if content.lower().startswith("concepts:"):
                current = "concepts"
                tail = content[len("concepts:") :].strip()
                if tail:
                    sections[current].append(tail)

            elif content.lower().startswith("description:"):
                current = "description"
                tail = content[len("description:") :].strip()
                if tail:
                    sections[current].append(tail)

            # continuation line?
            elif current in sections:
                if content:
                    sections[current].append(content)

        else:
            # non‑comment line — reset until we hit next header
            current = None

    # join each section’s lines into one paragraph
    return {
        key: " ".join(lines)
        for key, lines in sections.items()
        if lines  # only include non‑empty sections
    }


def get_puzzle_url(puzzle_id: str, verbose: bool = True, v: bool = True) -> str:
    url = URL_TEMPLATE.format(puzzle_id=puzzle_id)
    if not (verbose and v):
        print(url)
    return url


def remove_barc_concepts_from_solution(solution: str) -> str:
    # we want to remove the "# concepts:" line and the line after it
    lines = solution.split("\n")
    new_lines = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if line.startswith("# concepts:"):
            skip_next = True
            continue
        new_lines.append(line)
    return "\n".join(new_lines)
