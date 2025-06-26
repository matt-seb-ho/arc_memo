import logging
import os
import re
from functools import cache, partial
from pathlib import Path
from typing import Optional

import arckit
import arckit.vis as vis
import drawsvg
import numpy as np
import orjson
import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from IPython.display import display
from PIL import Image

from concept_mem.constants import (
    BARC_DATASET_ID,
    BARC_SEED_UIDS_PATH,
    BARC_SEEDS_PATH,
    DEFAULT_CODE,
    EXCLUDED_CONCEPTS,
    REPO_ROOT,
)
from concept_mem.types import IOPair, Problem

# set up logging
logger = logging.getLogger(__name__)


DEFAULT_ENV_FILE = "/home/matt/.env"
# max line width:
# max dimension is 30 so that's
# [space][open brace][30 * ([digit][comma][space]))][close brace][comma]
# that's 4 chars for outer braces, and 30 * 3 = 90 chars for pixels
# to be safe, we'll use 120 as the max line width
MAX_LINE_WIDTH = 120


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_hf_access_token(env_file: str = DEFAULT_ENV_FILE) -> str:
    load_dotenv(dotenv_path=env_file)
    return os.getenv("HF_ACCESS_TOKEN")


def display_img_file(f):
    img = Image.open(f)
    display(img)
    img.close()


def draw_io_grids(
    task: arckit.data.Task,
    idx: int,
    add_size: bool = False,
    label: bool = False,
    pixel_scale: int = 40,
    out_file: Optional[str] = None,
) -> drawsvg.Drawing:
    # create individual grids
    arr0, arr1 = task.train[idx]
    labels = ("Input", "Output") if label else (None, None)
    grid_i, origin_i, size_i = vis.draw_grid(
        arr0, add_size=add_size, group=True, label=labels[0]
    )
    grid_o, _, size_o = vis.draw_grid(
        arr1, add_size=add_size, group=True, label=labels[1]
    )
    # combine
    width = size_i[0] + size_o[0]
    height = max(size_i[1], size_o[1])
    combined = drawsvg.Drawing(width, height)
    combined.append(drawsvg.Use(grid_i, -origin_i[0], -origin_i[1]))
    combined.append(drawsvg.Use(grid_o, size_i[0] - origin_i[0], -origin_i[1]))
    combined.set_pixel_scale(pixel_scale)
    # save to file
    if out_file:
        vis.output_drawing(combined, out_file)
    return combined


def grid_repr(grid: np.ndarray, show_size: bool = True) -> str:
    grid_string = np.array2string(
        grid,
        separator=", ",
        max_line_width=MAX_LINE_WIDTH,
    )
    if show_size:
        grid_string += f"\ngrid size = {grid.shape}"
    return grid_string


def example_repr(grid_pair: tuple, show_size: bool = True) -> str:
    if show_size:
        i_shape = grid_pair[0].shape
        o_shape = grid_pair[1].shape
        return (
            f"Input, size = ({i_shape[0]}, {i_shape[1]}):\n"
            f"{grid_repr(grid_pair[0], show_size=False)}\n"
            f"Output, size = ({o_shape[0]}, {o_shape[1]}):\n"
            f"{grid_repr(grid_pair[1], show_size=False)}"
        )
    else:
        return f"Input:\n{grid_repr(grid_pair[0])}\nOutput:\n{grid_repr(grid_pair[1])}"


def task_repr(task: arckit.data.Task, test: bool = True) -> str:
    lines = []
    for i in range(1, len(task.train) + 1):
        lines.append(f"# Example {i}")
        lines.append(example_repr(task.train[i]))
    if test:
        for i in range(len(task.test)):
            lines.append(f"# Test Example Input {i}")
            lines.append(grid_repr(task.test[i][0]))
    return "\n".join(lines)


def read_json(file_path: str | Path) -> dict:
    file_path_string = str(file_path)
    if not file_path_string.startswith("/"):
        file_path = REPO_ROOT / file_path_string
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def write_json(data: dict, file_path: str, indent: bool = False) -> None:
    if indent:
        byte_string = orjson.dumps(data, option=orjson.OPT_INDENT_2)
    else:
        byte_string = orjson.dumps(data)
    with open(file_path, "wb") as f:
        f.write(byte_string)


def read_json_lines(file_path: str) -> list:
    lines = []
    with open(file_path, "rb") as f:
        for line in f:
            lines.append(orjson.loads(line))
    return lines


def write_json_lines(data: list, file_path: str, append: bool = False) -> None:
    file_path = REPO_ROOT / file_path
    file_mode = "ab" if (append and Path(file_path).exists()) else "wb"
    with open(file_path, file_mode) as f:
        for line in data:
            f.write(orjson.dumps(line) + b"\n")


def read_yaml(src: Path | str) -> dict | list:
    if isinstance(src, Path):
        with open(src) as f:
            return yaml.safe_load(f)
    return yaml.safe_load(src)


def write_yaml(data: dict | list, file_path: Path) -> None:
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


@cache
def load_barc() -> Dataset:
    dataset = load_dataset(BARC_DATASET_ID, split="train")
    return dataset


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


def visualize_problem(problem: Problem, **kwargs):
    task = convert_problem_to_task(problem, "custom", dataset="custom")
    return arckit.vis.draw_task(task, include_test="all", **kwargs)


def visualize_barc_problem(idx: int, dataset: Optional[Dataset] = None, **kwargs):
    if dataset is None:
        dataset = load_barc()
    problem = get_problem_from_barc(dataset, idx)
    visualize_problem(problem, **kwargs)


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
        uids = set(
            read_json(REPO_ROOT / "data/vlm_desc_retrieval/validation_n100_uids.json")
        )
        problem_list = [problem for problem in full_problem_list if problem.uid in uids]
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'validation'")
    return {problem.uid: problem for problem in problem_list}


def get_arc_problem_by_uid(uid: str) -> tuple[Problem | None, str | None]:
    """
    Get an ARC problem by its UID.
    Returns None if the problem does not exist.
    """
    for split in ["train", "validation"]:
        problems = load_arc_data(split)
        if uid in problems:
            return problems[uid], split
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


def extract_comment_sections(solution: str) -> dict[str, str]:
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


# model output parsing
code_block_pattern = re.compile(r"```\n(.*?)\n```", re.DOTALL)
python_block_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
yaml_block_pattern = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)


def extract_code_block_contents(s: str, first_pattern: re.Pattern | None) -> str | None:
    if first_pattern is not None:
        m = first_pattern.search(s)
        if m:
            return m.group(1).strip()
    m = code_block_pattern.search(s)
    if m:
        return m.group(1).strip()
    return None


extract_yaml_block = partial(
    extract_code_block_contents, first_pattern=yaml_block_pattern
)
extract_python_block = partial(
    extract_code_block_contents, first_pattern=python_block_pattern
)
