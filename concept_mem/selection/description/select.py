# prompt based concept retrieval
import asyncio
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

import hydra
import yaml
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from concept_mem.constants import DOTENV_PATH, HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.utils import (
    read_json,
    run_llm_job,
    write_json,
)

logger = logging.getLogger(__name__)


AIME_CONTEXT = """\
You are selecting helper lessons for an AIME (American Invitational Mathematics Examination) problem. AIME answers are three-digit integers (000–999). Typical domains include:
- Algebra (equations, inequalities, polynomials, sequences, Vieta, functional relations)
- Number theory (divisibility, primes, modular arithmetic, Diophantine equations, divisor/sum functions)
- Combinatorics and probability (counting arguments, binomial identities, probability setups)
- Geometry (plane geometry, circles, power of a point, occasional 3D configurations)

Only choose lessons whose situation genuinely matches the math structure of the given problem."""


LLM_RET_TOPK_SITUATION_TEMPLATE = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of situations and a puzzle description.
- Your task is to identify the most relevant {top_k} situations for the given puzzle.
- Please output your final selection as a list of situation numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Situation List
{concept_list}

### Puzzle Description
{description} 
"""

AIME_RET_TOPK_SITUATION_TEMPLATE = """\
### Context
{aime_context}

### Task
We will provide you with a numbered list of situations and an AIME problem statement.
- Select the {top_k} situations whose advice would help solve the problem.
- Return the situation numbers in a markdown YAML list:
```yaml
- 3
- 12
- 19
```

### Situations
{concept_list}

### AIME Problem
{description}
"""

LLM_RET_SCORE_SITUATION_TEMPLATE = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "situations" that suggest how to approach the puzzle.

### Instructions
Given the list of situations and the puzzle description, assign a **relevance score** (between 0.0 and 1.0) to each situation based on how relevant it is for solving the puzzle.

Please output your scores as a markdown YAML dictionary like below:
```yaml
1: 0.92
2: 0.13
3: 0.78
```
### Situation List
{concept_list}

### Puzzle Description
{description}
"""

AIME_RET_SCORE_SITUATION_TEMPLATE = """\
### Context
{aime_context}

### Task
Given the AIME problem and situation list, assign each situation a relevance score between 0.0 and 1.0. High scores should go to situations whose suggestion directly matches the problem’s structure or topic.

Output the scores in YAML dictionary form.

### Situations
{concept_list}

### AIME Problem
{description}
"""

LLM_RET_TOPK_CONCEPT_TEMPLATE = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons and a puzzle description.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Puzzle Description
{description} 
"""

AIME_RET_TOPK_CONCEPT_TEMPLATE = """\
### Context
{aime_context}

### Task
You are given an AIME problem and a catalog of lessons. Each lesson has:
- **situation** – when the lesson is applicable
- **suggestion** – the recommended strategy or check

Choose the {top_k} lessons whose situations *truly* match the structure of the
problem and whose suggestions would materially help reach the boxed integer
answer. Favor lessons that align with the problem's mathematical domain and the
specific reasoning steps implied by the statement. Discard lessons that only
vaguely relate. If relevance is uncertain, prefer lessons that offer diagnostic
checks (e.g., parity, modular sanity checks) to catch mistakes.

Return the lesson numbers in a YAML list, ordered by decreasing relevance, like this:
```yaml
- 3
- 12
- 19
- 7
- 25
```

### Lessons
{concept_list}

### AIME Problem
{description}
"""

LLM_RET_SCORE_CONCEPT_TEMPLATE = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving concepts/lessons/rules that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
Given the list of situations and the puzzle description, assign a **relevance score** (between 0.0 and 1.0) to each lesson based on how relevant it is for solving the puzzle.

Please output your scores as a markdown YAML dictionary like below:
```yaml
1: 0.92
2: 0.13
3: 0.78
```
### Concept List
{concept_list}

### Puzzle Description
{description}
"""

AIME_RET_SCORE_CONCEPT_TEMPLATE = """\
### Context
{aime_context}

### Task
Score each lesson (0.0–1.0) for how relevant its situation/suggestion is to the given AIME problem. Prioritize lessons whose math domain and structure align closely with the problem.

Return a YAML dictionary mapping lesson numbers to scores.

### Lessons
{concept_list}

### AIME Problem
{description}
"""

TOPK_HINT_V2 = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons and a puzzle description.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Puzzle Description
{description} 
"""

AIME_TOPK_HINT_TEMPLATE = """\
### Context
{aime_context}

### Task
Select the {top_k} lessons from the list below whose situations and suggestions would help solve the AIME problem. Output the lesson numbers in a YAML list.

### Lessons
{concept_list}

### AIME Problem
{description}
"""

RESELECT_PROMPT = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons and a previous attempt at solving the puzzle.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Previous Attempt
{completion} 
"""

AIME_RESELECT_PROMPT = """\
### Context
{aime_context}

### Task
Given the lesson list and a previous AIME attempt, re-select the {top_k} lessons that would most improve the next attempt. Return lesson numbers in a YAML list.

### Lessons
{concept_list}

### Previous Attempt
{completion}
"""

RESELECT_PROMPT_WITH_DESC = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons, a visual description of the puzzle, and a previous attempt at solving the puzzle.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Puzzle Description
{description}

### Previous Attempt
{completion} 
"""

AIME_RESELECT_PROMPT_WITH_DESC = """\
### Context
{aime_context}

### Task
We provide (1) the lesson list, (2) the AIME problem statement, and (3) a previous attempt. Choose the {top_k} lessons whose advice best addresses this problem and prior attempt. Return lesson numbers in YAML list form.

### Lessons
{concept_list}

### AIME Problem
{description}

### Previous Attempt
{completion}
"""


CONCAT_DESC_INTRO = "Here are the descriptions of the puzzles from different sources:"
DEFAULT_SCORE_THRESHOLD = 0.5


def _normalize_description_table(raw_table: Any) -> dict[str, str]:
    """
    Normalize description json blobs to a {uid: description} mapping.

    Supports files that directly store the mapping as well as blobs with a
    `problems` list (e.g., AIME exports).
    """
    if isinstance(raw_table, dict):
        if "problems" in raw_table and isinstance(raw_table["problems"], list):
            normalized: dict[str, str] = {}
            for entry in raw_table["problems"]:
                if not isinstance(entry, dict):
                    continue
                uid = entry.get("id") or entry.get("problem_id")
                text = entry.get("problem") or entry.get("description")
                if uid and text:
                    normalized[uid] = text
            return normalized
        return raw_table
    if isinstance(raw_table, list):
        normalized: dict[str, str] = {}
        for entry in raw_table:
            if not isinstance(entry, dict):
                continue
            uid = entry.get("id") or entry.get("problem_id")
            text = entry.get("description") or entry.get("problem")
            if uid and text:
                normalized[uid] = text
        if normalized:
            return normalized
    raise ValueError("Description file must map puzzle ids to description text.")


def _write_prompt_info_file(
    output_dir: Path,
    prompt_key: str,
    retrieved_lessons: dict[str, str],
    descriptions: dict[str, str] | None,
    include_description: bool,
) -> None:
    """
    Emit a prompt_info.json artifact compatible with ArcMemo inference.
    """
    prompt_info: dict[str, dict[str, dict[str, str]]] = {}
    all_uids = set(retrieved_lessons.keys())
    if descriptions:
        all_uids.update(descriptions.keys())
    for uid in sorted(all_uids):
        hint_text = retrieved_lessons.get(uid, "")
        entry: dict[str, str] = {}
        if include_description:
            entry["description"] = (descriptions or {}).get(uid, "")
        entry["hint"] = hint_text or ""
        prompt_info[uid] = {prompt_key: entry}
    path = output_dir / "prompt_info.json"
    write_json(prompt_info, path)
    logger.info(f"Wrote prompt info to {path}")


def prepare_concept_list(
    lessons: dict[str, list[dict]],
) -> tuple[str, dict[int, tuple[str, int]]]:
    # returns a formatted string of lessons and a mapping from lesson number to (uid, index)
    concept_entries = []
    concept_number_to_uid = {}
    for uid, puzzle_lessons in lessons.items():
        for i, lesson in enumerate(puzzle_lessons):
            num = len(concept_entries) + 1
            entry = f"lesson {num}.\n- situation: {lesson['situation']}\n- suggestion: {lesson['suggestion']}"
            concept_entries.append(entry)
            concept_number_to_uid[num] = (uid, i)
    concept_list = "\n".join(concept_entries)
    return concept_list, concept_number_to_uid


async def reselect_concepts(
    puzzles: list[str],
    descriptions: dict[str, str] | None,
    completions: dict[str, str],
    lessons: dict[str, list[dict]],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    top_k: int,
    output_dir: Path | None,
    dry_run: bool = False,
    domain: str = "arc",
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Reselect concepts based on previous completion."""
    logger.info("Reselecting concepts based on previous completion...")

    # prepare prompts
    formatted_concept_list, concept_number_to_uid = prepare_concept_list(lessons)
    uids = []
    prompts = []
    aime_mode = domain.lower() == "aime"
    for uid in puzzles:
        uids.append(uid)
        if descriptions is None:
            template = AIME_RESELECT_PROMPT if aime_mode else RESELECT_PROMPT
            prompt = template.format(
                top_k=top_k,
                concept_list=formatted_concept_list,
                completion=completions[uid],
                aime_context=AIME_CONTEXT,
            )
        else:
            template = (
                AIME_RESELECT_PROMPT_WITH_DESC
                if aime_mode
                else RESELECT_PROMPT_WITH_DESC
            )
            prompt = template.format(
                top_k=top_k,
                concept_list=formatted_concept_list,
                description=descriptions[uid],
                completion=completions[uid],
                aime_context=AIME_CONTEXT,
            )
        prompts.append(prompt)

    # gather completions
    completions = await run_llm_job(
        prompts=prompts,
        metadata=uids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    # parse completions, extract retrieved lesson uids, and prepare hint file
    retrieved_concept_uids = {}
    retrieved_lessons = {}
    parsing_errors = []
    for uid, completion in zip(uids, completions):
        try:
            concept_numbers = parse_top_k_yaml_list(completion[0])
        except yaml.YAMLError as e:
            parsing_errors.append((uid, completion[0], str(e)))
            continue
        concept_uids = [concept_number_to_uid[i] for i in concept_numbers]
        retrieved_concept_uids[uid] = concept_uids
        retrieved_lessons[uid] = format_retrieved_lesson_hint(lessons, concept_uids)
    logger.info(f"Parsing error count: {len(parsing_errors)}")

    # write out parsed results to file
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(retrieved_concept_uids, output_dir / "concept_uids.json")
        write_json(retrieved_lessons, output_dir / "reselected_lessons.json")
        write_json(parsing_errors, output_dir / "parsing_errors.json")
        logger.info(f"Wrote to {output_dir}")

    return retrieved_lessons, retrieved_concept_uids


async def select_concepts(
    puzzles: list[str],
    descriptions: dict[str, str],
    lessons: dict[str, list[dict]],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    situation_only: bool = False,
    score_concepts: bool = False,
    use_score_threshold: bool = True,
    top_k: int = 3,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    top_p: float = 0.95,
    output_dir: Path = REPO_ROOT / "data/llm_concept_retrieval",
    use_hint_v2: bool = False,
    dry_run: bool = False,
    domain: str = "arc",
) -> dict[str, str]:
    # prepare prompts
    concept_entries = []
    concept_number_to_uid = {}
    if use_hint_v2:
        for num, (lesson_uid, lesson) in enumerate(lessons.items(), start=1):
            # expecting keys: title, rule, notes
            # output format:
            # {number}: {title}
            # - rule: {rule}
            # - notes: {notes}
            entry = f"{num}: {lesson['title']}\n- rule: {lesson['rule']}\n- notes: {lesson['notes']}"
            concept_entries.append(entry)
            # parse out index from lesson_uid
            try:
                lesson_idx_from_puzzle = int(lesson_uid.split("_")[-1])
            except ValueError:
                lesson_idx_from_puzzle = 0
            concept_number_to_uid[num] = (lesson_uid, lesson_idx_from_puzzle)
    elif situation_only:
        for uid, puzzle_lessons in lessons.items():
            for i, lesson in enumerate(puzzle_lessons):
                num = len(concept_entries) + 1
                entry = f"{num}. {lesson['situation']}"
                concept_entries.append(entry)
                concept_number_to_uid[num] = (uid, i)
    else:
        for uid, puzzle_lessons in lessons.items():
            for i, lesson in enumerate(puzzle_lessons):
                num = len(concept_entries) + 1
                entry = f"lesson {num}.\n- situation: {lesson['situation']}\n- suggestion: {lesson['suggestion']}"
                concept_entries.append(entry)
                concept_number_to_uid[num] = (uid, i)
    concept_list = "\n".join(concept_entries)
    uids = []
    prompts = []
    template = _route_template(
        score_concepts,
        situation_only,
        use_hint_v2,
        domain=domain,
    )
    for uid in puzzles:
        prompt = template.format(
            top_k=top_k,
            description=descriptions[uid],
            concept_list=concept_list,
            aime_context=AIME_CONTEXT,
        )
        uids.append(uid)
        prompts.append(prompt)

    # query model
    completions = await run_llm_job(
        prompts=prompts,
        metadata=uids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    # parse completions, extract retrieved lesson uids, and prepare hint file
    completion_dict = {}
    for uid, completion in zip(uids, completions):
        if not completion:
            logger.warning("No completion received for %s; skipping.", uid)
            continue
        completion_dict[uid] = completion[0]
    retrieved_concept_uids = {}
    retrieved_lessons = {}
    parsing_errors = []
    for uid, completion in completion_dict.items():
        try:
            if score_concepts:
                # get scores and threshold for numbers
                concept_scores = parse_scored_yaml_dict(completion)
                if use_score_threshold:
                    concept_numbers = [
                        i
                        for i, score in concept_scores.items()
                        if score >= score_threshold
                    ]
                else:
                    # top-p
                    sorted_scores = sorted(
                        list(concept_scores.items()), key=lambda x: x[1]
                    )
                    concept_numbers = []
                    total_score = 0.0
                    while sorted_scores and total_score < top_p:
                        num, score = sorted_scores.pop(-1)
                        concept_numbers.append(num)
                        total_score += score
            else:
                concept_numbers = parse_top_k_yaml_list(completion)
        except yaml.YAMLError as e:
            parsing_errors.append((uid, completion, str(e)))
            continue
        
        # Skip if parsing returned None
        if concept_numbers is None:
            parsing_errors.append((uid, completion, "Could not extract lesson numbers"))
            continue
        
        concept_uids = [concept_number_to_uid[i] for i in concept_numbers]
        retrieved_concept_uids[uid] = concept_uids
        retrieved_lessons[uid] = format_retrieved_lesson_hint(lessons, concept_uids)
    logger.info(f"Parsing error count: {len(parsing_errors)}")

    # write out parsed results to file
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(retrieved_concept_uids, output_dir / "retrieved_concept_uids.json")
    write_json(retrieved_lessons, output_dir / "retrieved_lessons.json")
    write_json(parsing_errors, output_dir / "parsing_errors.json")
    logger.info(f"Wrote to {output_dir}")
    return retrieved_lessons


def format_retrieved_lesson_hint(
    all_lessons: dict[str, list[dict]] | dict[str, dict[str, str]],
    retrieved_uids: list[tuple[str, int]],
) -> str:
    components = []
    value0 = next(iter(all_lessons.values()))
    if isinstance(value0, dict):
        # implies that we are using lessons: dict[f"{uid}_{idx}", dict]
        for uid, idx in retrieved_uids:
            lesson = all_lessons[uid]
            s = f"- {lesson['title']}\n  - rule: {lesson['rule']}\n  - notes: {lesson['notes']}"
            components.append(s)
    else:
        # implies that we are using lessons: dict[uid, list[dict]]
        for uid, idx in retrieved_uids:
            lesson = all_lessons[uid][idx]
            components.append(
                f"- situation: {lesson['situation']}\n  suggestion: {lesson['suggestion']}"
            )
    return "\n".join(components)


def parse_top_k_yaml_list(yaml_block: str) -> list[int] | None:
    # expected format:
    # ```yaml
    # - 18
    # - 77
    # ...
    # ```
    # step 1: remove markdown delimiters
    yaml_string = extract_first_yaml_block(yaml_block)
    
    # step 2: parse yaml string
    if yaml_string:
        try:
            yaml_data = yaml.safe_load(yaml_string)
            if isinstance(yaml_data, list):
                return yaml_data
        except yaml.YAMLError:
            pass
    
    # Fallback: extract lesson numbers from prose (e.g., "Lesson 57", "lesson 3")
    # This handles cases where the model explains choices but doesn't wrap in yaml
    numbers = re.findall(r'[Ll]esson\s+(\d+)', yaml_block)
    if numbers:
        return [int(n) for n in numbers]
    
    # Last resort: try parsing the whole block as yaml
    try:
        yaml_data = yaml.safe_load(yaml_block)
        if isinstance(yaml_data, list):
            return yaml_data
    except yaml.YAMLError:
        pass
    
    return None


def parse_scored_yaml_dict(yaml_block: str) -> dict[int, float]:
    yaml_string = extract_first_yaml_block(yaml_block) or yaml_block
    yaml_data = yaml.safe_load(yaml_string)
    if not isinstance(yaml_data, dict):
        raise ValueError("Expected a YAML dictionary of concept scores")
    return {int(k): float(v) for k, v in yaml_data.items()}


def parse_lessons_from_yaml_block(
    model_responses: dict[str, str | list[str]],
) -> tuple[dict[str, list[dict]], list[tuple]]:
    individual_lessons = {}
    errors = []
    for uid, lesson_block in tqdm(model_responses.items()):
        # step 0: handle old output format
        if isinstance(lesson_block, list):
            lesson_block = lesson_block[0]
        # step 1: remove markdown block
        yaml_string = extract_first_yaml_block(lesson_block)
        # step 2: parse yaml string
        try:
            yaml_data = yaml.safe_load(yaml_string)
        except yaml.YAMLError:
            try:
                yaml_data = parse_situation_suggestion(yaml_string)
                assert isinstance(yaml_data, list)
                assert isinstance(yaml_data[0], dict)
            except Exception as e:
                errors.append((uid, yaml_string, str(e)))
        individual_lessons[uid] = yaml_data
    # report parsing error count
    logger.info(f"{len(errors)} parsing errors")
    # report total number of lessons and average number of lessons per uid
    total_lesson_count = sum(len(v) for v in individual_lessons.values())
    logger.info(
        f"total lessons: {total_lesson_count}, average lessons per uid: {total_lesson_count / len(individual_lessons)}"
    )
    return individual_lessons, errors


def extract_first_yaml_block(text):
    pattern = r"```yaml\s*(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else None


def parse_situation_suggestion(text: str) -> list[dict]:
    entries = []
    current = {}
    current_key = None
    buffer = []

    def flush_buffer():
        if current_key and buffer:
            current[current_key] = " ".join(line.strip() for line in buffer).strip()
        buffer.clear()

    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("- situation:"):
            # Save previous entry
            flush_buffer()
            if current:
                entries.append(current)
            current = {}
            current_key = "situation"
            buffer = [line[len("- situation:") :].strip()]
        elif line.startswith("suggestion:"):
            flush_buffer()
            current_key = "suggestion"
            buffer = [line[len("suggestion:") :].strip()]
        elif line.startswith("-") and "situation:" in line:
            # Unexpected format, treat it like new situation
            flush_buffer()
            if current:
                entries.append(current)
            current = {}
            match = re.match(r"- situation:\s*(.*)", line)
            if match:
                current_key = "situation"
                buffer = [match.group(1).strip()]
        else:
            buffer.append(line)

    flush_buffer()
    if current:
        entries.append(current)

    return entries


def _combine_description_tables(
    description_tables: list[dict[str, str]],
) -> dict[str, list[str]]:
    combined = {}
    for table in description_tables:
        for uid, description in table.items():
            if uid not in combined:
                combined[uid] = []
            combined[uid].append(description)
    return combined


def _build_summary_prompt(descriptions: list[str]) -> str:
    prompt = (
        "Please summarize the following descriptions of the puzzle into a single description:\n"
        + "\n".join(f"- {desc}" for desc in descriptions)
    )
    return prompt


def _route_template(
    score_concepts: bool,
    situation_only: bool,
    use_hint_v2: bool,
    domain: str = "arc",
) -> str:
    aime_mode = domain.lower() == "aime"
    if use_hint_v2:
        if aime_mode:
            return AIME_TOPK_HINT_TEMPLATE
        return TOPK_HINT_V2
    if score_concepts:
        if situation_only:
            if aime_mode:
                return AIME_RET_SCORE_SITUATION_TEMPLATE
            return LLM_RET_SCORE_SITUATION_TEMPLATE
        if aime_mode:
            return AIME_RET_SCORE_CONCEPT_TEMPLATE
        return LLM_RET_SCORE_CONCEPT_TEMPLATE
    else:
        if situation_only:
            if aime_mode:
                return AIME_RET_TOPK_SITUATION_TEMPLATE
            return LLM_RET_TOPK_SITUATION_TEMPLATE
        if aime_mode:
            return AIME_RET_TOPK_CONCEPT_TEMPLATE
        return AIME_RET_TOPK_CONCEPT_TEMPLATE


async def get_descriptions(
    cfg: DictConfig, llm_client: LLMClient, model: Enum, output_dir: Path
) -> dict[str, str]:
    ensemble_method = cfg.selection.ensemble_method
    # handle simple case
    if (not ensemble_method) or ensemble_method == "none":
        assert isinstance(cfg.selection.description_file, str)
        raw_desc = read_json(REPO_ROOT / cfg.selection.description_file)
        desc_dict = _normalize_description_table(raw_desc)
        return desc_dict

    # combine multiple description files into dict[str, list[str]]
    assert isinstance(cfg.selection.description_file, list)
    description_tables = []
    for description_file in cfg.selection.description_file:
        raw_desc = read_json(REPO_ROOT / description_file)
        desc_dict = _normalize_description_table(raw_desc)
        description_tables.append(desc_dict)
    description_lists = _combine_description_tables(description_tables)
    descriptions = {}

    # combine descriptions
    if ensemble_method == "concatenate":
        for uid, desc_list in description_lists.items():
            concatenated_components = [CONCAT_DESC_INTRO]
            for i, desc_entry in enumerate(desc_list, start=1):
                concatenated_components.append(f"Description {i}:")
                concatenated_components.append(desc_entry)
            descriptions[uid] = "\n".join(concatenated_components)
        return descriptions
    elif ensemble_method == "llm_summary":
        uids = []
        prompts = []
        for uid, desc_list in description_lists.items():
            uids.append(uid)
            prompts.append(_build_summary_prompt(desc_list))
        # query model
        summary_generation_kwargs = OmegaConf.to_container(
            cfg.selection.summary_generation, resolve=True
        )
        completions = await llm_client.batch_generate_async(
            prompts, n=1, model=model, **summary_generation_kwargs
        )
        # logger.info(f"LLM query error count: {sum(len(e) for e in errors)}")
        descriptions = {}
        for uid, completion_list in zip(uids, completions):
            if len(completion_list) == 0:
                logger.debug(
                    f"Empty completion for uid: {uid}, defaulting to first input description."
                )
                completion = description_lists[uid][0]
            else:
                completion = completion_list[0]
            descriptions[uid] = completion
        # save summarized descriptions to file
        summarized_description_path = output_dir / "summarized_descriptions.json"
        summarized_description_info = {
            "sources": cfg.selection.description_file,
            "ensemble_method": ensemble_method,
            "descriptions": descriptions,
        }
        write_json(summarized_description_info, summarized_description_path)
        logger.info(f"Wrote summarized descriptions to {summarized_description_path}")
        return descriptions
    else:
        raise ValueError(f"Unsupported ensemble method: {ensemble_method}")


async def async_main(cfg: DictConfig) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # set up LLM client
    provider = Provider(cfg.selection.model.provider)
    model = cfg.selection.model.name
    llm_client = LLMClient(
        provider=provider,
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    logger.info(f"Using model: {model}, from provider: {provider.value}")

    # preprocess config values
    if isinstance(cfg.selection.problems, str):
        problems = read_json(REPO_ROOT / cfg.selection.problems)
    else:
        assert isinstance(cfg.selection.problems, list)
        problems = cfg.selection.problems
    descriptions = await get_descriptions(
        cfg=cfg,
        llm_client=llm_client,  # needed only for summary generation
        model=model,
        output_dir=output_dir,
    )
    lessons = read_json(REPO_ROOT / cfg.selection.lesson_file)
    gen_cfg = hydra.utils.instantiate(cfg.selection.generation)

    # retrieve concepts
    retrieved_lessons = await select_concepts(
        puzzles=problems,
        descriptions=descriptions,
        lessons=lessons,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        situation_only=cfg.selection.situation_only,
        top_k=cfg.selection.top_k,
        output_dir=output_dir,
        use_hint_v2=cfg.selection.use_hint_v2,
        domain=cfg.selection.get("domain", "arc"),
    )
    prompt_info_key = cfg.selection.get("prompt_info_key")
    if prompt_info_key:
        include_desc = cfg.selection.get(
            "prompt_info_include_description", True
        )
        _write_prompt_info_file(
            output_dir=output_dir,
            prompt_key=prompt_info_key,
            retrieved_lessons=retrieved_lessons,
            descriptions=descriptions,
            include_description=include_desc,
        )


@hydra.main(config_path=HYRDA_CONFIG_PATH, config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
