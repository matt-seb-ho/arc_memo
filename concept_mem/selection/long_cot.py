import logging
import re
from pathlib import Path

import yaml
from llmplus import GenerationConfig, LLMClient

from concept_mem.concept_memory import ConceptMemory
from concept_mem.evaluation.prompts import ARC_INTRO, format_puzzle_for_prompt
from concept_mem.types import Problem
from concept_mem.utils import run_llm_job, write_json

logger = logging.getLogger(__name__)

SPECIAL_CONCEPT_CLASS_NOTE = """\
- note 2 special concept classes:
1. guide object: reference objects informing output constructs' color, shape, position, direction, size, etc.
2. criteria: predicates that help with conditional operations (only executing under certain conditions or executing differently based on conditions).
"""

CONCEPT_SELECTION_PROMPT_TEMPLATE = f"""\
{ARC_INTRO}

### Puzzle Grids
{{puzzle_grids}}

### Concepts
{SPECIAL_CONCEPT_CLASS_NOTE}
{{concepts}}

### Instructions
Please identify the concepts most relevant to this puzzle by thinking through possible solution. Output your selected list of concepts along with notes summarizing what you tried, the corresponding results, and your observations about the puzzle.
- Please format your output inside a yaml code block (inside triple backticks with the yaml). e.g.:
```yaml
concepts:
- concept 1
- concept 2
...
notes: notes about what you observe/tried/concluded
```
- Be sure to have top level keys `concepts` and `notes` with `concepts` mapping to a list of strings (concept names), and `notes` mapping to a string summarizing your thought process.
- Please use the exact name of the concept as it appears in the concepts list.
- The purpose of this task is as a first attempt in a collaborative effort to solve the puzzle. Future attempts willl have access to further details about the concepts you selected as well as your notes from this query so be sure to be thorough in your notes to avoid duplicating work.
- It is vital that your notes do not contain misleading information. Only include information you are confident about.
- If you think the puzzle contains a concept that is not in the list, you can include it in the list anyways, we just won't be able to provide examples of that concept until we have a solved puzzle that uses it."""


relaxed_yaml_pattern = re.compile(r"```yaml(.*?)```", re.DOTALL)


async def select_concepts_using_long_cot(
    puzzles: dict[str, Problem],
    concept_mem: ConceptMemory,
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, list[str] | str]]:
    # prepare prompts
    puzzle_ids = []
    prompts = []
    for puzzle_id, puzzle in puzzles.items():
        puzzle_grids = format_puzzle_for_prompt(puzzle)
        prompt = CONCEPT_SELECTION_PROMPT_TEMPLATE.format(
            puzzle_grids=puzzle_grids,
            concepts=concept_mem.to_string(),
        )
        puzzle_ids.append(puzzle_id)
        prompts.append(prompt)

    # generate responses
    responses = await run_llm_job(
        prompts=prompts,
        metadata=puzzle_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    # parse response
    puzzle_responses = {}
    res = {}
    for puzzle_id, response_list in zip(puzzle_ids, responses):
        if len(response_list) == 0:
            logger.info(f"No response for puzzle {puzzle_id}, skipping.")
            continue
        response = response_list[0]
        puzzle_responses[puzzle_id] = response
        try:
            if "```yaml" in response and response.count("```") == 1:
                response = response + "\n```"
            m = relaxed_yaml_pattern.search(response)
            if m is None:
                block = response
            else:
                block = m.group(1).strip()
            parsed = yaml.safe_load(block)
            parsed.setdefault("notes", "")
            parsed.setdefault("concepts", [])
            res[puzzle_id] = parsed
        except Exception as e:
            logger.info(f"Error processing puzzle {puzzle_id}: {e}, output: {response}")

    # save parsed output
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(res, output_dir / "concept_selection.json")

    return res
