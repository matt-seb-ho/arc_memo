import asyncio
import logging
from pathlib import Path
from typing import Callable

import hydra
import yaml
from llmplus import GenerationConfig, LLMClient
from omegaconf import DictConfig
from tqdm import tqdm

from concept_mem.concept_memory import ConceptMemory
from concept_mem.constants import DOTENV_PATH, HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.types import Problem
from concept_mem.utils import (
    get_arc_problem_by_id,
    load_arc_data,
    read_json,
    read_yaml,
    run_llm_job,
    write_json,
)

logger = logging.getLogger(__name__)

annotation_instruction_path = (
    REPO_ROOT / "data/abstract_anno/annotation_instruction_v7.yaml"
)
hand_annotation_path = REPO_ROOT / "data/abstract_anno/icl_anno_v8.yaml"

ANNOTATE_TEMPLATE = """\
# Introduction
Given a puzzle containing input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Within a puzzle, each pair follows the same transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and is often the background color.

# Instructions
Examine and annotate a puzzle solution.
- Format your final annotation inside a markdown yaml block.
- The following block demonstrates this yaml format and fields to include with values describing instructions and guidelines you should follow for the corresponding fields.
```yaml
{annotation_instruction_block}
```

# Concept Repository
Here is the current concept repository:
{concept_list}

# Examples
Here are examples of puzzle solutions and expected annotation:
{examples}

# Your Puzzle Solution
Create the annotation for the following puzzle solution:
```python
{solution}
```"""

ANNOTATION_EXAMPLE_TEMPLATE = """\
{header}
Puzzle Solution:
```python
{solution}
```
Annotation:
```yaml
{annotation}
```"""


def format_annotation_example(
    puzzle: Problem,
    annotation: dict,
    header: str,
    transform_solution: Callable[[str], str] | None = None,
) -> str:
    # get solution by loading
    solution = puzzle.code
    if transform_solution:
        solution = transform_solution(solution)

    # want the following order: pseudocode, specific, general, concepts
    key_order = ["pseudocode", "general", "concepts"]
    reordered = {key: annotation[key] for key in key_order}
    annotation_str = yaml.dump(reordered, sort_keys=False)
    return ANNOTATION_EXAMPLE_TEMPLATE.format(
        header=header,
        solution=solution,
        annotation=annotation_str.strip(),
    )


def format_annotation_examples(
    puzzles: dict[str, Problem],
    example_annotations: dict[str, dict],
    skip_puzzles: list[str] | None = None,
    transform_solution: Callable[[str], str] | None = None,
) -> str:
    # make a list of formatted examples
    # join with "\n\n"
    # use `## Example {i}` as headers
    formatted_examples = []
    for i, (puzzle_id, annotation) in enumerate(example_annotations.items()):
        if skip_puzzles and puzzle_id in skip_puzzles:
            continue
        header = f"## Example {i + 1}"
        puzzle = puzzles[puzzle_id]
        example = format_annotation_example(
            puzzle, annotation, header, transform_solution
        )
        formatted_examples.append(example)
    return "\n\n".join(formatted_examples)


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


async def run_annotation(
    target_puzzles: dict[str, Problem],
    instruction_block: str,
    examples: str,
    concept_mem: ConceptMemory,
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    puzzle_ids = []
    prompts = []
    model_outputs = []
    for puzzle_id, puzzle in tqdm(target_puzzles.items(), total=len(target_puzzles)):
        puzzle_ids.append(puzzle_id)
        processed_solution = remove_barc_concepts_from_solution(puzzle.code)
        prompt = ANNOTATE_TEMPLATE.format(
            annotation_instruction_block=instruction_block,
            concept_list=concept_mem.to_string(),
            examples=examples,
            solution=processed_solution,
        )
        if dry_run:
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                dry_run_prompt_file = output_dir / f"dr_{puzzle_id}_prompt.yaml"
                dry_run_prompt_file.write_text(prompt)
                logger.info(f"wrote example prompt to {dry_run_prompt_file}")
            else:
                logger.info(f"prompt: {prompt}")
            return
        prompts.append(prompt)

        try:
            completions = await llm_client.async_generate(
                prompt=prompt,
                model=model,
                gen_cfg=gen_cfg,
            )
            model_outputs.append(completions)
            completion = completions[0]
        except Exception as e:
            logger.info(e)
            continue

        # update the memory with the new annotation
        concept_mem.update_from_model_output(
            puzzle_id=puzzle_id, model_output=completion
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # write prompts and model outputs
        write_json(puzzle_ids, output_dir / "metadata.json")
        write_json(prompts, output_dir / "prompts.json")
        write_json(model_outputs, output_dir / "model_outputs.json")

        # save token usage
        token_usage = llm_client.get_token_usage_dict()
        write_json(token_usage, output_dir / "token_usage.json")

        # save the concept memory state
        concept_mem.save_to_file(output_dir / "concept_memory.json")
        logger.info(f"wrote to {output_dir}")


async def batch_annotate():
    instr_yaml_block = annotation_instruction_path.read_text()
    hand_annotations = read_yaml(hand_annotation_path)

    concept_mem = ConceptMemory()
    concept_mem.initialize_from_annotations(hand_annotations)

    barc_seeds = load_arc_data("barc_seeds")
    target_seeds = [
        seed_id for seed_id in barc_seeds if seed_id not in hand_annotations
    ]

    # create prompts
    examples = format_annotation_examples(
        puzzles=barc_seeds,
        example_annotations=hand_annotations,
        skip_puzzles=target_seeds,
        transform_solution=remove_barc_concepts_from_solution,
    )
    prompts = []
    for seed_id in target_seeds:
        puzzle = barc_seeds[seed_id]
        processed_solution = remove_barc_concepts_from_solution(puzzle.code)
        prompt = ANNOTATE_TEMPLATE.format(
            annotation_instruction_block=instr_yaml_block,
            concept_list=concept_mem.to_string(),
            examples=examples,
            solution=processed_solution,
        )
        prompts.append(prompt)

    # run the LLM job
    llm_client = LLMClient(
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    gen_cfg = GenerationConfig()
    output_dir = REPO_ROOT / "data/anno_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = await run_llm_job(
        prompts=prompts,
        metadata=target_seeds,
        llm_client=llm_client,
        model="gpt-4.1-2025-04-14",
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=False,
    )

    for seed_id, completions in zip(target_seeds, responses):
        completion = completions[0]
        # update the concept memory with the new annotation
        concept_mem.update_from_model_output(puzzle_id=seed_id, output=completion)

    concept_mem.save_to_file(output_dir / "concept_memory.yaml")


async def async_main(cfg: DictConfig) -> None:
    # output directory setup
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # load instructions and ICL demos
    barc_seeds = load_arc_data("barc_seeds")
    if cfg.annotate.instruction_block_file:
        annotation_instruction_path = Path(cfg.annotate.instruction_block_file)
    instr_yaml_block = annotation_instruction_path.read_text().strip()
    if cfg.annotate.hand_annotations_file:
        hand_annotation_path = Path(cfg.annotate.hand_annotations_file)
    hand_annotations = read_yaml(hand_annotation_path)
    examples = format_annotation_examples(
        puzzles=barc_seeds,
        example_annotations=hand_annotations,
        transform_solution=remove_barc_concepts_from_solution,
    )

    # set up target puzzles
    target_puzzles = {}
    puzzle_limit = cfg.annotate.limit_problems
    if cfg.annotate.problem_ids is None:
        pzids = list(barc_seeds.keys())
    else:
        pzids = read_json(cfg.annotate.problem_ids)
    for pzid in pzids:
        if pzid in hand_annotations:
            continue
        target_puzzles[pzid] = get_arc_problem_by_id(pzid)
        if puzzle_limit and len(target_puzzles) >= puzzle_limit:
            break

    # memory setup
    concept_mem = ConceptMemory()
    concept_mem.initialize_from_annotations(hand_annotations)

    # model related setup
    llm_client = LLMClient(
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    gen_cfg = hydra.utils.instantiate(cfg.annotate.generation)

    await run_annotation(
        target_puzzles=target_puzzles,
        instruction_block=instr_yaml_block,
        examples=examples,
        concept_mem=concept_mem,
        llm_client=llm_client,
        model=cfg.annotate.model.name,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.dry_run,
    )


@hydra.main(
    version_base=None,
    config_path=HYRDA_CONFIG_PATH,
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the annotation process.
    """
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
