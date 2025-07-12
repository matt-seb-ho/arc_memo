import asyncio
import logging
from pathlib import Path
from typing import Callable

import hydra
import yaml
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import DATA_DIR, DOTENV_PATH, HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.arc_agi import load_arc_data
from concept_mem.utils import (
    parse_markup_tag,
    read_json,
    read_yaml,
    run_llm_job,
    write_json,
)
from concept_mem.utils.barc_seed_processing import (
    remove_concepts_from_barc_seed_solution,
)

logger = logging.getLogger(__name__)

PSEUDOCODE_INSTR_PATH = DATA_DIR / "abstract_anno/op2/pseudocode_instr.txt"
PSEUDOCODE_INSTR = PSEUDOCODE_INSTR_PATH.read_text()


PSEUDOCODE_GEN_EX_TEMPLATE = """\
{header}
Puzzle Solution:
```python
{solution}
```
Annotation:
<pseudocode>
{pseudocode}
</pseudocode>
<summary>
{summary}
</summary>"""


def format_pseudocode_examples(
    problem_solutions: dict[str, str],
    annotations: dict[str, dict],
    header_template: str = "## Example {example_number}",
    transform_solution: Callable[[str], str] | None = None,
    delimiter: str = "\n\n",
) -> str:
    # format ICL examples for problem solution -> pseudocode task
    example_strings = []
    for i, (problem_id, annotation) in enumerate(annotations.items(), start=1):
        if problem_id not in problem_solutions:
            logger.info(f"Missing solution for ICL example problem {problem_id}")
            continue
        solution = problem_solutions[problem_id]
        if "pseudocode" not in annotation or "summary" not in annotation:
            logger.info(f"ICL example {problem_id} missing pseudocode or summary")
            continue
        if transform_solution:
            solution = transform_solution(solution)
        formatted_pseudocode = yaml.dump(annotation["pseudocode"], sort_keys=False)
        example = PSEUDOCODE_GEN_EX_TEMPLATE.format(
            header=header_template.format(example_number=i),
            solution=solution,
            pseudocode=formatted_pseudocode.strip(),
            summary=annotation["summary"],
        )
        example_strings.append(example)
    return delimiter.join(example_strings)


def parse_model_output(
    model_output: str,
) -> tuple[str, str]:
    # returns (pseudocode, summary)
    code_results = parse_markup_tag(model_output, "pseudocode")
    if len(code_results) != 1:
        logger.info(
            f"parse error: expected 1 pseudocode block, got {len(code_results)}"
        )
        pseudocode = ""
    else:
        pseudocode = code_results[0].strip()
    summary_results = parse_markup_tag(model_output, "summary")
    if len(summary_results) != 1:
        logger.info(
            f"parse error: expected 1 summary block, got {len(summary_results)}"
        )
        summary = ""
    else:
        summary = summary_results[0].strip()
    return pseudocode, summary


async def generate_pseudocode(
    problems: list[str],
    solutions: dict[str, str],
    examples: dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    # prepare ICL demo string
    formatted_examples = format_pseudocode_examples(
        problem_solutions=solutions,
        annotations=examples,
    )
    # prepare prompts
    puzzle_ids = []
    prompts = []
    for puzzle_id in problems:
        if puzzle_id not in solutions:
            logger.warning(f"Missing solution for puzzle {puzzle_id}, skipping")
            continue
        prompt = PSEUDOCODE_INSTR.format(
            examples=formatted_examples,
            solution=solutions[puzzle_id],
        ).strip()
        puzzle_ids.append(puzzle_id)
        prompts.append(prompt)

    # run LLM job
    model_output = await run_llm_job(
        prompts=prompts,
        metadata=puzzle_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    # parse results
    results = {}
    for puzzle_id, completions in zip(puzzle_ids, model_output):
        if not completions:
            logger.warning(f"No completions for puzzle {puzzle_id}")
            continue
        completion = completions[0]
        try:
            pseudocode, summary = parse_model_output(completion)
            results[puzzle_id] = {
                "pseudocode": pseudocode,
                "summary": summary,
                "solution": solutions[puzzle_id],
            }
        except Exception as e:
            logger.error(f"Error parsing output for puzzle {puzzle_id}: {e}")
            continue

    # save to output directory if specified
    if output_dir:
        output_file = output_dir / "initial_analysis.json"
        write_json(results, output_file, indent=True)
    return results


async def async_main(cfg: DictConfig) -> None:
    # output directory setup
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # load instructions and ICL demos
    hand_annotation_path = Path(cfg.annotate.hand_annotations_file)
    hand_annotations = read_yaml(hand_annotation_path)

    # set up target puzzles and solutions
    limit = cfg.annotate.limit_problems
    target_puzzles = []
    solutions = {}
    if cfg.annotate.problem_ids is None:
        barc_seeds = load_arc_data("barc_seeds")
        if cfg.annotate.solutions:
            file_solutions = read_json(cfg.annotate.solutions)
        else:
            file_solutions = {}
        for pzid in barc_seeds:
            seed_puzzle = barc_seeds[pzid]
            if pzid in file_solutions:
                solutions[pzid] = file_solutions[pzid]
            else:
                solutions[pzid] = remove_concepts_from_barc_seed_solution(
                    seed_puzzle.code or "# no solution provided",
                )
            if pzid in hand_annotations or (limit and len(target_puzzles) >= limit):
                continue
            target_puzzles.append(pzid)
    else:
        pzids = read_json(cfg.annotate.problem_ids)
        solutions = read_json(cfg.annotate.solutions)
        if limit:
            target_puzzles = pzids[:limit]
        else:
            target_puzzles = pzids

    # model related setup
    llm_client = LLMClient(
        provider=Provider(cfg.annotate.model.provider),
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    gen_cfg = hydra.utils.instantiate(cfg.annotate.generation)

    # run pseudocode generation
    await generate_pseudocode(
        problems=target_puzzles,
        solutions=solutions,
        examples=hand_annotations,
        llm_client=llm_client,
        model=cfg.annotate.model.name,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.dry_run,
    )
    logger.info(f"Wrote to output directory: {output_dir}")


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
