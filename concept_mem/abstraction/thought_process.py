import asyncio
import logging
from pathlib import Path

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import DEFAULT_CODE, HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.data.barc_seed_processing import extract_barc_seed_comment_sections
from concept_mem.evaluation.prompts import format_puzzle_for_prompt
from concept_mem.utils import (
    read_json,
    run_llm_job,
    write_json,
)

logger = logging.getLogger(__name__)

DEFAULT_THOUGHT_PROCESS_GEN_CFG = GenerationConfig(
    temperature=0.3,
    max_tokens=1024,
)

DEFAULT_THOUGHT_PROCESS_EXAMPLE_FILES = [
    Path("data/icl_examples/thought_process/cf98881b.md"),
    Path("data/icl_examples/thought_process/4093f84a.md"),
    Path("data/icl_examples/thought_process/5168d44c.md"),
]

THOUGHT_PROCESS_FS_TEMPLATE = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

We are trying to learn from puzzles solutions to improve our puzzle solving capabilities. Your task is to analyze a puzzle and its solution (the transformation rule) write out a thought process of someone solving the puzzle including things they might try but may not work.

We provide some examples of (puzzle, solution, thought process) to help you understand the task, and then ask you to analyze a puzzle and its solution.

{examples}

### Your Puzzle Grid Pairs
{puzzle}

### Your Puzzle Solution
{solution}

### Instruction
Write out the thought process of someone solving the puzzle.
- do not include the code implementation part of solving the puzzle; going as far as identifying the rule in text is sufficient
- organize the thought process as a list of thoughts and observations marked as follows:
```
- t 1: [thought 1]
- o 1: [observation 1]
...
```
"""


def get_soluton_summary(problem: Problem, description_only: bool = False) -> str | None:
    """
    Extract a solution summary from a BARC seed puzzle solution
    where summary := the description comment (and optionally the concepts comment)
    NOTE: the summary excludes the solution code itself
    """
    problem_code = getattr(problem, "code", None)
    if problem_code is None or problem_code == DEFAULT_CODE:
        return None
    comments = extract_barc_seed_comment_sections(problem.code)
    concepts = comments.get("concepts", "")
    description = comments.get("description", "")
    if concepts == "" or description == "":
        logger.info(f"concept/description extraction failed on barc seed {problem.uid}")
    if description_only:
        return f"# description:\n{description}"
    return f"# concepts:\n{concepts}\n\n# description:\n{description}"


def format_thought_process_examples(examples):
    pieces = []
    for i, example in enumerate(examples):
        pieces.append(example.format(example_num=i + 1))
    return "\n\n".join(pieces)


def prepare_prompts(
    problem_solutions: dict[str, str],
    use_barc_solution: bool = True,
    example_files: list[Path] = DEFAULT_THOUGHT_PROCESS_EXAMPLE_FILES,
) -> tuple[list[str], list[str]]:
    """Prepare prompts for the thought process task. Returns (prompts, uids)."""
    prompts = []
    uids = []

    example_texts = [(REPO_ROOT / f).read_text() for f in example_files]
    thought_process_examples = format_thought_process_examples(example_texts)

    barc_seeds = load_arc_data("barc_seeds")

    for uid, solution in problem_solutions.items():
        uids.append(uid)
        if use_barc_solution and uid in barc_seeds:
            problem = barc_seeds[uid]
            concepts, description = extract_barc_seed_comment_sections(problem.code)
            solution = f"```python\n# concepts:\n{concepts}\n\n# description:\n{description}\n```"
        else:
            problem = Problem.from_puzzle_id(uid)
            if problem is None:
                logger.warning(f"Problem with UID {uid} not found in the dataset.")
                continue
        puzzle = format_puzzle_for_prompt(
            problem=problem,
            include_dim=True,
            include_test=False,
        )
        fewshot_thought_prompt = THOUGHT_PROCESS_FS_TEMPLATE.format(
            examples=thought_process_examples,
            puzzle=puzzle,
            solution=solution,
        ).strip()
        prompts.append(fewshot_thought_prompt)
    return prompts, uids


async def thought_process(
    problem_solutions: dict[str, str] | None,
    llm_client: LLMClient,
    model: str,
    output_dir: Path | None = REPO_ROOT / "data/thought_process",
    example_files: list[Path] = DEFAULT_THOUGHT_PROCESS_EXAMPLE_FILES,
    gen_cfg: GenerationConfig = DEFAULT_THOUGHT_PROCESS_GEN_CFG,
    dry_run: bool = False,
) -> tuple[dict[str, str], dict]:
    """returns dict[uid, thought_process], token_usage_dict"""
    prompts, uids = prepare_prompts(
        problem_solutions=problem_solutions,
        example_files=example_files,
    )
    outputs = await run_llm_job(
        prompts=prompts,
        metadata=uids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    thought_processes = {uid: output for uid, output in zip(uids, outputs)}
    token_usage_dict = llm_client.get_token_usage_dict()
    return thought_processes, token_usage_dict


async def async_main(cfg: DictConfig) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")
    provider = Provider(cfg.abstraction.model.provider)
    model = cfg.abstraction.model.name
    print(
        f"model: {model}, provider: {provider.value}, "
        f"temperature: {cfg.abstraction.generation.temperature}, "
        f"max_tokens: {cfg.abstraction.generation.max_tokens}"
    )
    gen_cfg = hydra.utils.instantiate(cfg.abstraction.generation)
    llm_client = LLMClient(
        provider=provider,
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=REPO_ROOT / ".env",
    )

    problem_solutions = cfg.abstraction.problem_solutions
    if problem_solutions is None:
        problem_solutions = {uid: "" for uid in load_arc_data("barc_seeds")}
    else:
        problem_solutions = read_json(problem_solutions)

    thought_procs, _ = await thought_process(
        problem_solutions=problem_solutions,
        llm_client=llm_client,
        model=model,
        output_dir=output_dir,
        gen_cfg=gen_cfg,
        dry_run=cfg.dry_run,
    )
    write_json(thought_procs, output_dir / "thought_processes.json")
    logger.info(f"Output directory: {output_dir}")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
