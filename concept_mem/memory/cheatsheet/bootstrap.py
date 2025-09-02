import asyncio
import logging
from pathlib import Path

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig
from tqdm import tqdm

from concept_mem.constants import DOTENV_PATH, HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.evaluation.prompts import format_puzzle_for_prompt
from concept_mem.utils import (
    read_json,
    write_json,
)

from .prompts import ARC_CHEATSHEET_BOOTSTRAP_PROMPT

logger = logging.getLogger(__name__)

INITIAL_SHEET = "(empty)"


async def bootstrap_cheatsheet(
    solved_problems: list[Problem],
    solutions: dict[str, str],
    prompt_template: str,
    output_dir: Path,
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    initial_sheet: str = INITIAL_SHEET,
    **gen_kwargs,
) -> str:
    # following DC-Cu implementation
    cheatsheet = initial_sheet
    metadata = []
    prompts = []
    cheatsheet_history = []
    cheatsheet_path = output_dir / "cheatsheet_history.json"
    for i, problem in tqdm(enumerate(solved_problems), total=len(solved_problems)):
        prompt = _format_update_cheatsheet_prompt(
            previous_cheatsheet=cheatsheet,
            problem=problem,
            solution=solutions[problem.uid],
            prompt_template=prompt_template,
        )
        metadata.append(problem.uid)
        prompts.append(prompt)
        try:
            responses = await llm_client.async_generate(
                prompt=prompt,
                model=model,
                gen_cfg=gen_cfg,
                **gen_kwargs,
            )
            updated_cheatsheet = extract_cheatsheet(
                response=responses[0],
                old_cheatsheet=cheatsheet,
            )
            cheatsheet_history.append(
                {"idx": i, "problem": problem.uid, "cheatsheet": updated_cheatsheet}
            )
            cheatsheet = updated_cheatsheet
            write_json(cheatsheet_history, cheatsheet_path)
        except Exception as e:
            logger.error(
                f"Error generating cheatsheet for problem {problem.id} (skipping update): {e}"
            )
            continue

    # get token usage
    token_usage_dict = llm_client.get_token_usage_dict()

    # write artifacts to file
    write_json(metadata, output_dir / "metadata.json")
    write_json(prompts, output_dir / "prompts.json")
    write_json(token_usage_dict, output_dir / "token_usage.json")
    write_json(cheatsheet_history, cheatsheet_path)
    final_cheatsheet_path = output_dir / "final_cheatsheet.txt"
    final_cheatsheet_path.write_text(cheatsheet)

    logger.info(f"Token usage: {token_usage_dict}")
    logger.info(f"Wrote artifacts to {output_dir}")

    return cheatsheet


def _format_update_cheatsheet_prompt(
    previous_cheatsheet: str,
    problem: Problem,
    solution: str,
    prompt_template: str = ARC_CHEATSHEET_BOOTSTRAP_PROMPT,
) -> str:
    return prompt_template.format(
        previous_cheatsheet=previous_cheatsheet,
        puzzle=format_puzzle_for_prompt(problem, include_dim=True),
        solution=solution,
    )


def extract_cheatsheet(
    response: str,
    old_cheatsheet: str,
) -> str:
    """
    Extracts the cheatsheet from the model response.

    Arguments:
        response : str : The response from the model.
        old_cheatsheet : str : The old cheatsheet to return if the new one is not found.

    Returns:
        str : The extracted cheatsheet (if not found, returns the old cheatsheet).
    """
    response = response.strip()
    # <cheatsheet> (content) </cheatsheet>
    if "<cheatsheet>" in response:
        try:
            txt = response.split("<cheatsheet>")[1].strip()
            txt = txt.split("</cheatsheet>")[0].strip()
            return txt
        except Exception:
            logger.debug("Error extracting cheatsheet from response")
            return old_cheatsheet
    else:
        return old_cheatsheet


async def async_main(cfg: DictConfig):
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # load problems
    # TODO: fix this
    dataset = load_arc_data(cfg.annotate.data.split)
    # if cfg.annotate.data.problems:
    #     # load uids from config directly or from specified file
    #     if isinstance(cfg.annotate.data.problems, str):
    #         problem_ids = read_json(cfg.annotate.data.problems)
    #     else:
    #         problem_ids = cfg.annotate.data.problems
    # else:
    #     problem_ids = list(dataset.keys())
    problem_ids = list(dataset.keys())
    if cfg.annotate.data.limit_problems:
        problem_ids = problem_ids[: cfg.annotate.data.limit_problems]
    problems = [dataset[_id] for _id in problem_ids]

    # get solutions
    if cfg.annotate.solutions:
        solutions = read_json(cfg.annotate.solutions)
    else:
        # use BARC annotations
        solutions = {problem.uid: problem.code for problem in problems}

    llm_client = LLMClient(
        provider=Provider(cfg.annotate.model.provider),
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    gen_cfg = hydra.utils.instantiate(cfg.annotate.generation)

    # TODO:
    # - make prompt dict with config specifying key
    # - add config provided initial sheet file
    final_cheatsheet = await bootstrap_cheatsheet(
        solved_problems=problems,
        solutions=solutions,
        prompt_template=ARC_CHEATSHEET_BOOTSTRAP_PROMPT,
        output_dir=output_dir,
        llm_client=llm_client,
        model=cfg.annotate.model.name,
        gen_cfg=gen_cfg,
    )

    # convert into a hint file for some target uids
    # hint_file_uids = read_json(REPO_ROOT / cfg.data.hint_file_target_uids)
    # hint_file_contents = {uid: final_cheatsheet for uid in hint_file_uids}
    # write_json(hint_file_contents, output_dir / "hint_file.json")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig):
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
