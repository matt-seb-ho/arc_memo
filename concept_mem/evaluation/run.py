import asyncio
import logging
import random
from pathlib import Path
from typing import Optional

import hydra
from llmplus import LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import (
    DOTENV_PATH,
    HYRDA_CONFIG_PATH,
    REPO_ROOT,
)
from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.evaluation.prompts import SYSTEM_PROMPTS
from concept_mem.evaluation.runner import EvaluationRunner
from concept_mem.utils import read_json

logger = logging.getLogger(__name__)


def _load_problems(
    dataset: str,
    split: str,
    num_problems: Optional[int],
    problem_ids: Optional[list | str],
) -> dict[str, Problem]:
    """Load ARC‑AGI problems and subset them according to the config."""
    if dataset.lower() == "arc-agi":
        data = load_arc_data(split)
        if problem_ids is None:
            problem_ids = list(data.keys())
        elif isinstance(problem_ids, str):
            problem_ids = read_json(REPO_ROOT / problem_ids)
        elif num_problems and num_problems < len(problem_ids):
            problem_ids = random.sample(problem_ids, num_problems)
        return {pid: data[pid] for pid in problem_ids}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


async def async_main(cfg: DictConfig) -> None:
    # set up output_dir for file writing and logging
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # data and prompt preparation
    problems = _load_problems(
        dataset=cfg.data.dataset,
        split=cfg.data.split,
        num_problems=cfg.data.num_problems,
        problem_ids=cfg.data.problem_ids,
    )
    prompt_options = hydra.utils.instantiate(cfg.prompt)

    # llm generation preparation
    llm_client = LLMClient(
        provider=Provider(cfg.model.provider),
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    llm_client.system_prompt = SYSTEM_PROMPTS[cfg.prompt.system_prompt_key]
    logger.info(f"Using system prompt: {cfg.prompt.system_prompt_key}")
    retry_policy = hydra.utils.instantiate(cfg.puzzle_retry)
    gen_cfg = hydra.utils.instantiate(cfg.generation)
    lcs_cfg = hydra.utils.instantiate(cfg.long_cot_selection)

    # initialize and run
    eval_runner = EvaluationRunner(
        llm=llm_client,
        model=cfg.model.name,
        prompt_options=prompt_options,
        retry_policy=retry_policy,
        gen_cfg=gen_cfg,
        long_cot_sel_cfg=lcs_cfg,
        output_dir=output_dir,
        dry_run=cfg.dry_run,
    )
    await eval_runner.run(problems=problems)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
