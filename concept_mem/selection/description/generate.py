import asyncio
import logging
from pathlib import Path

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import (
    DOTENV_PATH,
    HYRDA_CONFIG_PATH,
    REPO_ROOT,
)
from concept_mem.description.parse import parse_obs_spec_output, reformat_description
from concept_mem.description.prompts import build_image_caption_query_messages
from concept_mem.evaluation.run import _load_problems
from concept_mem.types import Problem
from concept_mem.utils import read_json, write_json

logger = logging.getLogger(__name__)


DESC_CONFIG_PATH = str(Path(HYRDA_CONFIG_PATH) / "description")


async def generate_image_captions(
    problems: dict[str, Problem],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path,
    concept_list: str | None = None,
    include_puzzle_text: bool = False,
    dry_run: bool = False,
) -> dict[str, str]:
    """
    Generate image captions for a given set of problems using an LLM client.
    """
    uids = []
    queries = []
    logger.info("Preparing prompts...")
    for uid, problem in problems.items():
        uids.append(uid)
        queries.append(
            build_image_caption_query_messages(
                problem,
                concept_list=concept_list,
                include_puzzle_text=include_puzzle_text,
            )
        )
    if dry_run:
        # write the queries to a file and exit
        logger.info("Dry run mode: writing queries to file and exiting.")
        write_json(queries, output_dir / "queries.json")
        return {}
    logger.info("Generating captions...")
    responses = await llm_client.async_batch_generate(
        prompts=queries,
        model=model,
        gen_cfg=gen_cfg,
        progress_file=(output_dir / "gen_progress.json"),
    )
    logger.info("Processing responses...")
    results = {}
    descriptions = {}
    for uid, completion_list in zip(uids, responses):
        response = completion_list[0]
        parsed_output = parse_obs_spec_output(response)
        results[uid] = parsed_output
        descriptions[uid] = reformat_description(parsed_output)
    write_json(results, output_dir / "caption_data.json")
    write_json(descriptions, output_dir / "descriptions.json")
    return results


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

    # llm generation preparation
    llm_client = LLMClient(
        provider=Provider(cfg.model.provider),
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    gen_cfg = hydra.utils.instantiate(cfg.generation)

    # prepare concept list
    if isinstance(cfg.module.concept_list, (str, Path)):
        concept_list = read_json(cfg.module.concept_list)
    else:
        concept_list = cfg.module.concept_list

    # generate image captions
    if cfg.module.mode == "image_caption":
        await generate_image_captions(
            problems=problems,
            llm_client=llm_client,
            model=cfg.model.name,
            gen_cfg=gen_cfg,
            output_dir=output_dir,
            concept_list=concept_list,
            include_puzzle_text=cfg.module.include_puzzle_text,
            dry_run=cfg.dry_run,
        )
    else:
        raise NotImplementedError(
            f"Mode {cfg.mode} is not implemented. Please choose 'image_caption'."
        )

    # save token usage
    write_json(
        llm_client.get_token_usage_dict(),
        output_dir / "token_usage.json",
    )


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
