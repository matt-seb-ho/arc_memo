import logging
from pathlib import Path

from llmplus import GenerationConfig, LLMClient

from concept_mem.utils import write_json

logger = logging.getLogger(__name__)


async def run_llm_job(
    prompts: list[str],
    metadata: list[str | list[str]],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path | None,
    dry_run: bool = False,
) -> list[list[str]]:
    # prepare file paths
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_file = output_dir / "gen_progress.json"
    else:
        progress_file = "gen_progress.json"

    # save prompts only for dry runs
    prompt_info = {
        "prompts": prompts,
        "metadata": metadata,
    }
    if dry_run:
        logger.info(
            "dry run enabled, writing prompts to output directory and returning dummy strings"
        )
        if output_dir:
            write_json(prompt_info, output_dir / "prompts.json")
        return [["dummy"] for _ in prompts]

    # generate completions
    results = await llm_client.async_batch_generate(
        prompts=prompts,
        model=model,
        gen_cfg=gen_cfg,
        progress_file=progress_file,
    )

    # write artifacts to output directory
    if output_dir:
        # save metadata, prompt info, model output, and token usage
        write_json(metadata, output_dir / "metadata.json")
        write_json(prompts, output_dir / "prompts.json")
        write_json(results, output_dir / "model_outputs.json")
        token_usage = llm_client.get_token_usage_dict()
        write_json(token_usage, output_dir / "token_usage.json")

    # pass results through
    return results
