import logging
import tempfile
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
    # set up file paths and write prompts and metadata
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_file = output_dir / "gen_progress.json"
        write_json(prompts, output_dir / "prompts.json")
        write_json(metadata, output_dir / "metadata.json")
    else:
        logger.info(
            "no output directory specified, using sys tmp dir for progress file"
        )
        tmp_dir = Path(tempfile.gettempdir())
        progress_file = tmp_dir / "gen_progress.json"

    # save prompts only for dry runs
    if dry_run:
        logger.info("dry run enabled, returning dummy results without generation")
        return [["dummy"] for _ in prompts]

    # generate completions and track token usage
    previous_token_usage = llm_client.get_token_usage_dict()
    results = await llm_client.async_batch_generate(
        prompts=prompts,
        model=model,
        gen_cfg=gen_cfg,
        progress_file=progress_file,
    )
    token_usage = llm_client.get_token_usage_dict()
    token_info = {
        "before": previous_token_usage,
        "after": token_usage,
    }

    # write generation artifacts to output directory
    if output_dir:
        # save metadata, prompt info, model output, and token usage
        write_json(results, output_dir / "model_outputs.json")
        write_json(token_info, output_dir / "token_usage.json")
    else:
        tmp_dir = Path(tempfile.gettempdir())
        write_json(token_info, tmp_dir / "token_usage.json")

    # pass results through
    return results
