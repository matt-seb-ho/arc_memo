"""
Gemini 2.5 Lite self-reflection stage for AIME problems.

Takes the joint reasoning outputs, asks the model to review its own attempt,
and produces a refined thought process without relying on an explicit answer key.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.aime_math import load_aime_data
from concept_mem.utils import read_json, run_llm_job, write_json

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM_PROMPT = """You are reviewing your own solution to an AIME problem.

Your goal is to produce an improved thought process:
- Audit the previous reasoning and check the logic carefully.
- Retain useful strategies, calculations, and insights.
- Identify any dubious steps and correct them or flag the uncertainty.
- You may run mental calculations or short code snippets if helpful.
- Do not assume the final answer is correct just because it was written.
- Do not expect external feedback; rely on internal verification.

Write a clear, step-by-step reasoning narrative that would be useful to revisit later."""

REFLECTION_PROMPT_TEMPLATE = """### Problem
{problem_text}

### Previous Attempt
Reasoning:
{attempt_reasoning}

Final Answer Reported: {attempt_answer}

### Task
Review the attempt, assess the logic, and produce an improved thought process that:
- Keeps correct insights and derivations.
- Repairs or replaces questionable steps.
- Notes any uncertainty that cannot be resolved internally.
- Ends with a concise summary of the proposed solution path (do NOT restate a boxed answer).
"""


async def run_reflection(
    problems: Dict[str, dict],
    joint_responses: Dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path,
    dry_run: bool = False,
) -> Dict[str, str]:
    prompts: list[str] = []
    problem_ids: list[str] = []

    for pid, prob in problems.items():
        joint = joint_responses.get(pid, {})
        attempt_reasoning = joint.get("raw_response", "").strip()
        attempt_answer = joint.get("answer", "") or "unknown"
        if not attempt_reasoning:
            logger.warning("%s: missing joint reasoning; using placeholder.", pid)
            attempt_reasoning = "(No reasoning recorded.)"
        prompt = REFLECTION_PROMPT_TEMPLATE.format(
            problem_text=prob["problem"],
            attempt_reasoning=attempt_reasoning,
            attempt_answer=attempt_answer,
        )
        prompts.append(prompt)
        problem_ids.append(pid)

    llm_client.system_prompt = REFLECTION_SYSTEM_PROMPT
    responses = await run_llm_job(
        prompts=prompts,
        metadata=problem_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    reflections: Dict[str, str] = {}
    extraction_failures: list[dict[str, str]] = []

    for pid, response_list in zip(problem_ids, responses):
        if not response_list:
            logger.warning("%s: no reflection generated", pid)
            reflections[pid] = ""
            continue
        response = response_list[0]
        if not response or not response.strip():
            logger.warning("%s: empty reflection response", pid)
            reflections[pid] = ""
            extraction_failures.append(
                {"problem_id": pid, "response_preview": ""}
            )
            continue
        reflections[pid] = response.strip()

    if extraction_failures:
        write_json(extraction_failures, output_dir / "reflection_warnings.json")
        print("  ⚠ Warnings saved to reflection_warnings.json")

    return reflections


async def async_main(cfg: DictConfig) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info("Output directory: %s", output_dir)

    provider = Provider(cfg.model.provider)
    model = cfg.model.name
    llm_client = LLMClient(
        provider=provider,
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=REPO_ROOT / ".env",
    )
    gen_cfg = hydra.utils.instantiate(cfg.generation)

    problems_data = load_aime_data(split=cfg.data.split)
    
    # Filter by problem_ids if specified
    if cfg.data.get('problem_ids'):
        problems_data = {
            pid: p for pid, p in problems_data.items()
            if pid in cfg.data.problem_ids
        }
    # Or limit by num_problems if specified
    elif cfg.data.get('num_problems'):
        problems_data = dict(list(problems_data.items())[:cfg.data.num_problems])
    
    problems = {
        problem.problem_id: {
            "problem": problem.problem_text,
            "answer": problem.answer,
        }
        for problem in problems_data.values()
    }

    joint_path = cfg.get("joint_responses_file")
    if not joint_path:
        raise ValueError("joint_responses_file must be provided.")
    joint_file = Path(joint_path)
    if not joint_file.is_absolute():
        joint_file = REPO_ROOT / joint_path
    if not joint_file.exists():
        raise FileNotFoundError(f"joint responses file not found: {joint_file}")
    joint_responses = read_json(joint_file)

    reflections = await run_reflection(
        problems=problems,
        joint_responses=joint_responses,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.get("dry_run", False),
    )

    write_json(reflections, output_dir / "thought_processes.json")
    print(f"\n✓ Refined thought processes saved to: {output_dir}/thought_processes.json")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()


