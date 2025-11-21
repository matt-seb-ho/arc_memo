"""
Gemini 2.5 Lite joint solver for AIME:
- Generates reasoning + final answer in a single pass.
- Saves both the extracted integer answer and the full narrative for downstream reflection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.aime_math import load_aime_data
from concept_mem.utils import read_json, run_llm_job, write_json

logger = logging.getLogger(__name__)

JOINT_SYSTEM_PROMPT = """You are solving problems from the American Invitational Mathematics Examination (AIME).

Think carefully, explain your reasoning clearly, and finish with a single line of the form:
Final Answer: XYZ

where XYZ is an integer between 0 and 999. Do not place the answer in LaTeX boxes."""

JOINT_PROMPT_TEMPLATE = """### Problem
{problem_text}

Provide a detailed solution strategy. You may include short code snippets, equations, or checks if helpful.
After you are fully satisfied with the reasoning, end with:
Final Answer: XYZ"""


def truncate_hint_lessons(hint_text: str, max_lessons: int | None) -> str:
    if not hint_text or max_lessons is None:
        return hint_text
    if max_lessons <= 0:
        return ""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in hint_text.splitlines():
        if line.startswith("- ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    if len(blocks) <= max_lessons:
        return hint_text
    limited_blocks = blocks[:max_lessons]
    trimmed_lines: list[str] = []
    for block in limited_blocks:
        trimmed_lines.extend(block)
    return "\n".join(trimmed_lines).strip()


def load_problem_contexts(prompt_cfg: DictConfig | None) -> dict[str, dict[str, str]]:
    if not prompt_cfg:
        return {}
    problem_data_path = prompt_cfg.get("problem_data")
    if not problem_data_path:
        return {}
    data_path = Path(problem_data_path)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / problem_data_path
    if not data_path.exists():
        logger.warning("Prompt data file not found: %s", data_path)
        return {}
    prompt_info = read_json(data_path)
    contexts: dict[str, dict[str, str]] = {}
    preferred_variant = prompt_cfg.get("problem_data_variant")
    for pid, variants in prompt_info.items():
        if not isinstance(variants, dict) or len(variants) == 0:
            continue
        if preferred_variant and preferred_variant in variants:
            variant_entry = variants[preferred_variant]
        else:
            variant_entry = next(iter(variants.values()))
        if isinstance(variant_entry, dict):
            contexts[pid] = variant_entry
    logger.info(
        "Loaded prompt data for %d problems from %s", len(contexts), data_path
    )
    return contexts


def build_joint_prompt(
    problem_text: str,
    context: dict[str, str] | None,
    lesson_limit: int | None = None,
) -> str:
    components = [JOINT_PROMPT_TEMPLATE.format(problem_text=problem_text.strip())]
    if context:
        hint = context.get("hint") or context.get("lessons")
        if hint:
            hint = truncate_hint_lessons(hint, lesson_limit)
        if hint:
            components.extend(
                [
                    "",
                    "### Retrieved Context",
                    hint.strip(),
                    "",
                    "Integrate useful insights from the context above if appropriate.",
                ]
            )
    return "\n".join(components)


def extract_final_answer(response: str) -> str:
    if not response:
        return ""
    match = re.search(r"Final\s+Answer\s*:\s*(\d{1,3})", response, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    numbers = re.findall(r"\b(\d{1,3})\b", response[-500:])
    if numbers:
        for num_str in reversed(numbers):
            num = int(num_str)
            if 0 <= num <= 999:
                return num_str
    return ""


async def joint_solve(
    problems: dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path,
    problem_contexts: dict[str, dict[str, str]] | None = None,
    lesson_limit: int | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    problem_ids: list[str] = []
    prompts: list[str] = []
    contexts = problem_contexts or {}

    for pid, prob_data in problems.items():
        problem_ids.append(pid)
        context = contexts.get(pid)
        prompt = build_joint_prompt(
            problem_text=prob_data["problem"],
            context=context,
            lesson_limit=lesson_limit,
        )
        prompts.append(prompt)

    print(f"Solving {len(prompts)} AIME problems with {model} (joint reasoning)...")
    llm_client.system_prompt = JOINT_SYSTEM_PROMPT

    responses = await run_llm_job(
        prompts=prompts,
        metadata=problem_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    solutions: dict[str, str] = {}
    narratives: dict[str, dict[str, str]] = {}
    extraction_failures: list[dict[str, str]] = []
    correct = 0

    for pid, response_list in zip(problem_ids, responses):
        if not response_list:
            logger.warning("%s: no response returned", pid)
            solutions[pid] = ""
            narratives[pid] = {"raw_response": "", "answer": ""}
            continue

        response = response_list[0]
        answer = extract_final_answer(response)
        if not answer:
            logger.warning("%s: could not extract answer from response", pid)
            extraction_failures.append(
                {"problem_id": pid, "response_preview": response[:200]}
            )
        solutions[pid] = answer
        narratives[pid] = {
            "raw_response": response,
            "answer": answer,
        }

        gt = problems.get(pid, {}).get("answer", "")
        if gt and answer and gt.strip() == answer.strip():
            correct += 1

    extracted = sum(1 for ans in solutions.values() if ans)
    total = len(problem_ids)
    print("\nResults:")
    print(f"  Answers extracted: {extracted}/{total}")
    print(f"  Exact matches vs. labels: {correct}/{total} ({correct/total:.2%})")
    if extraction_failures:
        write_json(extraction_failures, output_dir / "extraction_failures.json")
        print("  ⚠ Extraction failures saved to extraction_failures.json")

    return solutions, narratives


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

    prompt_cfg = cfg.get("prompt", None)
    problem_contexts = load_problem_contexts(prompt_cfg)
    lesson_limit = None
    if prompt_cfg is not None:
        lesson_limit = prompt_cfg.get("hint_lessons_limit")

    solutions, narratives = await joint_solve(
        problems=problems,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        problem_contexts=problem_contexts,
        lesson_limit=lesson_limit,
        dry_run=cfg.get("dry_run", False),
    )

    write_json(solutions, output_dir / "solutions.json")
    write_json(narratives, output_dir / "joint_responses.json")
    print(f"\n✓ Solutions saved to: {output_dir}/solutions.json")
    print(f"✓ Joint reasoning saved to: {output_dir}/joint_responses.json")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()


