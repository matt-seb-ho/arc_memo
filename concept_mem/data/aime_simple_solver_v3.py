"""
AIME solver v3 - Ultra-strict output format
- No fragile parsing needed
- Model outputs ONLY the integer, nothing else
- Much more robust than v1/v2
"""

import asyncio
import json
import logging
import re
from pathlib import Path

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.aime_math import load_aime_data
from concept_mem.utils import read_json, run_llm_job, write_json

logger = logging.getLogger(__name__)

# Ultra-strict system prompt - emphasizes output format
AIME_SYSTEM_PROMPT = """You are solving AIME (American Invitational Mathematics Examination) problems.

You may think step by step internally, but your final answer MUST appear exactly once as \\boxed{XYZ}.

Requirements for the final output:
- Use \\boxed{XYZ} for the answer and nowhere else in the response.
- \\boxed{XYZ} must contain only the integer (0–999) with no spaces, wording, punctuation, or leading zeros (unless the answer is 0).
- Do not print unboxed standalone integers on their own lines; embed intermediate numbers inside sentences or equations.

Example of correct output: \\boxed{237}

Examples of incorrect output: "Final Answer: 237", "237 ", "\\boxed{The answer is 237}", "\\boxed{0237}", or any response with multiple boxed expressions."""

# Minimal user prompt - start from the problem text
AIME_SOLVE_PROMPT = """Solve this AIME problem:

{problem}"""
AIME_OUTPUT_REQUIREMENT = """Output only the integer answer, and wrap it exactly once as \\boxed{XYZ}.

Guidelines:
- Mention intermediate numbers only inside sentences/equations so they are not mistaken for answers.
- Do not use \\boxed anywhere else in the response.
- The grader will read the boxed integer wherever it appears."""


def truncate_hint_lessons(hint_text: str, max_lessons: int | None) -> str:
    """Return only the first `max_lessons` lesson bullets from hint text."""
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
    trimmed_lines = []
    for block in limited_blocks:
        trimmed_lines.extend(block)
    return "\n".join(trimmed_lines).strip()


def extract_answer_from_response(response: str) -> tuple[str, bool]:
    """
    Extract numerical answer from model response.
    
    Prefers the last \\boxed{XYZ} expression, but falls back to older heuristics
    when the model fails to follow instructions.
    
    Returns:
        (answer string, used_box_format flag)
    """
    if not response or not response.strip():
        return "", False
    
    response = response.strip()
    
    # Prefer the final boxed integer anywhere in the response
    boxed = re.findall(r'\\boxed\{(\d{1,3})\}', response)
    if boxed:
        num_str = boxed[-1]
        num = int(num_str)
        if 0 <= num <= 999:
            return num_str, True
    
    if response.isdigit():
        num = int(response)
        if 0 <= num <= 999:
            return response, False
    
    match = re.search(r'[Ff]inal\s+[Aa]nswer\s*:\s*(\d+)', response)
    if match:
        num_str = match.group(1)
        if 0 <= int(num_str) <= 999:
            return num_str, False
    
    match = re.search(r'[Aa]nswer\s*:\s*(\d+)', response)
    if match:
        num_str = match.group(1)
        if 0 <= int(num_str) <= 999:
            return num_str, False
    
    match = re.search(r'[Tt]he\s+answer\s+is\s+(\d+)', response)
    if match:
        num_str = match.group(1)
        if 0 <= int(num_str) <= 999:
            return num_str, False
    
    numbers = re.findall(r'\b(\d{1,3})\b', response[-500:])
    if numbers:
        for num_str in reversed(numbers):
            num = int(num_str)
            if 0 <= num <= 999:
                return num_str, False
    
    return "", False
    
    
def build_problem_prompt(
    problem_text: str,
    context: dict[str, str] | None,
    lesson_limit: int | None = None,
) -> str:
    """Compose the user prompt, optionally including retrieval hints."""
    components = [
        AIME_SOLVE_PROMPT.format(problem=problem_text.strip()),
    ]
    if context:
        hint = context.get("hint") or context.get("lessons")
        if hint:
            hint = truncate_hint_lessons(hint, lesson_limit)
            if hint:
                components.extend(
                    [
                        "",
                        "Lessons distilled from similar problems:",
                        hint.strip(),
                    ]
                )
    components.extend(["", AIME_OUTPUT_REQUIREMENT])
    return "\n".join(components)


def load_problem_contexts(prompt_cfg: DictConfig | None) -> dict[str, dict[str, str]]:
    """Load retrieval hints formatted like prompt_info.json."""
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
        variant_entry = None
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


async def solve_aime_simple(
    problems: dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path,
    dry_run: bool = False,
    problem_contexts: dict[str, dict[str, str]] | None = None,
    lesson_limit: int | None = None,
) -> dict[str, str]:
    """
    Solve AIME problems with ultra-strict output format (v3)
    
    Returns:
        {problem_id: answer_string}
    """
    
    problem_ids = []
    prompts = []
    contexts = problem_contexts or {}
    
    for pid, prob_data in problems.items():
        problem_ids.append(pid)
        context = contexts.get(pid)
        prompt = build_problem_prompt(prob_data['problem'], context, lesson_limit)
        prompts.append(prompt)
    
    print(f"Solving {len(prompts)} AIME problems with {model}...")
    print(f"Using v3 solver with ultra-strict output format (integer only)")
    
    # Set system prompt on client
    llm_client.system_prompt = AIME_SYSTEM_PROMPT
    
    responses = await run_llm_job(
        prompts=prompts,
        metadata=problem_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run
    )
    
    # Extract answers from responses
    solutions = {}
    full_responses = {}  # Save for debugging
    extraction_failures = []
    max_samples = getattr(gen_cfg, "n", 1) or 1
    pass_thresholds = [k for k in (1, 2) if k <= max_samples]
    pass_hits = {k: 0 for k in pass_thresholds}
    
    for pid, response_list in zip(problem_ids, responses):
        if not response_list:
            logger.warning(f"{pid}: No response received")
            solutions[pid] = ""
            full_responses[pid] = []
            continue
        
        # Try each response in order (n=3 means 3 attempts)
        answer = ""
        valid_responses = []
        gt_answer = problems.get(pid, {}).get("answer", "")
        gt_answer = gt_answer.strip() if isinstance(gt_answer, str) else ""
        per_problem_pass = {k: False for k in pass_thresholds}
        
        for i, response in enumerate(response_list):
            if not response or not response.strip():
                logger.warning(f"{pid}: Empty response in attempt {i+1}")
                continue
            
            valid_responses.append(response)
            extracted, used_box = extract_answer_from_response(response)
            
            if extracted:
                if not answer:
                    answer = extracted
                    if used_box:
                        logger.info(f"{pid}: ✓ Boxed answer '{answer}' from attempt {i+1}")
                    else:
                        logger.info(
                            f"{pid}: ⚠ Extracted '{answer}' from attempt {i+1} (model didn't follow box format)"
                        )
                    first_valid_logged = True
                if (
                    gt_answer
                    and extracted.strip() == gt_answer
                ):
                    for k in pass_thresholds:
                        if i + 1 <= k:
                            per_problem_pass[k] = True
            else:
                logger.warning(f"{pid}: Could not extract valid answer from attempt {i+1}")
                extraction_failures.append({
                    'problem_id': pid,
                    'attempt': i+1,
                    'response_preview': response[:200]
                })
        
        if not answer and valid_responses:
            logger.error(
                f"{pid}: No valid answer after {len(valid_responses)} attempts. "
                f"First response: {valid_responses[0][:100]}..."
            )
        
        solutions[pid] = answer
        full_responses[pid] = valid_responses
        for k in pass_thresholds:
            if per_problem_pass[k]:
                pass_hits[k] += 1
    
    extracted_count = sum(1 for a in solutions.values() if a)
    num_correct = sum(
        1
        for pid, ans in solutions.items()
        if ans and problems.get(pid, {}).get("answer", "").strip() == ans.strip()
    )
    print(f"\nResults:")
    print(f"  Responses extracted: {extracted_count}/{len(problem_ids)}")
    print(f"  Exact matches vs. labels: {num_correct}/{len(problem_ids)}")
    for k in pass_thresholds:
        rate = pass_hits[k] / len(problem_ids) if len(problem_ids) else 0.0
        print(f"  pass@{k}: {pass_hits[k]}/{len(problem_ids)} ({rate:.2%})")
    print(f"  Extraction failures logged: {len(extraction_failures)}")
    
    # Save debugging artifacts
    write_json(full_responses, output_dir / "full_responses.json")
    if extraction_failures:
        write_json(extraction_failures, output_dir / "extraction_failures.json")
        print(f"  ⚠ Extraction failures saved to extraction_failures.json")
    
    return solutions


async def async_main(cfg: DictConfig) -> None:
    """Main entry point"""
    
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Setup LLM
    provider = Provider(cfg.model.provider)
    model = cfg.model.name
    print(f"\n{'='*70}")
    print(f"AIME Solver v3 - Ultra-Strict Output Format")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Format: Single boxed integer (\\boxed{{XYZ}})")
    print(f"{'='*70}\n")
    
    llm_client = LLMClient(
        provider=provider,
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=REPO_ROOT / ".env"
    )
    
    gen_cfg = hydra.utils.instantiate(cfg.generation)
    
    # Load AIME data
    problems_data = load_aime_data(split=cfg.data.split)
    problems = {
        p.problem_id: {'problem': p.problem_text, 'answer': p.answer}
        for p in problems_data.values()
    }
    
    print(f"Loaded {len(problems)} problems\n")
    
    prompt_cfg = cfg.get("prompt", None)
    problem_contexts = load_problem_contexts(prompt_cfg)
    if problem_contexts:
        logger.info("Using retrieval hints for inference.")
    lesson_limit = None
    if prompt_cfg is not None:
        lesson_limit = prompt_cfg.get("hint_lessons_limit")

    # Solve
    solutions = await solve_aime_simple(
        problems=problems,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.get('dry_run', False),
        problem_contexts=problem_contexts,
        lesson_limit=lesson_limit,
    )
    
    # Save solutions
    write_json(solutions, output_dir / "solutions.json")
    print(f"\n✓ Solutions saved to: {output_dir}/solutions.json")
    print(f"✓ Full responses saved to: {output_dir}/full_responses.json")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
