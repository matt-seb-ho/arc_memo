"""
Improved AIME solver for reasoning models (o4-mini, o3-mini)
- Allows models to show their reasoning
- Extracts final answer from the full response
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
from concept_mem.utils import run_llm_job, write_json

logger = logging.getLogger(__name__)

# Modified prompts for reasoning models
AIME_SYSTEM_PROMPT = """You are solving AIME (American Invitational Mathematics Examination) problems.
Think through the problem step by step, then provide your final answer in the format:
Final Answer: [number]
where [number] is an integer between 0 and 999."""

AIME_SOLVE_PROMPT = """Solve this AIME problem:

{problem}

Show your reasoning, then provide your final answer in the format:
Final Answer: [number]"""


def extract_answer_from_response(response: str) -> str:
    """
    Extract numerical answer from reasoning model response.
    Tries multiple patterns to find the answer.
    
    Returns:
        answer string (empty if not found)
    """
    if not response or not response.strip():
        return ""
    
    # Pattern 1: "Final Answer: 123" or "Final answer: 123"
    match = re.search(r'[Ff]inal\s+[Aa]nswer\s*:\s*(\d+)', response)
    if match:
        return match.group(1)
    
    # Pattern 2: "Answer: 123" at the end
    match = re.search(r'[Aa]nswer\s*:\s*(\d+)\s*$', response.strip())
    if match:
        return match.group(1)
    
    # Pattern 3: "The answer is 123" near the end
    match = re.search(r'[Tt]he\s+answer\s+is\s+(\d+)', response[-200:])
    if match:
        return match.group(1)
    
    # Pattern 4: Number in a box (LaTeX or markdown)
    match = re.search(r'\\boxed\{(\d+)\}', response)
    if match:
        return match.group(1)
    
    # Pattern 5: Last standalone number (0-999)
    numbers = re.findall(r'\b(\d{1,3})\b', response)
    if numbers:
        # Return the last number that's in valid AIME range
        for num in reversed(numbers):
            if 0 <= int(num) <= 999:
                return num
    
    return ""


async def solve_aime_simple(
    problems: dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path,
    dry_run: bool = False
) -> dict[str, str]:
    """
    Solve AIME problems with reasoning models
    
    Returns:
        {problem_id: answer_string}
    """
    
    problem_ids = []
    prompts = []
    
    for pid, prob_data in problems.items():
        problem_ids.append(pid)
        prompt = AIME_SOLVE_PROMPT.format(problem=prob_data['problem'])
        prompts.append(prompt)
    
    print(f"Solving {len(prompts)} AIME problems with {model}...")
    print(f"Using reasoning-friendly prompts (allows model to show work)")
    
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
    
    # Extract answers from full responses
    solutions = {}
    full_responses = {}  # Save full responses for debugging
    
    for pid, response_list in zip(problem_ids, responses):
        if not response_list:
            logger.warning(f"No response for {pid}")
            solutions[pid] = ""
            full_responses[pid] = []
            continue
        
        # Try each response in order (n=3 means 3 attempts)
        answer = ""
        valid_responses = []
        
        for i, response in enumerate(response_list):
            if not response or not response.strip():
                logger.warning(f"{pid}: Empty response in attempt {i+1}")
                continue
                
            valid_responses.append(response)
            extracted = extract_answer_from_response(response)
            
            if extracted:
                answer = extracted
                logger.info(f"{pid}: Extracted answer '{answer}' from attempt {i+1}")
                break
            else:
                logger.warning(
                    f"{pid}: Could not extract answer from attempt {i+1} "
                    f"(length: {len(response)})"
                )
        
        if not answer and valid_responses:
            # Log first response for debugging
            logger.warning(
                f"{pid}: No valid answer after {len(valid_responses)} attempts. "
                f"First response preview: {valid_responses[0][:200]}..."
            )
        
        solutions[pid] = answer
        full_responses[pid] = valid_responses
    
    solved_count = sum(1 for a in solutions.values() if a)
    print(f"Successfully solved: {solved_count}/{len(problem_ids)}")
    
    # Save full responses for debugging
    write_json(full_responses, output_dir / "full_responses.json")
    
    return solutions


async def async_main(cfg: DictConfig) -> None:
    """Main entry point"""
    
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Setup LLM
    provider = Provider(cfg.model.provider)
    model = cfg.model.name
    print(f"Using model: {model}")
    print(f"Note: Using v2 solver with reasoning-friendly prompts")
    
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
    
    print(f"Loaded {len(problems)} problems")
    
    # Solve
    solutions = await solve_aime_simple(
        problems=problems,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.get('dry_run', False)
    )
    
    # Save solutions
    write_json(solutions, output_dir / "solutions.json")
    print(f"\nSolutions saved to: {output_dir}/solutions.json")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()

