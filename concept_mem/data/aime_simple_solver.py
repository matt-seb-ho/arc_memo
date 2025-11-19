"""
Simple AIME solver - just get numerical answers
Reasoning is generated separately using thought_process.py (reused from ArcMemo)
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

AIME_SYSTEM_PROMPT = "Return only the final numerical answer (an integer between 0 and 999). No explanation needed."

AIME_SOLVE_PROMPT = """
Solve this AIME problem and return only the numerical answer.

Problem: {problem}

Answer: """


async def solve_aime_simple(
    problems: dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path,
    dry_run: bool = False
) -> dict[str, str]:
    """
    Solve AIME problems - get just numerical answers
    
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
    
    # Extract answers (with retry logic for empty responses)
    solutions = {}
    for pid, response_list in zip(problem_ids, responses):
        if not response_list:
            logger.warning(f"No response for {pid}")
            solutions[pid] = ""
            continue
        
        # Try each response in order (n=3 means 3 attempts)
        answer = ""
        for attempt in response_list:
            if attempt and attempt.strip():
                # Extract number (in case model added extra text)
                number_match = re.search(r'\d+', attempt)
                if number_match:
                    answer = number_match.group(0)
                    break
        
        if not answer:
            logger.warning(f"No valid answer for {pid} after {len(response_list)} attempts")
        
        solutions[pid] = answer
    
    solved_count = sum(1 for a in solutions.values() if a)
    print(f"Successfully solved: {solved_count}/{len(problem_ids)}")
    return solutions


async def async_main(cfg: DictConfig) -> None:
    """Main entry point"""
    
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    # Setup LLM
    provider = Provider(cfg.model.provider)
    model = cfg.model.name
    print(f"Using model: {model}")
    
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

