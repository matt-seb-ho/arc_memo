"""
AIME thought process generation - adapter for ArcMemo's thought_process.py
Generates reasoning from problem + answer (simulates solver explaining their solution)
"""

import asyncio
import logging
from pathlib import Path

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.constants import HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.aime_math import load_aime_data
from concept_mem.utils import read_json, run_llm_job, write_json

logger = logging.getLogger(__name__)

DEFAULT_GEN_CFG = GenerationConfig(
    temperature=0.3,
    max_tokens=4096,
)

AIME_THOUGHT_PROCESS_TEMPLATE = """You are a mathematics expert who has just solved an AIME problem. Your task is to explain your reasoning process - how you approached the problem, what insights led to the solution, and what techniques you used.

### Your Problem
{problem}

### Your Answer
{answer}

### Instruction
Explain your thought process for solving this problem.
- Focus on key insights and reasoning steps
- Explain why you chose this approach
- Mention what you tried or considered
- Organize as a coherent narrative (not code)

Format: Write a clear explanation of your solution approach and reasoning.
"""


def prepare_aime_prompts(
    problems: dict[str, dict],  # {pid: {'problem': text, 'answer': answer}}
) -> tuple[list[str], list[str]]:
    """Prepare prompts for AIME thought process generation"""
    
    prompts = []
    problem_ids = []
    
    for pid, data in problems.items():
        problem_ids.append(pid)
        
        prompt = AIME_THOUGHT_PROCESS_TEMPLATE.format(
            problem=data['problem'],
            answer=data['answer']
        ).strip()
        
        prompts.append(prompt)
    
    return prompts, problem_ids


async def generate_thought_processes(
    problems: dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig = DEFAULT_GEN_CFG,
    output_dir: Path | None = None,
    dry_run: bool = False
) -> dict[str, str]:
    """
    Generate thought processes for AIME solutions
    
    Args:
        problems: {pid: {'problem': text, 'answer': answer}}
    
    Returns:
        {problem_id: thought_process_text}
    """
    
    prompts, problem_ids = prepare_aime_prompts(problems)
    
    outputs = await run_llm_job(
        prompts=prompts,
        metadata=problem_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run
    )
    
    thought_processes = {pid: output for pid, output in zip(problem_ids, outputs)}
    
    return thought_processes


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
    
    # Load solutions (from aime_simple_solver.py output)
    # Check both cfg.abstraction.solutions_file and cfg.solutions_file
    if hasattr(cfg, 'abstraction') and hasattr(cfg.abstraction, 'solutions_file'):
        solutions_file = cfg.abstraction.solutions_file
    else:
        solutions_file = cfg.get('solutions_file', 'data/aime/o4_solutions.json')
    
    solutions = read_json(REPO_ROOT / solutions_file)
    
    # Load AIME problems
    problems_data = load_aime_data(split=cfg.data.split)
    
    # Combine
    problems = {}
    for pid, answer in solutions.items():
        if pid in problems_data:
            problems[pid] = {
                'problem': problems_data[pid].problem_text,
                'answer': answer
            }
    
    print(f"Generating thought processes for {len(problems)} problems")
    
    # Generate thought processes
    thought_processes = await generate_thought_processes(
        problems=problems,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.get('dry_run', False)
    )
    
    # Save
    write_json(thought_processes, output_dir / "thought_processes.json")
    print(f"Thought processes saved to: {output_dir}/thought_processes.json")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()

