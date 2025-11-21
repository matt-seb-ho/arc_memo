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
{correct_answer}

### Instruction
Explain your thought process for solving this problem.
- Focus on key insights and reasoning steps
- Explain why you chose this approach
- Mention what you tried or considered
- Organize as a coherent narrative (not code)

Format: Write a clear explanation of your solution approach and reasoning.
"""

REFLECTIVE_THOUGHT_PROCESS_TEMPLATE = """You are a math expert helping to analyze solutions. For each problem, you will be given:
- The Problem.
- An initial Solution Attempt (with the answer given by the solver).
- The Correct Answer.

Your job: Explain **why the attempt was wrong and how to correctly solve the problem**. Identify any mistakes in the attempt, then provide a step-by-step corrected solution that leads to the correct answer.

**Example 1:**
Problem: A jar has 5 red and 3 blue marbles. If 2 marbles are drawn without replacement, what is the probability both are red?
Initial Attempt and Answer: "We treat it as independent draws: P(red, then red) = 5/8 × 5/8 = 25/64. So the answer is 25/64."
Correct Answer: 5/14.
**Reflection:** The attempt assumed the draws were independent with replacement, which was a mistake. After drawing one red, the total marbles and red count change. The correct calculation is P(red then red) = (5/8) × (4/7) = 20/56 = 5/14. The error was not accounting for the decreased total and red count on the second draw.

**Example 2:**
Problem: How many ways can you choose 2 people out of 5 people to form a team?
Initial Attempt and Answer: "Order might matter, so compute 5P2 = 5 × 4 = 20 ways."
Correct Answer: 10.
**Reflection:** The attempt counted each pair twice because order doesn’t matter in a combination. The correct approach is to use combinations: C(5,2) = (5 × 4)/2! = 10. The mistake was treating it as an ordered permutation instead of an unordered selection.

**Now your turn:**
Problem: {problem}
Initial Attempt and Answer: "{attempt}" (Incorrect)
Correct Answer: {correct_answer}
**Reflection:**"""

REFLECTIVE_THOUGHT_PROCESS_TEMPLATE_MISTAKE_ONLY = """You are a math expert reviewing your previous attempt at an AIME problem.

### Problem
{problem}

### Your Previous Attempt
- Provided answer: "{attempt}"
- The grader reported that this answer is incorrect.

### Instruction
- Diagnose why the previous reasoning/answer failed.
- Highlight faulty assumptions, overlooked constraints, or computation errors.
- Describe concrete checks or heuristics to avoid repeating this pitfall.
- Do **not** reveal the final correct answer; focus on the mistake analysis and preventative guidance.

**Reflection:**"""


def prepare_aime_prompts(
    problems: dict[str, dict],
    reflection_style: str,
) -> tuple[list[str], list[str]]:
    """Prepare prompts for AIME thought process generation."""
    
    prompts = []
    problem_ids = []
    
    for pid, data in problems.items():
        problem_ids.append(pid)
        
        if data["was_correct"]:
            prompt = AIME_THOUGHT_PROCESS_TEMPLATE.format(
                problem=data["problem"],
                correct_answer=data["correct_answer"],
            )
        else:
            attempt = data["solver_answer"] or "No answer provided."
            if reflection_style == "mistake_only":
                prompt = REFLECTIVE_THOUGHT_PROCESS_TEMPLATE_MISTAKE_ONLY.format(
                    problem=data["problem"],
                    attempt=attempt,
                )
            else:
                prompt = REFLECTIVE_THOUGHT_PROCESS_TEMPLATE.format(
                    problem=data["problem"],
                    attempt=attempt,
                    correct_answer=data["correct_answer"],
                )

        prompt = prompt.strip()
        prompts.append(prompt)
    
    return prompts, problem_ids


async def generate_thought_processes(
    problems: dict[str, dict],
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig = DEFAULT_GEN_CFG,
    output_dir: Path | None = None,
    dry_run: bool = False,
    reflection_style: str = "corrective",
) -> dict[str, str]:
    """
    Generate thought processes for AIME solutions
    
    Args:
        problems: {pid: {'problem': text, 'answer': answer}}
    
    Returns:
        {problem_id: thought_process_text}
    """
    
    prompts, problem_ids = prepare_aime_prompts(
        problems,
        reflection_style=reflection_style,
    )
    
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
            solver_answer = str(answer or "").strip()
            correct_answer = str(problems_data[pid].answer or "").strip()
            was_correct = bool(solver_answer) and solver_answer == correct_answer

            problems[pid] = {
                'problem': problems_data[pid].problem_text,
                'solver_answer': solver_answer,
                'correct_answer': correct_answer,
                'was_correct': was_correct,
            }

    # Determine reflection style (corrective vs mistake_only)
    reflection_style = "corrective"
    if hasattr(cfg, "abstraction"):
        reflection_style = getattr(cfg.abstraction, "reflection_style", "corrective")
    reflection_style = reflection_style.lower()
    if reflection_style not in {"corrective", "mistake_only"}:
        raise ValueError(
            f"Unsupported reflection_style '{reflection_style}'. "
            "Use 'corrective' or 'mistake_only'."
        )
    
    print(f"Generating thought processes for {len(problems)} problems")
    
    # Generate thought processes
    thought_processes = await generate_thought_processes(
        problems=problems,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.get('dry_run', False),
        reflection_style=reflection_style,
    )
    
    # Save
    write_json(thought_processes, output_dir / "thought_processes.json")
    print(f"Thought processes saved to: {output_dir}/thought_processes.json")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()

