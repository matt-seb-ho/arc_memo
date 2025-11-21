import asyncio
import logging
from pathlib import Path
from typing import Any

import hydra
import yaml
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

# ARC-specific prompts (grid puzzles)
from concept_mem.abstraction.analysis_concept_prompts import (
    EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE,
    EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE_RETRIEVAL,
    EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE,
    EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE_RETRIEVAL,
    EXTRACT_LESSON_FROM_TRACE_ZS_TEMPLATE,
    LESSON_FROM_PUZZLE_EXAMPLE_TEMPLATE,
)

# AIME-specific prompts (mathematical reasoning)
from concept_mem.abstraction.aime_concept_prompts import (
    EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT,
    EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT_UNCERTAIN,
    LESSON_FROM_REFLECTION_EXAMPLE_TEMPLATE,
    LESSON_FROM_TRACE_EXAMPLE_TEMPLATE,
)

# from detective.abstraction.retriever import ProblemRetriever
from concept_mem.abstraction.thought_process import get_soluton_summary
from concept_mem.constants import HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.evaluation.prompts import format_puzzle_for_prompt
from concept_mem.utils import (
    extract_yaml_block,
    read_json,
    run_llm_job,
    write_json,
)

logger = logging.getLogger(__name__)

DEFAULT_EXAMPLE_UIDS = []
DEFAULT_EXAMPLE_FILE = Path("")
DEFAULT_CONCEPT_ABSTRACTION_GEN_CFG = GenerationConfig(
    temperature=0.3,
    max_tokens=1024,
)


async def extract_lessons(
    problems: dict[str, Problem],
    solutions: dict[str, str],
    thought_processes: dict[str, str] | None,
    example_thought_processes: dict[str, str] | None,
    fixed_examples: dict[str, list[dict]] | None,
    retrieved_examples: dict[str, dict[str, list[dict]]] | None,
    llm_client: LLMClient,
    model: str = "gpt-4o",
    gen_cfg: GenerationConfig = DEFAULT_CONCEPT_ABSTRACTION_GEN_CFG,
    output_dir: Path = REPO_ROOT / "data/lessons",
    use_barc_solution: bool = True,
    domain_template: str = "arc",
    dry_run: bool = False,
) -> tuple[dict[str, list[dict]], dict]:
    """Return lesssons and token usage"""
    problem_ids = []
    prompts = []
    for problem_id, problem in problems.items():
        # solution = get_compressed_solution(problem)
        solution = _get_puzzle_solution(
            puzzle_id=problem_id,
            problems=problems,
            solutions=solutions,
            use_barc_solution=use_barc_solution,
        )
        if not solution:
            logger.warning(
                f"No solution found for puzzle {problem_id}. Skipping lesson extraction."
            )
            continue
        thought_proc = thought_processes.get(problem_id) if thought_processes else None
        if thought_processes and thought_proc is None:
            logger.warning(f"Problem {problem_id} not found in thought_processes dict")
        if problem_id == list(problems.keys())[0]:  # Log first problem for debugging
            logger.info(f"First problem: {problem_id}, thought_proc is None: {thought_proc is None}, domain_template: {domain_template}")
        rxs_for_puzzle = (
            retrieved_examples.get(problem_id, None) if retrieved_examples else None
        )
        prompt = build_abstraction_prompt(
            problem=problem,
            solution=solution,
            thought_process=thought_proc,
            fixed_examples=fixed_examples,
            retrieved_examples=rxs_for_puzzle,
            example_thought_processes=example_thought_processes,
            domain_template=domain_template,
        )
        problem_ids.append(problem_id)
        prompts.append(prompt)
    res = await run_llm_job(
        prompts=prompts,
        metadata=problem_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    token_usage_dict = llm_client.get_token_usage_dict()
    lessons = parse_lessons(
        problem_ids=problem_ids,
        model_outputs=res,
    )
    write_json(lessons, output_dir / "lessons.json")
    return lessons, token_usage_dict


def build_abstraction_prompt(
    problem: Problem,
    solution: str,
    thought_process: str | None = None,
    fixed_examples: dict | None = None,
    retrieved_examples: dict | None = None,
    example_thought_processes: dict[str, str] | None = None,
    domain_template: str = "arc",
) -> str:
    """
    Build abstraction prompt based on domain template.
    
    Args:
        domain_template: Template style - "arc", "aime", "aime_strict", "gpqa" (future)
    """
    if thought_process is None:
        # puzzle, solution -> lesson
        puzzle = format_puzzle_for_prompt(
            problem=problem,
            include_dim=True,
            include_test=False,
        )
        if retrieved_examples is not None:
            formatted_examples = format_lesson_examples(retrieved_examples)
            prompt = EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE_RETRIEVAL.format(
                examples=formatted_examples,
                puzzle=puzzle,
                solution=solution,
            )
        else:
            assert fixed_examples is not None, "0S puzzle->lesson(s) is not supported."
            formatted_examples = format_lesson_examples(fixed_examples)
            prompt = EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE.format(
                examples=formatted_examples,
                puzzle=puzzle,
                solution=solution,
            )
    else:
        # [previous step:] puzzle, solution -> thought process
        # thought process, solution -> lesson(s)
        
        # Select template based on domain
        if domain_template in ["aime_strict", "aime_strict_uncertain"]:
            # AIME-specific templates (no ARC/grid references)
            # Only support the two active templates used by current pipelines
            if fixed_examples is None:
                raise ValueError(
                    f"AIME domain templates require few-shot examples. "
                    f"Please provide examples in config (domain_template={domain_template})"
                )
            
            include_solution_in_examples = domain_template != "aime_strict_uncertain"
            formatted_examples = format_lesson_examples(
                fixed_examples,
                thought_processes=example_thought_processes,
                include_solution_in_examples=include_solution_in_examples,
            )
            
            if domain_template == "aime_strict":
                # Label-Guided pipeline: verified correct solutions
                prompt = EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT.format(
                    examples=formatted_examples,
                    solution=solution,
                    thought_process=thought_process,
                )
            else:  # "aime_strict_uncertain"
                # Self-Reflective pipeline: uncertain reflections
                prompt = EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT_UNCERTAIN.format(
                    examples=formatted_examples,
                    thought_process=thought_process,
                )
        elif domain_template == "gpqa":
            # Future: GPQA-specific templates
            raise NotImplementedError("GPQA templates not yet implemented")
        else:
            # ARC-specific templates (original)
            if retrieved_examples is not None:
                # few-shot ICL using retrieved examples
                formatted_examples = format_lesson_examples(
                    formatted_examples,
                    thought_processes=example_thought_processes,
                )
                prompt = EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE_RETRIEVAL.format(
                    examples=formatted_examples,
                    solution=solution,
                    thought_process=thought_process,
                )
            elif fixed_examples is not None:
                # few-shot ICL using fixed examples
                formatted_examples = format_lesson_examples(
                    fixed_examples,
                    thought_processes=example_thought_processes,
                )
                prompt = EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE.format(
                    examples=formatted_examples,
                    solution=solution,
                    thought_process=thought_process,
                )
            else:
                # zero-shot abstraction (no examples)
                prompt = EXTRACT_LESSON_FROM_TRACE_ZS_TEMPLATE.format(
                    solution=solution,
                    thought_process=thought_process,
                )
    return prompt


def parse_lessons(
    problem_ids: list[str],
    model_outputs: list[list[str]],
) -> dict[str, list[dict]]:
    """Parse the model outputs into a dict of lessons."""
    lessons = {}
    for problem_id, model_output in zip(problem_ids, model_outputs):
        try:
            yaml_block = extract_yaml_block(model_output[0])
            # Handle LaTeX math in YAML: double backslashes to escape them for YAML
            # YAML interprets \c, \a, etc. as invalid escape sequences
            # We need \\ to represent a literal backslash
            yaml_block_safe = yaml_block.replace('\\', '\\\\') if yaml_block else yaml_block
            lesson_list = yaml.safe_load(yaml_block_safe)
        except Exception as e:
            logger.error(
                f"Error extracting lesson for problem {problem_id}: {e}. Model output: {model_output}"
            )
            lesson_list = []
        if lesson_list:
            # Restore single backslashes in the parsed data (for LaTeX)
            lesson_list = _restore_latex_backslashes(lesson_list)
            lessons[problem_id] = lesson_list
    return lessons


def _restore_latex_backslashes(data):
    """Recursively restore single backslashes that were doubled for YAML parsing."""
    if isinstance(data, dict):
        return {k: _restore_latex_backslashes(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_restore_latex_backslashes(item) for item in data]
    elif isinstance(data, str):
        return data.replace('\\\\', '\\')
    else:
        return data


def retrieve_examples(
    problems: dict[str, Problem],
    top_k: int,
    embed_model: str,
    cache_path: Path,
    init_with_barc_seeds: bool = False,
) -> dict[str, dict[str, list[dict]]]:
    raise NotImplementedError(
        "Retrieval of examples is not implemented yet. Please use fixed examples or thought processes."
    )
    # TODO: re-implement ProblemRetriever
    # retriever = ProblemRetriever(
    #     embed_model=embed_model,
    #     cache_path=cache_path,
    # )
    # # OPTIONAL: add all BARC seeds into the pool --> only need to do once (or use cache)
    # if init_with_barc_seeds:
    #     barc_seeds = load_arc_data("barc_seeds")
    #     uid_text_lst = [
    #         (uid, get_soluton_summary(problem)) for uid, problem in barc_seeds.items()
    #     ]
    #     retriever.encode_batch(uid_text_lst, include=True)

    # # Retrieve UIDs for each of the problems
    # retrieved_examples: dict[str, dict[str, list[dict]]] = {}
    # lesson_saved_path = REPO_ROOT / "data/lessons" / "lessons.json"
    # all_lessons = read_json(lesson_saved_path)

    # for problem_id, problem in problems.items():
    #     solution = get_soluton_summary(problem)
    #     closest_uids = retriever.find_closest(solution, top_k=top_k)

    #     # load solved problems' lessons given uid.
    #     retrieved_examples[problem_id] = {
    #         c_uid: all_lessons[c_uid] for c_uid in closest_uids
    #     }
    # return retrieved_examples


def _normalize_example_lessons(example: Any) -> list[dict]:
    """Ensure example lessons are in list-of-dict format."""
    if isinstance(example, dict):
        for key in ("concepts", "lessons", "examples"):
            if key in example and isinstance(example[key], list):
                example = example[key]
                break
        else:
            raise TypeError(
                "Lesson example dictionary must contain a list under one of "
                "'concepts', 'lessons', or 'examples'"
            )
    if not isinstance(example, list):
        raise TypeError(
            f"Lesson example must be a list of dicts, got {type(example)!r}"
        )

    normalized: list[dict] = []
    for entry in example:
        if not isinstance(entry, dict):
            logger.debug(
                "Skipping non-dict lesson entry in few-shot examples: %r", entry
            )
            continue
        if "situation" not in entry or "suggestion" not in entry:
            logger.debug(
                "Skipping lesson missing required keys: %s", entry.keys()
            )
            continue
        normalized.append(entry)
    return normalized


def format_lesson_as_yaml_block(example: Any) -> str:
    lessons = _normalize_example_lessons(example)
    components = ["```yaml"]
    for lesson in lessons:
        components.append(
            f'- situation: "{lesson["situation"]}"\n  suggestion: "{lesson["suggestion"]}"'
        )
    components.append("```")
    return "\n".join(components)


def format_lesson_examples(
    examples: dict[str, list[dict]],
    example_solutions: dict[str, str] | None = None,
    thought_processes: dict[str, str] | None = None,
    include_solution_in_examples: bool = True,
) -> str:
    use_thought_process = thought_processes is not None
    components = []
    for i, (puzzle_id, example) in enumerate(examples.items(), start=1):
        lessons = format_lesson_as_yaml_block(example)
        
        if use_thought_process:
            # Trace-based: only needs solution text + thought_process + lessons
            # No Problem objects needed - works for AIME!
            solution = example_solutions.get(puzzle_id, "") if example_solutions else ""
            thought_proc = thought_processes.get(puzzle_id, "")
            if include_solution_in_examples and solution.strip():
                lesson = LESSON_FROM_TRACE_EXAMPLE_TEMPLATE.format(
                    example_num=i,
                    solution=solution,
                    thought_process=thought_proc,
                    lessons=lessons,
                )
            else:
                lesson = LESSON_FROM_REFLECTION_EXAMPLE_TEMPLATE.format(
                    example_num=i,
                    thought_process=thought_proc,
                    lessons=lessons,
                )
        else:
            # Puzzle-based: needs Problem objects for formatting (ARC only)
            problem = Problem.from_puzzle_id(puzzle_id)
            if problem is None:
                continue
            formatted_puzzle = format_puzzle_for_prompt(
                problem=problem,
                include_dim=True,
                include_test=False,
            )
            solution = _get_puzzle_solution(
                puzzle_id=puzzle_id,
                problems=examples,
                solutions=example_solutions or {},
                use_barc_solution=True,
            )
            lesson = LESSON_FROM_PUZZLE_EXAMPLE_TEMPLATE.format(
                example_num=i,
                puzzle=formatted_puzzle,
                solution=solution,
                lessons=lessons,
            )
        components.append(lesson)
    return "\n".join(components)


def _get_puzzle_solution(
    puzzle_id: str,
    problems: dict[str, Problem],
    solutions: dict[str, str],
    use_barc_solution: bool = True,
) -> str | None:
    puzzle = problems[puzzle_id]
    barc_solution = get_soluton_summary(puzzle)
    if use_barc_solution and barc_solution is not None:
        return barc_solution
    return solutions.get(puzzle_id, None)


async def async_main(cfg: DictConfig) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # set up model related components
    provider = Provider(cfg.abstraction.model.provider)
    model = cfg.abstraction.model.name
    llm_client = LLMClient(
        provider=provider,
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=REPO_ROOT / ".env",
    )
    gen_cfg = hydra.utils.instantiate(cfg.abstraction.generation)

    # load problems and solutions
    problem_solutions = cfg.abstraction.problem_solutions
    if problem_solutions is None:
        problem_solutions = {
            uid: seed.code for uid, seed in load_arc_data("barc_seeds").items()
        }
    else:
        problem_solutions = read_json(problem_solutions)
    
    # Load Problem objects only when needed
    # For trace-based extraction with thought processes, Problem objects are not used
    if cfg.abstraction.thought_processes:
        # When we have thought processes, the Problem object is never used in the prompt
        # Create dummy objects to satisfy the function signature
        class DummyProblem:
            def __init__(self, uid):
                self.uid = uid
        problems = {uid: DummyProblem(uid) for uid in problem_solutions.keys()}
    else:
        # Original ARC path: load full Problem objects for puzzle formatting
        problems = {uid: Problem.from_puzzle_id(uid) for uid in problem_solutions.keys()}

    # load thought processes and examples
    if cfg.abstraction.thought_processes:
        thought_processes = read_json(cfg.abstraction.thought_processes)
        logger.info(f"Loaded {len(thought_processes)} thought processes from {cfg.abstraction.thought_processes}")
        # Use .get() for optional config fields
        if cfg.abstraction.get('example_thought_processes'):
            etp = read_json(cfg.abstraction.example_thought_processes)
        else:
            etp = thought_processes
    else:
        thought_processes = None
        logger.info("No thought_processes file specified in config")

    # load examples
    if cfg.abstraction.get('examples'):
        # Support both JSON and YAML example files
        example_file_path = Path(cfg.abstraction.example_file)
        if example_file_path.suffix in ['.yaml', '.yml']:
            from concept_mem.utils import read_yaml
            all_examples = read_yaml(REPO_ROOT / cfg.abstraction.example_file)
        else:
            all_examples = read_json(cfg.abstraction.example_file)
        examples = {}
        for uid in cfg.abstraction.examples:
            examples[uid] = all_examples[uid]
    else:
        examples = None

    # retrieve examples if requested
    if cfg.abstraction.get('retrieve_examples', False):
        retrieved_examples = retrieve_examples(
            problems=problems,
            top_k=cfg.abstraction.example_retrieval.top_k,
            embed_model=cfg.abstraction.example_retrieval.embed_model,
            cache_path=cfg.abstraction.example_retrieval.cache_path,
            init_with_barc_seeds=cfg.abstraction.example_retrieval.init_with_barc_seeds,
        )
    else:
        retrieved_examples = None

    # run lesson extraction
    # Handle nested abstraction config structure
    abstraction_cfg = cfg.abstraction.get('abstraction', cfg.abstraction)
    await extract_lessons(
        problems=problems,
        solutions=problem_solutions,
        thought_processes=thought_processes,
        example_thought_processes=etp,
        fixed_examples=examples,
        retrieved_examples=retrieved_examples,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        use_barc_solution=abstraction_cfg.get('use_barc_solution', False),
        domain_template=abstraction_cfg.get('domain_template', 'arc'),
        dry_run=cfg.dry_run,
    )
    logger.info(f"lesson abstraction complete. wrote to {output_dir}")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
