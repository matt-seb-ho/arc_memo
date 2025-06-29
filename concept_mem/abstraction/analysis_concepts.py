import asyncio
import logging
from pathlib import Path
from typing import Any

import hydra
import yaml
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.abstraction.analysis_concept_prompts import (
    EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE,
    EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE_RETRIEVAL,
    EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE,
    EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE_RETRIEVAL,
    EXTRACT_LESSON_FROM_TRACE_ZS_TEMPLATE,
    LESSON_FROM_PUZZLE_EXAMPLE_TEMPLATE,
    LESSON_FROM_TRACE_EXAMPLE_TEMPLATE,
)

# from detective.abstraction.retriever import ProblemRetriever
from concept_mem.abstraction.thought_process import get_soluton_summary
from concept_mem.constants import HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.evaluation.prompts import format_puzzle_for_prompt
from concept_mem.types import Problem
from concept_mem.utils import (
    extract_yaml_block,
    get_arc_problem_by_uid,
    load_arc_data,
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
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict]:
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
) -> str:
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
            lesson_list = yaml.safe_load(yaml_block)
        except Exception as e:
            logger.error(
                f"Error extracting lesson for problem {problem_id}: {e}. Model output: {model_output}"
            )
            lesson_list = []
        if lesson_list:
            lessons[problem_id] = lesson_list
    return lessons


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


def format_lesson_as_yaml_block(lessons: list[dict]) -> str:
    components = ["```yaml"]
    for lesson in lessons:
        components.append(
            f'- situation: "{lesson["situation"]}"\n  suggestion: "{lesson["suggestion"]}"'
        )
    components.append("```")
    return "\n".join(components)


def format_lesson_examples(
    examples: dict[str, list[dict]],
    thought_processes: dict[str, str] | None = None,
) -> str:
    use_thought_process = thought_processes is not None
    components = []
    for i, (puzzle_id, example) in enumerate(examples.items(), start=1):
        problem, _ = get_arc_problem_by_uid(puzzle_id)
        if problem is None:
            continue
        formatted_puzzle = format_puzzle_for_prompt(
            problem=problem,
            include_dim=True,
            include_test=False,
        )
        solution = get_soluton_summary(problem)
        lessons = format_lesson_as_yaml_block(example)
        if use_thought_process:
            lesson = LESSON_FROM_TRACE_EXAMPLE_TEMPLATE.format(
                example_num=i,
                solution=solution,
                thought_process=thought_processes[puzzle_id],
                lessons=lessons,
            )
        else:
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
    problems = {uid: get_arc_problem_by_uid(uid)[0] for uid in problem_solutions.keys()}

    # load thought processes and examples
    if cfg.abstraction.thought_processes:
        thought_processes = read_json(cfg.abstraction.thought_processes)
        if cfg.abstraction.example_thought_processes:
            etp = read_json(cfg.abstraction.example_thought_processes)
        else:
            etp = thought_processes
    else:
        thought_processes = None

    # load examples
    if cfg.abstraction.examples:
        all_examples = read_json(cfg.abstraction.example_file)
        examples = {}
        for uid in cfg.abstraction.examples:
            examples[uid] = all_examples[uid]
    else:
        examples = None

    # retrieve examples if requested
    if cfg.abstraction.retrieve_examples:
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
        use_barc_solution=cfg.abstraction.use_barc_solution,
        dry_run=cfg.dry_run,
    )
    logger.info(f"lesson abstraction complete. wrote to {output_dir}")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
