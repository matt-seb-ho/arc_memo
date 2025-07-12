import asyncio
import logging
import random
from collections import defaultdict, namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.abstraction.analysis_concepts import extract_lessons
from concept_mem.abstraction.thought_process import thought_process
from concept_mem.constants import (
    DOTENV_PATH,
    HYRDA_CONFIG_PATH,
    REPO_ROOT,
)
from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.evaluation.prompt_builder import PromptOptions
from concept_mem.evaluation.prompts import (
    SYSTEM_PROMPTS,
    make_prompt,
    make_retry_prompt,
)
from concept_mem.evaluation.retry_policy import RetryPolicy
from concept_mem.evaluation.score_tree import (
    flatten_solution_trees,
    official_score,
    score_problem_attempt,
    strict_score,
)
from concept_mem.evaluation.solution_tree import (
    SolutionStep,
    SolutionTree,
)
from concept_mem.lesson_memory import LessonConceptMemory
from concept_mem.selection.description.select import reselect_concepts
from concept_mem.utils import read_json, run_llm_job, write_json

logger = logging.getLogger(__name__)


AttemptTag = namedtuple(
    "AttemptTag", ["puzzle_id", "branch_id", "thread_id", "step_idx"]
)


@dataclass
class ContinualEvaluationConfig:
    concept_memory: LessonConceptMemory
    descriptions: dict[str, str]
    prompt_options: PromptOptions
    retry_policy: RetryPolicy
    batch_size: int

    # model and generation config
    llm_client: LLMClient
    solve_model: str
    solve_gen_cfg: GenerationConfig

    # other stage configs
    select: DictConfig
    abstract: DictConfig

    # process
    output_dir: Path
    dry_run: bool = False


class ContinualEvaluationRunner:
    def __init__(self, cfg: ContinualEvaluationConfig) -> None:
        self.cfg = cfg
        self.concept_memory = cfg.concept_memory
        self.descriptions = cfg.descriptions
        self.llm_client = cfg.llm_client
        self.prompt_options = cfg.prompt_options
        self.retry_policy = cfg.retry_policy
        self.output_dir = cfg.output_dir
        self.dry_run = cfg.dry_run

        self.trees: dict[str, SolutionTree] = {}
        self.initial_prompts: dict[str, str] = {}
        self.abstracted_steps: set[str] = set()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.abstract.examples:
            all_abs_ex = read_json(cfg.abstract.example_file)
            self.abs_ex = {}
            for uid in cfg.abstract.examples:
                self.abs_ex[uid] = all_abs_ex[uid]
        else:
            self.abs_ex = None
        if cfg.abstract.example_thought_processes:
            self.etp = read_json(cfg.abstract.example_thought_processes)
        elif cfg.abstract.thought_processes:
            self.etp = read_json(cfg.abstract.thought_processes)

        self.select_gen_cfg = hydra.utils.instantiate(cfg.select.generation)
        self.abstract_gen_cfg = hydra.utils.instantiate(cfg.abstract.generation)

    async def run(self, problems: dict[str, Problem]) -> None:
        for i in range(1, self.retry_policy.max_passes + 1):
            logger.info(f"running iteration {i} of {self.retry_policy.max_passes}")
            await self.run_iteration(problems, iteration=i)

    async def run_iteration(
        self,
        problems: dict[str, Problem],
        iteration: int = 1,
    ) -> None:
        output_dir = self.output_dir / f"iter_{iteration}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if iteration == 1:
            await self.initial_solve_step(problems, self.cfg.batch_size, output_dir)
        else:
            await self.retry_solving_step(problems, self.cfg.batch_size, output_dir)
        self.compute_and_report_scores(iteration, output_dir)

    def compute_and_report_scores(
        self,
        iteration: int,
        output_dir: Path,
        save_tree: bool = True,
        save_csv: bool = False,
    ) -> None:
        df = flatten_solution_trees(self.trees)
        if save_tree:
            serializable_trees = {
                k: t.to_serializable_dict() for k, t in self.trees.items()
            }
            write_json(serializable_trees, output_dir / "solution_trees.json")
        if save_csv:
            df.to_csv(output_dir / "evaluation.csv")
        official_score_ = official_score(df, step_selection="all")
        strict_score_ = strict_score(df, include_train=True, step_selection="last")
        logger.info(
            f"Score Report (iter {iteration}):\n"
            f"  official: {official_score_}"
            f"  strict: {strict_score_}"
            f"  problems: {len(self.trees)}"
        )

    async def initial_solve_step(
        self,
        problems: dict[str, Problem],
        batch_size: int,
        output_dir: Path,
    ) -> None:
        # TODO
        problem_list = list(problems.values())
        batch_num = 0
        for i in range(0, len(problem_list), batch_size):
            batch = problem_list[i : i + batch_size]
            batch_output_dir = output_dir / f"batch_{batch_num}"
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            batch_num += 1
            logger.info(f"Processing batch {batch_num} with {len(batch)} problems.")
            await self._process_batch(
                problem_batch=batch,
                output_dir=batch_output_dir,
            )

    async def _process_batch(
        self,
        problem_batch: list[Problem],
        output_dir: Path,
    ) -> None:
        # prepare prompts
        batch_metadata = []
        batch_prompts = []
        # - select concepts, also initialize solution tree
        descriptions = []
        for problem in problem_batch:
            self.trees[problem.uid] = SolutionTree(puzzle_id=problem.uid)
            problem_id = problem.uid
            if problem_id not in self.descriptions:
                logger.warning(f"Description for problem {problem_id} not found.")
                continue
            descriptions.append(self.descriptions[problem_id])
        batch_select_dir = output_dir / "select_concepts"
        concept_hints = await self.concept_memory.select_concepts(
            puzzles=problem_batch,
            descriptions=descriptions,
            top_k=self.cfg.select.top_k,
            llm_client=self.llm_client,
            model=self.cfg.select.model.name,
            gen_cfg=self.select_gen_cfg,
            output_dir=batch_select_dir,
        )

        # - form the prompts
        for problem in problem_batch:
            batch_metadata.append(problem.uid)
            prompt = make_prompt(
                problem=problem,
                hint=concept_hints.get(problem.uid, None),
                description=self.descriptions.get(problem.uid, None),
                hint_template_key=self.prompt_options.hint_template_key,
                instruction_key=self.prompt_options.instruction_key,
                require_hint_citations=self.prompt_options.require_hint_citations,
            )
            self.initial_prompts[problem.uid] = prompt
            batch_prompts.append(prompt)

        # run the LLM job
        model_output = await run_llm_job(
            prompts=batch_prompts,
            metadata=batch_metadata,
            llm_client=self.llm_client,
            model=self.cfg.solve_model,
            gen_cfg=self.cfg.solve_gen_cfg,
            output_dir=output_dir,
            dry_run=self.dry_run,
        )

        # post process and score
        problem_solutions = {}
        for puzzle_id, puzzle, completions in zip(
            batch_metadata, problem_batch, model_output
        ):
            tree = self.trees[puzzle_id]
            branch_id = "0"
            branch = tree.get_or_create_branch(branch_id)
            for idx, completion in enumerate(completions):
                thread_id = str(idx)
                thread = branch.get_or_create_thread(thread_id)
                step = SolutionStep(
                    step_idx=0,
                    thread_id=thread_id,
                    branch_id=branch_id,
                    puzzle_id=puzzle_id,
                    completion=completion,
                )
                score_problem_attempt(problem=puzzle, attempt=step)
                thread.steps.append(step)
                if step.is_strictly_correct():
                    problem_solutions[puzzle_id] = completion

        # update memory
        logger.info(f"Solved {len(problem_solutions)} problems in batch.")
        await self.update_memory(
            problem_batch, problem_solutions, output_dir=output_dir
        )
        mem_save_file = output_dir / "concept_memory.json"
        self.concept_memory.save_to_file(mem_save_file)

    async def update_memory(
        self,
        problems: list[Problem],
        problem_solutions: dict[str, str],
        output_dir: Path,
    ) -> None:
        # generate post-hoc thought processes for the solutions
        thought_process_dir = output_dir / "thought_process"
        thought_processes, _ = await thought_process(
            problem_solutions=problem_solutions,
            llm_client=self.llm_client,
            model=self.cfg.abstract.model.name,
            output_dir=thought_process_dir,
            gen_cfg=self.abstract_gen_cfg,
        )

        # generate new lesson concepts
        lesson_dir = output_dir / "lesson_concepts"
        problem_dict = {p.uid: p for p in problems if p.uid in problem_solutions}
        new_lessons, _ = await extract_lessons(
            problems=problem_dict,
            solutions=problem_solutions,
            thought_processes=thought_processes,
            example_thought_processes=self.etp,
            fixed_examples=self.abs_ex,
            retrieved_examples=None,
            llm_client=self.llm_client,
            model=self.cfg.abstract.model.name,
            gen_cfg=self.abstract_gen_cfg,
            output_dir=lesson_dir,
        )
        try:
            total_new_lessons = sum(len(v) for v in new_lessons.values())
            logger.info(
                f"Extracted {total_new_lessons} new lessons from {len(new_lessons)} puzzles."
            )
            # add new lessons to the concept memory
            for puzzle_id, lesson_dicts in new_lessons.items():
                for lesson_dict in lesson_dicts:
                    situation = lesson_dict.get("situation", "")
                    suggestion = lesson_dict.get("suggestion", "")
                    if situation and suggestion:
                        self.concept_memory.add_lesson(
                            situation=situation,
                            suggestion=suggestion,
                            source=puzzle_id,
                        )
                    else:
                        logger.warning(
                            f"Skipping lesson for {puzzle_id} due to missing fields: {lesson_dict}"
                        )
        except Exception as e:
            logger.error(f"Error while extracting lessons: {e}")

    async def retry_solving_step(
        self,
        problems: dict[str, Problem],
        batch_size: int,
        output_dir: Path,
    ) -> None:
        # identify which puzzles need retrying
        puzzles_to_retry: list[AttemptTag] = []
        for puzzle_id, tree in self.trees.items():
            for branch_id, branch in tree.prompt_branches.items():
                for thread_id, thread in branch.threads.items():
                    if len(thread.steps) == 0:
                        continue
                    step = thread.steps[-1]
                    if not self.retry_policy.needs_retry(step):
                        continue
                    puzzles_to_retry.append(
                        AttemptTag(
                            puzzle_id=puzzle_id,
                            branch_id=branch_id,
                            thread_id=thread_id,
                            step_idx=len(thread.steps),
                        )
                    )
        if not puzzles_to_retry:
            logger.info("No puzzles to retry.")
            return

        batch_num = 0
        for i in range(0, len(puzzles_to_retry), batch_size):
            tag_batch = puzzles_to_retry[i : i + batch_size]
            problem_batch = [problems[tag.puzzle_id] for tag in tag_batch]
            batch_output_dir = output_dir / f"batch_{batch_num}"
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            batch_num += 1
            logger.info(
                f"Processing retry batch {batch_num} with {len(problem_batch)} problems."
            )
            await self._process_retry_batch(
                attempt_tags=tag_batch,
                problem_batch=problem_batch,
                output_dir=batch_output_dir,
            )

    async def _process_retry_batch(
        self,
        attempt_tags: list[AttemptTag],
        problem_batch: list[Problem],
        output_dir: Path,
    ) -> None:
        # prepare prompts
        batch_prompts: list[str] = []
        # - reselect concepts
        if self.retry_policy.reselect_concepts:
            pzids = []
            prev_completions = {}
            for puzzle_id, branch_id, thread_id, step_idx in attempt_tags:
                step = (
                    self.trees[puzzle_id]
                    .prompt_branches[branch_id]
                    .threads[thread_id]
                    .steps[step_idx - 1]
                )
                if step.completion:
                    pzids.append(puzzle_id)
                    prev_completions[puzzle_id] = step.completion
            reselect_output_dir = output_dir / "reselect_concepts"
            reselect_output_dir.mkdir(parents=True, exist_ok=True)
            # TODO: refactor reselect_concepts to match the updated mem fmt
            resel_all_lessons = defaultdict(list)
            for lesson in self.concept_memory.lessons:
                resel_all_lessons[lesson.source].append(asdict(lesson))
            if self.retry_policy.reselect_with_description:
                reselect_descriptions = self.descriptions
            else:
                reselect_descriptions = None
            if self.retry_policy.reselect_with_prev_attempt:
                new_concepts, _ = await reselect_concepts(
                    puzzles=pzids,
                    descriptions=reselect_descriptions,
                    completions=prev_completions,
                    lessons=resel_all_lessons,
                    llm_client=self.llm_client,
                    model=self.retry_policy.reselect_model,
                    gen_cfg=self.retry_policy.reselect_gen_cfg,
                    top_k=self.retry_policy.reselect_k,
                    output_dir=reselect_output_dir,
                )
            else:
                desc_list = [self.descriptions[pzid] for pzid in pzids]
                new_concepts = await self.concept_memory.select_concepts(
                    puzzles=problem_batch,
                    descriptions=desc_list,
                    top_k=self.cfg.select.top_k,
                    llm_client=self.llm_client,
                    model=self.cfg.select.model.name,
                    gen_cfg=self.select_gen_cfg,
                    output_dir=reselect_output_dir,
                )
        else:
            new_concepts = {}
        # - form the prompts
        for puzzle_id, branch_id, thread_id, step_idx in attempt_tags:
            thread = self.trees[puzzle_id].prompt_branches[branch_id].threads[thread_id]
            base_prompt = self.initial_prompts[puzzle_id]
            retry_prompt = make_retry_prompt(
                initial_prompt=base_prompt,
                solution_thread=thread,
                num_feedback_passes=self.retry_policy.num_feedback_passes,
                error_feedback=self.retry_policy.error_feedback,
                include_past_outcomes=self.retry_policy.include_past_outcomes,
                new_concepts=new_concepts.get(puzzle_id, None),
            )
            batch_prompts.append(retry_prompt)

        # run the LLM job
        model_output = await run_llm_job(
            prompts=batch_prompts,
            metadata=attempt_tags,
            llm_client=self.llm_client,
            model=self.cfg.solve_model,
            gen_cfg=self.cfg.solve_gen_cfg,
            output_dir=output_dir,
            dry_run=self.dry_run,
        )

        # register to solution tree and re-score
        problem_solutions: dict[str, str] = {}
        for tags, problem, completions in zip(
            attempt_tags, problem_batch, model_output
        ):
            puzzle_id, branch_id, thread_id, step_idx = tags
            tree = self.trees[puzzle_id]
            branch = tree.get_or_create_branch(branch_id)
            thread = branch.get_or_create_thread(thread_id)
            # retry only uses one completion per prompt
            completion = completions[0] if completions else ""
            step = SolutionStep(
                step_idx=step_idx,
                thread_id=thread_id,
                branch_id=branch_id,
                puzzle_id=puzzle_id,
                completion=completion,
            )
            score_problem_attempt(problem, step)
            thread.steps.append(step)
            if step.is_strictly_correct():
                problem_solutions[puzzle_id] = completion

        # update memory
        await self.update_memory(
            problems=problem_batch,
            problem_solutions=problem_solutions,
            output_dir=output_dir,
        )
        mem_save_file = output_dir / "concept_memory.json"
        self.concept_memory.save_to_file(mem_save_file)


def _load_problems(
    dataset: str,
    split: str,
    num_problems: int | None,
    problem_ids: list | str | None,
) -> dict[str, Problem]:
    """Load ARC‑AGI problems and subset them according to the config."""
    if dataset.lower() == "arc-agi":
        data = load_arc_data(split)
        if problem_ids is None:
            problem_ids = list(data.keys())
        elif isinstance(problem_ids, str):
            problem_ids = read_json(REPO_ROOT / problem_ids)
        elif num_problems and num_problems < len(problem_ids):
            problem_ids = random.sample(problem_ids, num_problems)
        return {pid: data[pid] for pid in problem_ids}
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


@hydra.main(version_base=None, config_path=HYRDA_CONFIG_PATH, config_name="default")
def main(cfg: DictConfig) -> None:
    asyncio.run(async_main(cfg))


async def async_main(cfg: DictConfig) -> None:
    # set up output_dir for file writing and logging
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # data and prompt preparation
    problems = _load_problems(
        dataset=cfg.data.dataset,
        split=cfg.data.split,
        num_problems=cfg.data.num_problems,
        problem_ids=cfg.data.problem_ids,
    )
    prompt_options = hydra.utils.instantiate(cfg.prompt)
    descriptions = read_json(cfg.selection.description_file)

    # llm generation preparation
    llm_client = LLMClient(
        provider=Provider(cfg.model.provider),
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    llm_client.system_prompt = SYSTEM_PROMPTS[cfg.prompt.system_prompt_key]
    logger.info(f"Using system prompt: {cfg.prompt.system_prompt_key}")
    retry_policy = hydra.utils.instantiate(cfg.puzzle_retry)
    gen_cfg = hydra.utils.instantiate(cfg.generation)

    # memory initialization
    concept_memory = LessonConceptMemory()
    abstraction_out = read_json(cfg.concept_mem_init_file)
    concept_memory.initialize_from_abstraction_output(
        abstraction_output=abstraction_out,
    )

    ce_config = ContinualEvaluationConfig(
        concept_memory=concept_memory,
        descriptions=descriptions,
        prompt_options=prompt_options,
        retry_policy=retry_policy,
        batch_size=cfg.continual_batch_size,
        llm_client=llm_client,
        solve_model=cfg.model.name,
        solve_gen_cfg=gen_cfg,
        select=cfg.selection,
        abstract=cfg.abstraction,
        output_dir=output_dir,
        dry_run=cfg.dry_run,
    )

    # initialize and run
    eval_runner = ContinualEvaluationRunner(ce_config)
    await eval_runner.run(problems=problems)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
