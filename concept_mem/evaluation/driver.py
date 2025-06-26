import asyncio
import dataclasses
import logging
import random
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import hydra
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig

from concept_mem.concept_memory import ConceptMemory
from concept_mem.constants import (
    DOTENV_PATH,
    HYRDA_CONFIG_PATH,
    REPO_ROOT,
)
from concept_mem.evaluation.prompt_builder import PromptBuilder, PromptOptions
from concept_mem.evaluation.prompts import (
    SYSTEM_PROMPTS,
    make_lcs_puzzle_solving_prompt,
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
from concept_mem.selection.description.select import reselect_concepts
from concept_mem.selection.long_cot import select_concepts_using_long_cot
from concept_mem.types import Problem
from concept_mem.utils import load_arc_data, read_json, run_llm_job, write_json

logger = logging.getLogger(__name__)


DEFAULT_LONG_COT_SELECTION_MEMORY_CONCEPT_FILE = Path(
    "data/anno_scheme/concept_memory_v5_g41.json"
)
DEFAULT_LCS_ANNOTATION_FILE = Path("data/anno_scheme/anno_v5_g41_merged.json")


AttemptTag = namedtuple(
    "AttemptTag", ["puzzle_id", "branch_id", "thread_id", "step_idx"]
)


@dataclass
class LongCoTSelectionConfig:
    use_lcs: bool = False
    concept_memory_file: Path = DEFAULT_LONG_COT_SELECTION_MEMORY_CONCEPT_FILE
    annotation_file: Path = DEFAULT_LCS_ANNOTATION_FILE
    detailed_examples: int = 3
    max_examples: int = 5
    pass_initial_analysis_notes: bool | None = None
    # if None, ensembles both with and without init notes
    selected_concepts_file: Path | None = None


class EvaluationRunner:
    def __init__(
        self,
        llm: LLMClient,
        model: str,
        prompt_options: PromptOptions,
        retry_policy: RetryPolicy,
        gen_cfg: GenerationConfig,
        long_cot_sel_cfg: LongCoTSelectionConfig,
        output_dir: Path,
        dry_run: bool = False,
    ) -> None:
        self.llm = llm
        self.model = model
        self.prompt_builder = PromptBuilder(
            prompt_options=prompt_options,
            retry_policy=retry_policy,
        )
        self.retry = retry_policy
        self.gen_cfg = gen_cfg
        self.retry_gen_cfg = dataclasses.replace(gen_cfg, n=1)
        self.lcs_cfg = long_cot_sel_cfg
        if self.lcs_cfg.use_lcs:
            self.concept_mem = ConceptMemory()
            self.concept_mem.load_from_file(self.lcs_cfg.concept_memory_file)
            self.annotations = read_json(self.lcs_cfg.annotation_file)
        self.output_dir = output_dir
        if retry_policy.lesson_file:
            self.lessons = read_json(retry_policy.lesson_file)
        else:
            self.lessons = {}
        self.dry_run = dry_run

        self.trees: dict[str, SolutionTree] = {}
        self.initial_prompts: dict[tuple[str, str], str] = {}  # (puzzle_id, variant_id)

    async def run(self, problems: dict[str, Problem]) -> None:
        for i in range(1, self.retry.max_passes + 1):
            logger.info(f"running iteration {i} of {self.retry.max_passes}")
            await self._run_iteration(problems, iteration=i)

    async def initial_solve_step(
        self, problems: dict[str, Problem], output_dir: Path
    ) -> None:
        # run long-cot concept selection if needed
        if self.lcs_cfg.use_lcs:
            if self.lcs_cfg.selected_concepts_file:
                lcs_initial_res = read_json(self.lcs_cfg.selected_concepts_file)
            else:
                lcs_output_dir = self.output_dir / "lcs_initial_selection"
                lcs_initial_res = await select_concepts_using_long_cot(
                    puzzles=problems,
                    concept_mem=self.concept_mem,
                    llm_client=self.llm,
                    model=self.model,
                    gen_cfg=self.gen_cfg,
                    output_dir=lcs_output_dir,
                    dry_run=self.dry_run,
                )

        # prepare prompts and metadata
        metadata: list[tuple[str, str]] = []
        prompts: list[str] = []
        for puzzle_id, problem in problems.items():
            self.trees[puzzle_id] = SolutionTree(puzzle_id=puzzle_id)
            if self.lcs_cfg.use_lcs:
                if self.dry_run:
                    backup_lcs_entry = {"concepts": ["connection"], "notes": "testing"}
                else:
                    backup_lcs_entry = {}
                lcs_entry = lcs_initial_res.get(puzzle_id, backup_lcs_entry)
                puzzle_prompts = self.build_initial_lcs_prompts(
                    puzzle=problem,
                    lcs_entry=lcs_entry,
                )
            else:
                puzzle_prompts = self.prompt_builder.build_initial_prompts(
                    problem=problem,
                )
            for branch_id, prompt in puzzle_prompts.items():
                self.initial_prompts[(puzzle_id, branch_id)] = prompt
                metadata.append((puzzle_id, branch_id))
                prompts.append(prompt)

        # run first step
        completions = await run_llm_job(
            prompts=prompts,
            metadata=metadata,
            llm_client=self.llm,
            model=self.model,
            gen_cfg=self.gen_cfg,
            output_dir=output_dir,
            dry_run=self.dry_run,
        )

        for md, puzzle_completions in zip(metadata, completions):
            puzzle_id, branch_id = md
            tree = self.trees[puzzle_id]
            branch = tree.get_or_create_branch(branch_id)
            for idx, completion in enumerate(puzzle_completions):
                thread_id = str(idx)
                thread = branch.get_or_create_thread(thread_id)
                step = SolutionStep(
                    step_idx=0,
                    thread_id=thread_id,
                    prompt_id=branch_id,
                    puzzle_id=puzzle_id,
                    completion=completion,
                )
                score_problem_attempt(problems[puzzle_id], step)
                thread.steps.append(step)

    async def retry_solving_step(
        self, problems: dict[str, Problem], output_dir: Path
    ) -> None:
        # identify which puzzles need retrying
        puzzles_to_retry: list[AttemptTag] = []
        for puzzle_id, tree in self.trees.items():
            for branch_id, branch in tree.prompt_branches.items():
                for thread_id, thread in branch.threads.items():
                    if len(thread.steps) == 0:
                        continue
                    step = thread.steps[-1]
                    if not self.retry.needs_retry(step):
                        continue
                    puzzles_to_retry.append(
                        AttemptTag(
                            puzzle_id=puzzle_id,
                            branch_id=branch_id,
                            thread_id=thread_id,
                            step_idx=len(thread.steps),
                        )
                    )

        # reselect concepts
        if self.retry.reselect_concepts and len(self.lessons):
            pzids = []
            completions = {}
            for puzzle_id, branch_id, thread_id, step_idx in puzzles_to_retry:
                step = (
                    self.trees[puzzle_id]
                    .prompt_branches[branch_id]
                    .threads[thread_id]
                    .steps[step_idx - 1]
                )
                if step.completion:
                    pzids.append(puzzle_id)
                    completions[puzzle_id] = step.completion
            reselect_output_dir = output_dir / "reselect_concepts"
            reselect_output_dir.mkdir(parents=True, exist_ok=True)
            new_concepts, _ = await reselect_concepts(
                puzzles=pzids,
                completions=completions,
                lessons=self.lessons,
                llm_client=self.llm,
                model=self.retry.reselect_model,
                gen_cfg=self.retry.reselect_gen_cfg,
                top_k=self.retry.reselect_k,
                output_dir=reselect_output_dir,
            )
        else:
            new_concepts = {}

        # create new prompts
        prompts: list[str] = []
        for puzzle_id, branch_id, thread_id, step_idx in puzzles_to_retry:
            thread = self.trees[puzzle_id].prompt_branches[branch_id].threads[thread_id]
            base_prompt = self.initial_prompts[(puzzle_id, branch_id)]
            retry_prompt = self.prompt_builder.build_retry_prompt(
                initial_prompt=base_prompt,
                solution_thread=thread,
                new_concepts=new_concepts.get(puzzle_id, None),
            )
            prompts.append(retry_prompt)

        # run retry generation
        completions = await run_llm_job(
            prompts=prompts,
            metadata=puzzles_to_retry,
            llm_client=self.llm,
            model=self.model,
            gen_cfg=self.retry_gen_cfg,
            output_dir=output_dir,
            dry_run=self.dry_run,
        )

        # register to solution tree and re-score
        for tags, completion_batch in zip(puzzles_to_retry, completions):
            puzzle_id, branch_id, thread_id, step_idx = tags
            tree = self.trees[puzzle_id]
            branch = tree.get_or_create_branch(branch_id)
            thread = branch.get_or_create_thread(thread_id)
            # retry only uses one completion per prompt
            completion = completion_batch[0] if completion_batch else ""
            step = SolutionStep(
                step_idx=step_idx,
                thread_id=thread_id,
                prompt_id=branch_id,
                puzzle_id=puzzle_id,
                completion=completion,
            )
            score_problem_attempt(problems[puzzle_id], step)
            thread.steps.append(step)

    async def _run_iteration(
        self,
        problems: dict[str, Problem],
        iteration: int = 1,
    ) -> None:
        output_dir = self.output_dir / f"iteration_{iteration}"
        output_dir.mkdir(parents=True, exist_ok=True)
        if iteration == 1:
            await self.initial_solve_step(problems, output_dir)
        else:
            await self.retry_solving_step(problems, output_dir)
        self.compute_and_report_scores(iteration, output_dir)

    def build_initial_lcs_prompts(
        self,
        lcs_entry: dict,
        puzzle: Problem,
    ) -> dict[str, str]:
        # prompts
        prompt = make_lcs_puzzle_solving_prompt(
            puzzle=puzzle,
            selected_concepts=lcs_entry.get("concepts", []),
            notes=lcs_entry.get("notes", ""),
            concept_memory=self.concept_mem,
            solution_annotations=self.annotations,
            detailed_examples=self.lcs_cfg.detailed_examples,
            max_examples=self.lcs_cfg.max_examples,
        )
        sans_notes_prompt = make_lcs_puzzle_solving_prompt(
            puzzle=puzzle,
            selected_concepts=lcs_entry.get("concepts", []),
            notes="",
            concept_memory=self.concept_mem,
            solution_annotations=self.annotations,
            detailed_examples=self.lcs_cfg.detailed_examples,
            max_examples=self.lcs_cfg.max_examples,
        )
        prompts = {
            "with_notes": prompt,
            "sans_notes": sans_notes_prompt,
        }
        if self.lcs_cfg.pass_initial_analysis_notes:
            prompts.pop("sans_notes")
        elif self.lcs_cfg.pass_initial_analysis_notes is not None:
            prompts.pop("with_notes")
        return prompts

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
    lcs_cfg = hydra.utils.instantiate(cfg.long_cot_selection)

    # initialize and run
    eval_runner = EvaluationRunner(
        llm=llm_client,
        model=cfg.model.name,
        prompt_options=prompt_options,
        retry_policy=retry_policy,
        gen_cfg=gen_cfg,
        long_cot_sel_cfg=lcs_cfg,
        output_dir=output_dir,
        dry_run=cfg.dry_run,
    )
    await eval_runner.run(problems=problems)
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
