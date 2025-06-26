import dataclasses
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from llmplus import GenerationConfig, LLMClient

from concept_mem.concept_memory import ConceptMemory
from concept_mem.evaluation.prompt_builder import PromptBuilder, PromptOptions
from concept_mem.evaluation.prompts import make_lcs_puzzle_solving_prompt
from concept_mem.evaluation.retry_policy import RetryPolicy
from concept_mem.evaluation.score_trace import score_traces
from concept_mem.evaluation.solution_trace_types import (
    PuzzleTrace,
    RefinementPass,
)
from concept_mem.selection.description.select import reselect_concepts
from concept_mem.selection.long_cot import select_concepts_using_long_cot
from concept_mem.types import Problem
from concept_mem.utils import read_json, write_json

logger = logging.getLogger(__name__)


DEFAULT_LONG_COT_SELECTION_MEMORY_CONCEPT_FILE = Path(
    "data/anno_scheme/concept_memory_v5_g41.json"
)
DEFAULT_LCS_ANNOTATION_FILE = Path("data/anno_scheme/anno_v5_g41_merged.json")


@dataclass
class _Task:
    """Internal: one prompt we will send for generation/retry."""

    prompt_text: str
    puzzle_id: str
    variant_id: str
    path_id: int  # existing path or 0‑based index when n>1 generating
    pass_id: int


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

        self.traces: dict[str, PuzzleTrace] = {}
        self.initial_prompts: dict[tuple[str, str], str] = {}  # (puzzle_id, variant_id)

    async def run(self, problems: dict[str, Problem]) -> None:
        for pass_num in range(1, self.retry.max_passes + 1):
            logger.info(f"running pass {pass_num} of {self.retry.max_passes}")
            await self._run_iteration(problems, pass_num)

    async def _create_initial_tasks(self, problems: dict[str, Problem]) -> list[_Task]:
        tasks = []

        if self.lcs_cfg.use_lcs:
            if self.lcs_cfg.selected_concepts_file:
                lcs_initial_res = read_json(self.lcs_cfg.selected_concepts_file)
            else:
                lcs_initial_res = await select_concepts_using_long_cot(
                    puzzles=problems,
                    concept_mem=self.concept_mem,
                    llm_client=self.llm,
                    model=self.model,
                    gen_cfg=self.gen_cfg,
                    output_dir=self.output_dir,
                    dry_run=self.dry_run,
                )

        for puzzle_id, problem in problems.items():
            self.traces[puzzle_id] = PuzzleTrace(puzzle_id=puzzle_id)
            if self.lcs_cfg.use_lcs:
                if self.dry_run:
                    backup_lcs_entry = {"concepts": ["connection"], "notes": "testing"}
                else:
                    backup_lcs_entry = {}
                lcs_entry = lcs_initial_res.get(puzzle_id, backup_lcs_entry)
                prompts = self.build_initial_lcs_prompts(
                    puzzle=problem,
                    lcs_entry=lcs_entry,
                )
            else:
                prompts = self.prompt_builder.build_initial_prompts(
                    problem=problem,
                )
            for variant_id, prompt in prompts.items():
                self.initial_prompts[(puzzle_id, variant_id)] = prompt
                tasks.append(
                    _Task(
                        prompt_text=prompt,
                        puzzle_id=puzzle_id,
                        variant_id=variant_id,
                        path_id=-1,
                        pass_id=1,
                    )
                )
        return tasks

    async def _create_retry_tasks(self, pass_num: int) -> list[_Task]:
        tasks = []
        puzzles_to_retry = []
        for puzzle_id, trace in self.traces.items():
            for prompt_id, variant in trace.prompt_variants.items():
                for path_id, path in variant.paths.items():
                    last_pass = path.last()
                    if not self.retry.needs_retry(last_pass):
                        continue
                    puzzles_to_retry.append((puzzle_id, prompt_id, path_id, path))

        # reselect concepts
        if self.retry.reselect_concepts and len(self.lessons):
            pzids = []
            completions = {}
            for puzzle_id, _, _, path in puzzles_to_retry:
                if path.last().completion:
                    pzids.append(puzzle_id)
                    completions[puzzle_id] = path.last().completion
            reselect_output_dir = self.output_dir / f"pass_{pass_num}/reselect"
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

        for puzzle_id, prompt_id, path_id, path in puzzles_to_retry:
            base_prompt = self.initial_prompts[(puzzle_id, prompt_id)]
            retry_prompt = self.prompt_builder.build_retry_prompt(
                initial_prompt=base_prompt,
                solution_path=path,
                new_concepts=new_concepts.get(puzzle_id, None),
            )
            tasks.append(
                _Task(
                    prompt_text=retry_prompt,
                    puzzle_id=puzzle_id,
                    variant_id=prompt_id,
                    path_id=path_id,
                    pass_id=pass_num,
                )
            )
        return tasks

    async def _run_iteration(
        self,
        problems: dict[str, Problem],
        pass_num: int = 1,
    ) -> None:
        if pass_num == 1:
            tasks = await self._create_initial_tasks(problems)
        else:
            tasks = await self._create_retry_tasks(pass_num)
        await self._execute_tasks(tasks, problems, pass_num)

    async def _execute_tasks(
        self, tasks: list[_Task], problems: dict[str, Problem], pass_num: int = 1
    ) -> None:
        """send a batch to LLM and register completions into traces."""
        # prepare pass directory
        pass_dir = self.output_dir / f"pass_{tasks[0].pass_id}"
        pass_dir.mkdir(parents=True, exist_ok=True)

        # write prompt artifact
        prompts = [t.prompt_text for t in tasks]
        prompt_artifact = [asdict(t) for t in tasks]
        write_json(prompt_artifact, pass_dir / "prompts.json")

        # gather completions
        if tasks[0].pass_id == 1:
            gen_cfg = self.gen_cfg
        else:
            gen_cfg = self.retry_gen_cfg
        if self.dry_run:
            completions_batch = [["# dry‑run placeholder"] * gen_cfg.n for _ in tasks]
        else:
            completions_batch = await self.llm.async_batch_generate(
                prompts=prompts,
                model=self.model,
                gen_cfg=gen_cfg,
                progress_file=self.output_dir / "gen_progress.json",
            )

        # save token usage
        write_json(
            self.llm.get_token_usage_dict(),
            pass_dir / "token_usage.json",
        )

        # register and score completions
        for task, completions in zip(tasks, completions_batch):
            # determine / create SolutionPath(s)
            puzzle_trace = self.traces[task.puzzle_id]
            variant = puzzle_trace.get_variant(task.variant_id)
            if task.path_id == -1:
                # initial batch: build new paths enumerated by index
                for idx, completion in enumerate(completions):
                    path = variant.get_path(idx)
                    pass_ = RefinementPass(
                        pass_id=1,
                        path_id=idx,
                        variant_id=task.variant_id,
                        completion=completion,
                    )
                    pass_.score(problems[task.puzzle_id])
                    path.passes.append(pass_)
            else:
                # retry: single completion appended to existing path
                completion = completions[0] if completions else ""
                path = variant.get_path(task.path_id)
                pass_ = RefinementPass(
                    pass_id=len(path.passes),
                    path_id=task.path_id,
                    variant_id=task.variant_id,
                    completion=completion,
                )
                pass_.score(problems[task.puzzle_id])
                path.passes.append(pass_)

        # create scoring artifact
        score_traces(
            traces=self.traces,
            pass_num=pass_num,
            pass_dir=pass_dir,
        )

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
