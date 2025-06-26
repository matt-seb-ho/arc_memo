import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from concept_mem.constants import REPO_ROOT
from concept_mem.evaluation.prompts import make_prompt, make_retry_prompt
from concept_mem.evaluation.solution_tree import SolutionThread
from concept_mem.evaluation.retry_policy import RetryPolicy
from concept_mem.types import Problem
from concept_mem.utils import read_json

logger = logging.getLogger(__name__)


@dataclass
class PromptOptions:
    # ICL examples
    include_examples: bool = False
    num_examples: int | None = None
    example_ids: list[str] | None = None

    # common lib
    include_common_lib: bool = False

    # hint
    include_hint: bool = False
    hint_file: str | None = None
    test_hint_file: str | None = None
    hint_template_key: Literal["min", "selected", "all_hints"] = "min"
    require_hint_citations: bool = False

    # concepts
    include_concepts: bool = False

    # problem data
    problem_data: str | None = None

    # instruction and system prompts
    # instruction_key: Literal["default", "concise", "cite"] = "default"
    # system_prompt_key: Literal["default", "concise"] = "default"
    instruction_key: str = "default"
    system_prompt_key: str = "default"

    # file override
    file: str | None = None


class PromptBuilder:
    """
    Routes prompt options and retry policy configs to the stateless prompt formatting functions.
    """

    def __init__(self, prompt_options: PromptOptions, retry_policy: RetryPolicy):
        self.prompt_options = prompt_options
        self.retry_policy = retry_policy

        # initialize problem data if included
        if self.prompt_options.problem_data:
            if self.prompt_options.problem_data.startswith("/"):
                data_path = Path(self.prompt_options.problem_data)
            else:
                data_path = REPO_ROOT / self.prompt_options.problem_data
            self.problem_data = read_json(data_path)
        else:
            self.problem_data = {}

    def build_initial_prompts(
        self,
        problem: Problem,
        **kwargs,
    ) -> dict[str, str]:
        """
        Build the initial prompt for a given problem and variant.

        currently unsupported args from make_prompt:
        - concept (ground truth concept label from BARC)
        - ICL
        - common_lib
        """
        puzzle_id = problem.uid
        if self.problem_data and puzzle_id not in self.problem_data:
            logger.warning(f"Problem data not found for {puzzle_id}. Using defaults.")
        puzzle_data = self.problem_data.get(puzzle_id, {"0": {}})

        prompts = {}
        for variant_id, variant_data in puzzle_data.items():
            prompts[variant_id] = make_prompt(
                problem=problem,
                hint=variant_data.get("hint", None),
                description=variant_data.get("description", None),
                hint_template_key=self.prompt_options.hint_template_key,
                instruction_key=self.prompt_options.instruction_key,
                require_hint_citations=self.prompt_options.require_hint_citations,
                **kwargs,
            )
        return prompts

    def build_retry_prompt(
        self,
        initial_prompt: str,
        solution_thread: SolutionThread,
        new_concepts: str | None = None,
    ) -> str:
        return make_retry_prompt(
            initial_prompt=initial_prompt,
            solution_thread=solution_thread,
            num_feedback_passes=self.retry_policy.num_feedback_passes,
            error_feedback=self.retry_policy.error_feedback,
            include_past_outcomes=self.retry_policy.include_past_outcomes,
            new_concepts=new_concepts,
        )
