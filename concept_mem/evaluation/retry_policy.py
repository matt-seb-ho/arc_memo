from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from llmplus import GenerationConfig

from concept_mem.evaluation.solution_trace_types import RefinementPass


class RetryCriterion(Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass
class RetryPolicy:
    """
    Defines when and how many times we retry a puzzle-solving request.

    Attributes:
        max_tries: int
            Maximum number of total attempts per sample (including the first).
        criterion: RetryCriterion
            Which scores to inspect (train vs test) to determine failure.
        error_feedback: str
            Mode for including past error messages in retry prompts ('first' or 'all').
        num_feedback_passes: int
            How many most recent attempts to include in the retry prompt.
            Use -1 to include all attempts.
        include_past_outcomes: bool
            Whether to show boolean pass/fail flags alongside errors.
    """

    max_passes: int = 3
    criterion: RetryCriterion = RetryCriterion.TRAIN
    error_feedback: str = "all"
    num_feedback_passes: int = 1
    include_past_outcomes: bool = True

    # reselect options
    reselect_concepts: bool = False
    lesson_file: Path | None = None
    reselect_model: str = "gpt-4.1-2025-04-14"
    reselect_gen_cfg: GenerationConfig = field(default_factory=GenerationConfig)
    reselect_k: int = 5

    def needs_retry(self, latest: RefinementPass) -> bool:
        """
        Determine if the latest attempt in should be retried.

        Steps:
        1. Select either its train_scores or test_scores based on `criterion`.
        2. Return True if:
           - scores is None (parsing/execution never produced scores),
           - or any score in the list is False.

        Return True if *latest* SampleResult fails the chosen criterion.
        """
        if self.criterion == RetryCriterion.TRAIN:
            scores = latest.train_scores
        else:
            scores = latest.test_scores
        return (scores is None) or (len(scores) == 0) or (not all(scores))
