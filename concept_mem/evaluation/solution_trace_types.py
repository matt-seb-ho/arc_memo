from dataclasses import asdict, dataclass, field

import numpy as np

from concept_mem.data.arc_agi import Problem
from concept_mem.evaluation.score_completions import (
    parse_code_solution,
    score_transform_function,
)


@dataclass
class RefinementPass:
    # id tags
    pass_id: int
    path_id: int
    variant_id: str

    # core data
    completion: str

    # evaluation
    scored: bool = False
    train_scores: list[bool] = field(default_factory=list)
    test_scores: list[bool] = field(default_factory=list)
    train_outputs: list[np.ndarray | str | None] = field(default_factory=list)
    test_outputs: list[np.ndarray | str | None] = field(default_factory=list)
    parsing_error: str | None = None

    def score(self, problem: Problem, timeout: float = 2.0):
        if self.scored:
            return
        self.scored = True
        code, err = parse_code_solution(self.completion)
        if err:
            self.parsing_error = err
            return
        _, train_correct, train_outputs = score_transform_function(
            problem, code, split="train", timeout=timeout
        )
        _, test_correct, test_outputs = score_transform_function(
            problem, code, split="test", timeout=timeout
        )
        self.train_scores = train_correct
        self.test_scores = test_correct
        self.train_outputs = train_outputs
        self.test_outputs = test_outputs

    def get_json_dict(self) -> dict:
        # need to convert output fields' numpy arrays to nested lists
        res = asdict(self)
        res["train_outputs"] = self.reformat_output_list_for_json(self.train_outputs)
        res["test_outputs"] = self.reformat_output_list_for_json(self.test_outputs)
        return res

    @staticmethod
    def reformat_output_list_for_json(
        outputs: list[np.ndarray | str | None],
    ) -> list[list | str | None]:
        """Convert a list of numpy arrays or strings to a list of lists or strings."""
        return [
            output.tolist() if isinstance(output, np.ndarray) else output
            for output in outputs
        ]


@dataclass
class SolutionPath:
    path_id: int
    passes: list[RefinementPass]

    def last(self) -> RefinementPass:
        assert self.passes, "No passes recorded for this path"
        return self.passes[-1]


@dataclass
class PromptVariant:
    variant_id: str
    paths: dict[int, SolutionPath] = field(default_factory=dict)

    def get_path(self, path_id: int) -> SolutionPath:
        if path_id not in self.paths:
            self.paths[path_id] = SolutionPath(path_id=path_id, passes=[])
        return self.paths[path_id]


@dataclass
class PuzzleTrace:
    puzzle_id: str
    prompt_variants: dict[str, PromptVariant] = field(default_factory=dict)

    def get_variant(self, variant_id: str) -> PromptVariant:
        if variant_id not in self.prompt_variants:
            self.prompt_variants[variant_id] = PromptVariant(variant_id=variant_id)
        return self.prompt_variants[variant_id]
