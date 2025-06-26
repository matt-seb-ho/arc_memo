from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class IOPairExecutionResult:
    # tags
    is_train: bool = False
    pair_idx: int = -1

    # payload
    # - output should be a numpy array if execution was successful
    # - we try to preserve it as a string if not to use as feedback
    # - if coercion to array/string fails, it will be None
    output: np.ndarray | str | None = None
    correct: bool = False
    error: str | None = None
    stdout: str | None = None


@dataclass
class SolutionStep:
    # tags
    step_idx: int
    thread_id: str
    prompt_id: str
    puzzle_id: str

    # model output
    completion: str | None = None

    # execution/evaluation results
    # - whether the solution was validated
    validated: bool = False
    # - whether we extracted a markdown code block from the completion
    parsing_error: str | None = None
    train_results: list[IOPairExecutionResult] = field(default_factory=list)
    test_results: list[IOPairExecutionResult] = field(default_factory=list)


@dataclass
class SolutionThread:
    thread_id: str
    steps: list[SolutionStep] = field(default_factory=list)


@dataclass
class PromptBranch:
    branch_id: str
    threads: dict[str, SolutionThread] = field(default_factory=dict)
    prompt: str | None = None

    def get_or_create_thread(self, thread_id: str):
        if thread_id not in self.threads:
            self.threads[thread_id] = SolutionThread(thread_id=thread_id)
        return self.threads[thread_id]


@dataclass
class SolutionTree:
    puzzle_id: str
    prompt_branches: dict[str, PromptBranch] = field(default_factory=dict)

    def to_serializable_dict(self) -> dict:
        initial_dict = asdict(self)
        _make_solution_tree_dict_serializable(initial_dict)
        return initial_dict

    def get_or_create_branch(self, branch_id: str):
        if branch_id not in self.prompt_branches:
            self.prompt_branches[branch_id] = PromptBranch(branch_id=branch_id)
        return self.prompt_branches[branch_id]


def _make_solution_tree_dict_serializable(d) -> None:
    if isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _make_solution_tree_dict_serializable(v)
    elif isinstance(d, list):
        for i in range(len(d)):
            d[i] = _make_solution_tree_dict_serializable(d[i])
