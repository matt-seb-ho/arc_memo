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

    def __post_init__(self):
        if isinstance(self.output, list):
            # if output is a list, convert it to a numpy array
            self.output = np.array(self.output)


@dataclass
class SolutionStep:
    # tags
    step_idx: int
    thread_id: str
    branch_id: str
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

    def is_strictly_correct(self) -> bool:
        if not self.validated:
            return False
        if self.parsing_error:
            return False
        for result in self.train_results + self.test_results:
            if not result.correct:
                return False
        return True


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
        _make_solution_tree_serializable_dict(initial_dict)
        return initial_dict

    def get_or_create_branch(self, branch_id: str):
        if branch_id not in self.prompt_branches:
            self.prompt_branches[branch_id] = PromptBranch(branch_id=branch_id)
        return self.prompt_branches[branch_id]


def _make_solution_tree_serializable_dict(d) -> dict:
    if isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _make_solution_tree_serializable_dict(v)
        return d
    elif isinstance(d, list):
        for i in range(len(d)):
            d[i] = _make_solution_tree_serializable_dict(d[i])
        return d
    else:
        return d


def create_solution_tree_from_serialized_dict(d: dict) -> SolutionTree:
    tree = SolutionTree(puzzle_id=d["puzzle_id"])
    branches_dict = d.get("prompt_branches", {})
    for b_id, b_dict in branches_dict.items():
        branch = tree.get_or_create_branch(b_id)
        threads = b_dict.get("threads", {})
        for t_id, t_dict in threads.items():
            thread = branch.get_or_create_thread(t_id)
            steps = t_dict.get("steps", [])
            for step_dict in steps:
                step = SolutionStep(
                    step_idx=step_dict["step_idx"],
                    thread_id=t_id,
                    branch_id=b_id,
                    puzzle_id=d["puzzle_id"],
                    completion=step_dict.get("completion", None),
                    validated=step_dict.get("validated", False),
                    parsing_error=step_dict.get("parsing_error", None),
                )
                step.train_results = [
                    IOPairExecutionResult(**res)
                    for res in step_dict.get("train_results", [])
                ]
                step.test_results = [
                    IOPairExecutionResult(**res)
                    for res in step_dict.get("test_results", [])
                ]
                thread.steps.append(step)
    return tree
