import logging
import random
from itertools import islice

import omegaconf

from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.utils import read_json

logger = logging.getLogger(__name__)


def load_problems_from_config(
    dataset: str,
    split: str,
    num_problems: int | None,
    problem_ids: list | str | dict[str, str] | None,
) -> dict[str, Problem]:
    """Load ARC‑AGI problems and subset them according to the config."""
    if dataset.lower() == "arc-agi":
        data = load_arc_data(split)
        if problem_ids is None:
            problem_ids = list(data.keys())
        elif isinstance(problem_ids, str):
            problem_ids = read_json(problem_ids)
        elif num_problems and num_problems < len(problem_ids):
            problem_ids = random.sample(problem_ids, num_problems)
        return {pid: data[pid] for pid in problem_ids}
    elif dataset.lower() == "custom":
        assert isinstance(problem_ids, omegaconf.DictConfig), type(problem_ids)
        data = {}
        for problem_id, problem_file in problem_ids.items():
            if problem_file:
                data[problem_id] = Problem.from_file(
                    file_path=problem_file,
                    uid=problem_id,
                )
            else:
                try:
                    data[problem_id] = Problem.from_puzzle_id(problem_id)
                except Exception as e:
                    logger.error(f"Failed to load problem {problem_id}: {e}")
                    continue
        if num_problems and num_problems < len(data):
            return {k: v for k, v in islice(data.items(), num_problems)}
        return data
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
