"""
Defines IOPair and Problem classes for formatting ARC problems into.
- these correspond to arc.types.ArcIOPair and arc.types.ArcProblem respectively.
- differences
  - ArcIOPair has a plot function
  - Problem is designed for the BARC dataset, so it has extra fields...
    - filename: seed file, uses to get the seed id
    - seed_id: if provided (by seed_id or filename), it loads the ARC problem
    - code: the (BARC synthesized/seed) solution

I'm not entirely convinced we need IOPair at all
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import orjson
from arc import ArcIOPair, ArcProblem

from concept_mem.constants import BARC_SEEDS_PATH, REPO_ROOT

from .barc_seed_processing import (
    extract_code_from_seed,
    get_barc_seed_ids,
)

CUSTOM_SPLITS = {
    "val100": REPO_ROOT / "data/testbeds/validation_n100_uids.json",
}


@dataclass
class IOPair:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        # check type
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        # check shape
        assert len(self.x.shape) == 2
        assert len(self.y.shape) == 2

    @classmethod
    def from_arc_io_pair(cls, arc_io_pair: ArcIOPair) -> "IOPair":
        """Convert from ArcIOPair to IOPair."""
        return cls(arc_io_pair.x, arc_io_pair.y)


@dataclass
class Problem:
    uid: str
    train_pairs: list[IOPair]
    test_pairs: list[IOPair]

    # puzzle category
    split: str | None = None

    # for BARC seeds or custom problems
    file_path: Path | None = None

    # for BARC seed problems with handwritten annotations
    code: str | None = None

    def __post_init__(self) -> None:
        # initialize uid from file instead
        if self.uid is None and self.file_path:
            file_stem = self.file_path.stem
            if "_" in file_stem:
                self.uid = file_stem.split("_")[0]
            else:
                self.uid = file_stem

        # if self.seed_id:
        #     pattern = r"[0-9a-f]{8}"
        #     assert re.match(pattern, self.seed_id)
        #     self.load_arc_problem(self.seed_id)

        # check validity
        # - must have train/test IO pairs
        assert self.train_pairs, "train pairs are not provided"
        assert self.test_pairs, "test pairs are not provided"
        # - check types
        assert isinstance(self.train_pairs, list)
        assert isinstance(self.test_pairs, list)
        assert all(isinstance(pair, IOPair) for pair in self.train_pairs)
        assert all(isinstance(pair, IOPair) for pair in self.test_pairs)

    def __repr__(self) -> str:
        if self.code is not None:
            return f"arc_puzzle # {self.uid} # {self.split} # barc seed"
        return f"arc_puzzle # {self.uid} # {self.split}"

    @classmethod
    def from_file(
        cls,
        file_path: Path | str,
        uid: str | None = None,
        split: str | None = None,
    ) -> "Problem":
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = REPO_ROOT / file_path
        assert file_path.exists(), f"File {file_path} does not exist"

        if file_path.suffix == ".json":
            # load file
            with open(file_path, "rb") as f:
                data = orjson.loads(f.read())
            # set uid, code
            puzzle_id = uid or data.get("uid", None)
            if puzzle_id is None:
                puzzle_id = file_path.stem
            code = data.get("code", None)
            # initialize train/test pairs
            train_pairs = cls._initialize_io_pair_list(data.get("train", []))
            test_pairs = cls._initialize_io_pair_list(data.get("test", []))
            return cls(
                uid=puzzle_id,
                train_pairs=train_pairs,
                test_pairs=test_pairs,
                split=split,
                file_path=file_path,
                code=code,
            )
        elif file_path.suffix == ".py":
            # assume it's a BARC seed file-- i.e. a solution for an ARC problem
            seed_id = file_path.stem.split("_")[0]
            problem = Problem.from_puzzle_id(seed_id)
            problem.file_path = file_path
            problem.code = extract_code_from_seed(file_path.read_text())
            return problem

    @classmethod
    def from_puzzle_id(cls, puzzle_id: str) -> "Problem":
        train_split = load_arc_data("train")
        validation_split = load_arc_data("validation")
        if puzzle_id in train_split:
            return train_split[puzzle_id]
        elif puzzle_id in validation_split:
            return validation_split[puzzle_id]
        else:
            raise ValueError(
                f"Puzzle ID {puzzle_id} not found in train or validation splits."
            )

    @classmethod
    def from_arc_problem(cls, arc_problem: ArcProblem) -> "Problem":
        """Convert from ArcProblem to Problem."""
        train_pairs = [
            IOPair.from_arc_io_pair(pair) for pair in arc_problem.train_pairs
        ]
        test_pairs = [IOPair.from_arc_io_pair(pair) for pair in arc_problem.test_pairs]
        return cls(
            uid=arc_problem.uid,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
        )

    @staticmethod
    def _initialize_io_pair_list(
        pairs_data: list[dict[str, list[list[int]]]],
    ) -> list[IOPair]:
        """Initialize a list of IOPairs from a list of dicts."""
        return [
            IOPair(np.array(pair["input"]), np.array(pair["output"]))
            for pair in pairs_data
        ]


@cache
def load_arc_data(
    split: str = "train",
    barc_seeds_path: Path = BARC_SEEDS_PATH,
    puzzle_ids: Path | str | list[str] | None = None,
) -> dict[str, Problem]:
    if puzzle_ids is None and split in CUSTOM_SPLITS:
        puzzle_ids = CUSTOM_SPLITS[split]

    if split == "train":
        from arc import train_problems

        train_split = {}
        seed_ids = get_barc_seed_ids(barc_seeds_path)
        for problem in train_problems:
            # convert from ArcProblem to Problem
            reformatted = Problem.from_arc_problem(problem)
            # tag with train split label
            reformatted.split = "train"
            # add code from BARC seed if available
            if problem.uid in seed_ids:
                seed_code_path = barc_seeds_path / f"{problem.uid}.py"
                if seed_code_path.exists():
                    reformatted.code = extract_code_from_seed(
                        seed_code_path.read_text()
                    )
            train_split[problem.uid] = reformatted
        return train_split
    elif split == "validation":
        from arc import validation_problems

        validation_split = {}
        for problem in validation_problems:
            # validation_split[problem.uid] = Problem.from_arc_problem(problem)
            reformatted = Problem.from_arc_problem(problem)
            reformatted.split = "validation"
            validation_split[problem.uid] = reformatted
        return validation_split
    elif split == "barc_seeds":
        train_split = load_arc_data("train")
        barc_seeds = {
            puzzle_id: problem
            for puzzle_id, problem in train_split.items()
            if problem.code
        }
        return barc_seeds
    elif puzzle_ids is not None:
        if isinstance(puzzle_ids, str):
            puzzle_ids = Path(puzzle_ids)
        if isinstance(puzzle_ids, Path):
            if not puzzle_ids.is_absolute():
                puzzle_ids = REPO_ROOT / puzzle_ids
            puzzle_ids = orjson.loads(puzzle_ids.read_bytes())
        else:
            assert isinstance(puzzle_ids, list)
        return {
            puzzle_id: Problem.from_puzzle_id(puzzle_id) for puzzle_id in puzzle_ids
        }

    raise ValueError(f"Undefined split: {split} provided without puzzle IDs.")
