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

import re
from typing import Optional

import numpy as np
from arc import train_problems, validation_problems

from concept_mem.constants import DEFAULT_CODE


class IOPair:
    x: np.ndarray
    y: np.ndarray

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        # check type
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        # check shape
        assert len(self.x.shape) == 2
        assert len(self.y.shape) == 2


class Problem:
    # typing hints
    uid: str
    filename: str
    seed_id: str
    code: str
    train_pairs: list
    test_pairs: list

    def __init__(
        self,
        uid: Optional[str] = None,
        filename: Optional[str] = None,
        code: Optional[str] = DEFAULT_CODE,
        seed_id: Optional[str] = None,
        train_pairs: Optional[list] = None,
        test_pairs: Optional[list] = None,
    ):
        self.uid = uid
        self.filename = filename
        self.seed_id = None
        if filename:
            self.seed_id = filename.split(".")[0]
            if "_" in self.seed_id:
                self.seed_id = self.seed_id.split("_")[0]
        if seed_id:
            self.seed_id = seed_id
        if self.seed_id:
            pattern = r"[0-9a-f]{8}"
            assert re.match(pattern, self.seed_id)
            self.load_arc_problem(self.seed_id)

        self.code = code
        if train_pairs:
            self.train_pairs = train_pairs
        if test_pairs:
            self.test_pairs = test_pairs

        assert self.code, "Code is not provided"
        assert self.train_pairs, "Train pairs are not provided"
        assert self.test_pairs, "Test pairs are not provided"
        # check type
        assert isinstance(self.train_pairs, list)
        assert isinstance(self.test_pairs, list)
        assert all(isinstance(pair, IOPair) for pair in self.train_pairs)
        assert all(isinstance(pair, IOPair) for pair in self.test_pairs)

    def load_arc_problem(self, seed_id):
        # using train_problems
        arc_problem = None
        for problem in train_problems + validation_problems:
            if problem.uid == seed_id:
                arc_problem = problem
                break
        assert arc_problem is not None
        self.train_pairs = []
        for pair in arc_problem.train_pairs:
            # self.train_pairs.append(IOPair(pair.x.T, pair.y.T))
            self.train_pairs.append(IOPair(pair.x, pair.y))
        self.test_pairs = []
        for pair in arc_problem.test_pairs:
            # self.test_pairs.append(IOPair(pair.x.T, pair.y.T))
            self.test_pairs.append(IOPair(pair.x, pair.y))
