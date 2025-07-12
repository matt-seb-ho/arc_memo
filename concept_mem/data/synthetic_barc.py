import logging
from functools import cache
from typing import Optional

import numpy as np
from datasets import Dataset, load_dataset

from concept_mem.constants import BARC_DATASET_ID
from concept_mem.data.arc_agi import IOPair, Problem

# set up logging
logger = logging.getLogger(__name__)


@cache
def load_barc() -> Dataset:
    dataset = load_dataset(BARC_DATASET_ID, split="train")
    return dataset


def get_problem_from_barc(
    dataset: Dataset,
    idx: int,
    uid: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[Problem]:
    r = dataset[idx]

    # get input-output pairs
    all_pairs = []
    for example in r["examples"]:
        input_grid = np.array(example[0])
        output_grid = np.array(example[1])
        all_pairs.append(IOPair(input_grid, output_grid))

    # get code
    code = extract_code_from_source(r["source"], idx)

    # use first 3 pairs as train pairs and 4th pair as test pair
    problem = Problem(
        uid=uid,
        filename=filename,
        code=code,
        train_pairs=all_pairs[0:3],
        test_pairs=[all_pairs[3]],
    )
    return problem


def extract_code_from_source(code: str, idx: int | None = None) -> str | None:
    """Extracts code from `source` column on BARC synthetic dataset."""

    # preprocess solution code
    if "def generate_input" in code:
        code = code.split("def generate_input")[0].strip()
    else:
        logger.debug(
            f"generate_input function not found in code for BARC problem {idx}"
        )
    if "def main(" in code:
        code = code.replace("def main(", "def transform(")
    else:
        logger.warning(f"main function not found in code for BARC problem {idx}")
        code = None
    return code
