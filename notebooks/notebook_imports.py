from functools import cache
import importlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

plt.style.use("rose-pine-dawn")

package_path = os.path.abspath("..")
if package_path not in sys.path:
    sys.path.append(package_path)

from llmplus import GenerationConfig, LLMClient, Provider

from concept_mem.constants import DATA_DIR, DOTENV_PATH, REPO_ROOT
from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.evaluation.score_tree import (
    flatten_solution_trees,
)
from concept_mem.evaluation.solution_tree import (
    SolutionTree,
    create_solution_tree_from_serialized_dict,
)
from concept_mem.utils import (
    get_puzzle_url,
    read_json,
    read_yaml,
    visualize_problem,
    write_json,
    write_yaml,
)


def result_dir_to_df(
    res_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    iter_dir = get_latest_iteration_dir(res_dir)
    json_data = read_json(iter_dir / "solution_trees.json")
    sln_trees = {
        k: create_solution_tree_from_serialized_dict(v) for k, v in json_data.items()
    }
    df = flatten_solution_trees(sln_trees)
    return df


def get_latest_iteration_dir(parent_dir: Path) -> Path:
    # first get highest iteration dir either "iteration_{i}" or "iter_{i}"
    ls = os.listdir(parent_dir)
    highest = -1
    prefix = "iteration_"
    for name in ls:
        if name.startswith("iteration_") or name.startswith("iter_"):
            try:
                iteration_num = int(name.split("_")[1])
                if iteration_num > highest:
                    highest = iteration_num
                    if name.startswith("iteration_"):
                        prefix = "iteration_"
                    else:
                        prefix = "iter_"
            except ValueError:
                continue
    assert highest != -1
    return parent_dir / f"{prefix}{highest}"


def print_to_file(s) -> None:
    target_path = REPO_ROOT / "notebooks/output/temp.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(s)
    print(f"Output written to:\n{target_path}")


def oai_token_len(text: str) -> int:
    gpt4o_encoder = _get_gpt4o_tokenizer()
    return len(gpt4o_encoder.encode(text))


@cache
def _get_gpt4o_tokenizer():
    return tiktoken.encoding_for_model("gpt-4o")
