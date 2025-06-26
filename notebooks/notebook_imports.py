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
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

plt.style.use("rose-pine-dawn")

package_path = os.path.abspath("..")
if package_path not in sys.path:
    sys.path.append(package_path)

from llmplus import GenerationConfig, LLMClient, Provider

from concept_mem.constants import DOTENV_PATH, REPO_ROOT
from concept_mem.types import Problem
from concept_mem.utils import (
    load_arc_data,
    read_json,
    read_yaml,
    visualize_problem,
    write_json,
    write_yaml,
)
