import os
import re
from functools import partial
from pathlib import Path

import orjson
import yaml
from dotenv import load_dotenv

from concept_mem.constants import REPO_ROOT


# --- file I/O utilities ---------------------------------------------


def prepend_repo_root(file_path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """
    Prepend the repository root to a file path if it is not already absolute.
    """
    file_path_string = str(file_path)
    if file_path_string.startswith("/"):
        return Path(file_path_string)
    return repo_root / file_path_string


def read_json(file_path: str | Path) -> dict:
    file_path = prepend_repo_root(file_path)
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def write_json(data: dict, file_path: str, indent: bool = False) -> None:
    file_path = prepend_repo_root(file_path)
    if indent:
        byte_string = orjson.dumps(data, option=orjson.OPT_INDENT_2)
    else:
        byte_string = orjson.dumps(data)
    with open(file_path, "wb") as f:
        f.write(byte_string)


def read_json_lines(file_path: str) -> list:
    file_path = prepend_repo_root(file_path)
    lines = []
    with open(file_path, "rb") as f:
        for line in f:
            lines.append(orjson.loads(line))
    return lines


def write_json_lines(data: list, file_path: str, append: bool = False) -> None:
    file_path = prepend_repo_root(file_path)
    file_mode = "ab" if (append and Path(file_path).exists()) else "wb"
    with open(file_path, file_mode) as f:
        for line in data:
            f.write(orjson.dumps(line) + b"\n")


def read_yaml(src: Path | str) -> dict | list:
    if isinstance(src, Path):
        file_path = prepend_repo_root(src)
        with open(file_path) as f:
            return yaml.safe_load(f)
    return yaml.safe_load(src)


def write_yaml(data: dict | list, file_path: Path) -> None:
    file_path = prepend_repo_root(file_path)
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# --- utilities for parsing markdown fenced code blocks ------------------

code_block_pattern = re.compile(r"```\n(.*?)\n```", re.DOTALL)
python_block_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
yaml_block_pattern = re.compile(r"```yaml\n(.*?)\n```", re.DOTALL)


def extract_code_block_contents(s: str, first_pattern: re.Pattern | None) -> str | None:
    if first_pattern is not None:
        m = first_pattern.search(s)
        if m:
            return m.group(1).strip()
    m = code_block_pattern.search(s)
    if m:
        return m.group(1).strip()
    return None


extract_yaml_block = partial(
    extract_code_block_contents, first_pattern=yaml_block_pattern
)
extract_python_block = partial(
    extract_code_block_contents, first_pattern=python_block_pattern
)

# --- utilities for parsing angle bracket tags ---------------------------


def parse_markup_tag(s: str, tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    matches = pattern.findall(s)
    return [match.strip() for match in matches if match.strip()]


# --- misc utilities -----------------------------------------------------


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_hf_access_token(env_file: str) -> str:
    load_dotenv(dotenv_path=env_file)
    return os.getenv("HF_ACCESS_TOKEN")
