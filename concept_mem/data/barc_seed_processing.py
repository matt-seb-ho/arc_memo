import logging
from functools import cache
from pathlib import Path

import orjson

from concept_mem.constants import BARC_SEED_UIDS_PATH

# set up logging
logger = logging.getLogger(__name__)


@cache
def get_barc_seed_ids(
    file_path: Path = BARC_SEED_UIDS_PATH,
) -> set[str]:
    """
    Load the BARC seed UIDs from the JSON file.
    Returns a set of UIDs.
    """
    if not file_path.exists():
        logger.error(f"BARC seed path does not exist: {file_path}")
        return set()
    logger.debug(f"Loading BARC seed UIDs from {file_path}")
    if file_path.suffix == ".json":
        return set(orjson.loads(file_path.read_bytes()))
    if file_path.is_dir():
        seed_ids = set()
        for item in file_path.iterdir():
            if item.suffix == ".py":
                seed_id = item.stem
                if "_" in seed_id:
                    seed_id = seed_id.split("_")[0]
                seed_ids.add(seed_id)
        return seed_ids
    else:
        raise ValueError(
            f"Invalid BARC seed path: {file_path}. Must be a JSON file or a directory."
        )


def extract_code_from_seed(content: str) -> str:
    """Extracts code from a BARC seed file (hand-written solution)."""

    assert (
        "# ============= remove below this point for prompting =============" in content
    )
    content = content.split(
        "# ============= remove below this point for prompting ============="
    )[0].strip()
    content = content.split("def generate_input")[0].strip()
    content = content.replace("def main(", "def transform(")
    return content


def extract_barc_seed_comment_sections(solution: str) -> dict[str, str]:
    """
    Scan through `lines` until the first function definition, pulling out
    any comment‑blocks labelled "# concepts:" and "# description:".

    Returns a dict with keys 'concepts' and/or 'description' if found.
    """
    lines = solution.splitlines()
    sections = {"concepts": [], "description": []}
    current = None

    for line in lines:
        stripped = line.strip()
        # stop once we hit the function definition
        if stripped.startswith("def "):
            break

        # is it a comment?
        if stripped.startswith("#"):
            # drop leading '#' and whitespace
            content = stripped[1:].strip()

            # section headers?
            if content.lower().startswith("concepts:"):
                current = "concepts"
                tail = content[len("concepts:") :].strip()
                if tail:
                    sections[current].append(tail)

            elif content.lower().startswith("description:"):
                current = "description"
                tail = content[len("description:") :].strip()
                if tail:
                    sections[current].append(tail)

            # continuation line?
            elif current in sections:
                if content:
                    sections[current].append(content)

        else:
            # non‑comment line — reset until we hit next header
            current = None

    # join each section’s lines into one paragraph
    return {
        key: " ".join(lines)
        for key, lines in sections.items()
        if lines  # only include non‑empty sections
    }


def remove_concepts_from_barc_seed_solution(solution: str) -> str:
    # we want to remove the "# concepts:" line and the line after it
    lines = solution.split("\n")
    new_lines = []
    skip_next = False
    for line in lines:
        if skip_next:
            skip_next = False
            continue
        if line.startswith("# concepts:"):
            skip_next = True
            continue
        new_lines.append(line)
    return "\n".join(new_lines)
