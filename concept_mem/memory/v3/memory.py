# concept_mem/memory.py
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import orjson
import yaml

from .concept import Concept
from concept_mem.constants import REPO_ROOT
from concept_mem.utils import extract_yaml_block

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------- #
#  Supplementary dataclass for storing full-puzzle info                     #
# ------------------------------------------------------------------------- #
@dataclass
class ProblemSolution:
    problem_id: str
    solution: Optional[str] = None
    summary: Optional[str] = None
    pseudocode: Optional[str] = None


# ------------------------------------------------------------------------- #
#  Core memory object                                                       #
# ------------------------------------------------------------------------- #
class ConceptMemory:
    """
    Thin in-memory store that handles deduplication / grouping of concepts.
    """

    DEFAULT_CATEGORIES = ["structure", "routine"]

    def __init__(self) -> None:
        self.concepts: Dict[str, Concept] = {}  # keyed by concept name
        self.categories: Dict[str, List[str]] = defaultdict(
            list
        )  # kind ➜ [concept names]
        self.solutions: Dict[str, ProblemSolution] = {}

    # ------------------------------------------------------------------ #
    #  Population helpers                                                #
    # ------------------------------------------------------------------ #
    def write_concept(self, puzzle_id: str, annotation: Dict) -> None:
        """Insert or merge a single concept description coming from a puzzle."""
        name = annotation.get("concept") or annotation.get("name")
        if not name:
            logger.info(f"[{puzzle_id}] Skipping – no 'concept' field.")
            return

        kind = annotation.get("kind")
        if kind not in {"structure", "routine"}:
            logger.info(
                f"[{puzzle_id}] Concept '{name}' missing/invalid `kind` – skipped."
            )
            return

        # Existing concept ➜ merge; otherwise create
        if name in self.concepts:
            self.concepts[name].update(puzzle_id, annotation)
        else:
            concept = Concept(
                name=name,
                kind=kind,
                description=annotation.get("description"),
                output=annotation.get("output"),
                # parameters=[
                #     # Accept already-canonicalised dicts or simple strings
                #     p if isinstance(p, dict) else {"name": str(p)}
                #     for p in annotation.get("parameters", [])
                # ],
                # implementation=annotation.get("implementation", []),
            )
            concept.update(puzzle_id, annotation)
            self.concepts[name] = concept
            self.categories[kind].append(name)

    def write_solution(
        self, puzzle_id: str, solution: Optional[str], annotation: Dict
    ) -> None:
        self.solutions[puzzle_id] = ProblemSolution(
            problem_id=puzzle_id,
            solution=solution,
            summary=annotation.get("summary"),
            pseudocode=annotation.get("pseudocode"),
        )

    # ------------------------------------------------------------------ #
    #  Pretty-print helpers                                              #
    # ------------------------------------------------------------------ #
    # def to_string(
    #     self,
    #     categories: Optional[List[str]] = None,
    #     *,
    #     include_description: bool = True,
    #     indentation: int = 0,
    #     filter_concept: Optional[Callable[[Concept], bool]] = None,
    # ) -> str:
    #     """Render a prompt-friendly view of memory."""
    #     if categories is None:
    #         categories = self.DEFAULT_CATEGORIES

    #     blocks: List[str] = []
    #     for cat in categories:
    #         names = self.categories.get(cat, [])
    #         if not names:
    #             continue

    #         blocks.append(f"## {cat} concepts")
    #         for n in names:
    #             c = self.concepts[n]
    #             if filter_concept and not filter_concept(c):
    #                 continue
    #             blocks.append(
    #                 c.to_string(
    #                     include_description=include_description,
    #                     indentation=indentation,
    #                 )
    #             )
    #         blocks.append("")  # blank line spacer
    #     return "\n".join(blocks)

    def to_string(
        self,
        categories: Optional[List[str]] = None,
        *,
        include_description: bool = True,
        indentation: int = 0,
        filter_concept: Optional[Callable[[Concept], bool]] = None,
        concept_names: Optional[List[str]] = None,  # NEW
    ) -> str:
        """
        Render a prompt-friendly view of memory.

        Parameters
        ----------
        categories
            Order of high-level groups to print (defaults to ['structure', 'routine']).
        include_description
            Whether to include the natural-language description of each concept.
        indentation
            Number of spaces to indent every line.
        filter_concept
            Callable that receives a `Concept` and returns True/False to keep/drop it.
        concept_names
            If provided, only concepts whose *name* appears in this list are rendered,
            though they still appear under their usual category headings.
        """
        if categories is None:
            categories = self.DEFAULT_CATEGORIES

        name_whitelist = set(concept_names) if concept_names else None
        blocks: List[str] = []

        for cat in categories:
            cat_names = self.categories.get(cat, [])
            # Apply user-supplied concept filter first
            if name_whitelist is not None:
                cat_names = [n for n in cat_names if n in name_whitelist]

            if not cat_names:  # nothing survives → skip category
                continue

            blocks.append(f"## {cat} concepts")
            for n in cat_names:
                concept = self.concepts[n]
                if filter_concept and not filter_concept(concept):
                    continue
                blocks.append(
                    concept.to_string(
                        include_description=include_description,
                        indentation=indentation,
                    )
                )
            blocks.append("")  # blank spacer line

        return "\n".join(blocks)

    # ------------------------------------------------------------------ #
    #  Parsing model output                                              #
    # ------------------------------------------------------------------ #
    def update_from_model_output(self, puzzle_id: str, llm_output: str) -> None:
        """
        Expects the model to emit a YAML list of concept dicts inside a
        ```yaml ... ``` fenced block.
        """
        raw_yaml = extract_yaml_block(llm_output)
        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError as e:
            logger.error(f"[{puzzle_id}] YAML parse error: {e}")
            return

        if not isinstance(parsed, list):
            logger.info(f"[{puzzle_id}] Expected list of concepts, got {type(parsed)}.")
            return

        for concept_anno in parsed:
            self.write_concept(puzzle_id, concept_anno)

    # ------------------------------------------------------------------ #
    #  Bulk initialise helpers                                           #
    # ------------------------------------------------------------------ #
    def initialise_solutions(self, mapping: Dict[str, Dict]) -> None:
        for pid, ann in mapping.items():
            self.solutions[pid] = ProblemSolution(
                problem_id=pid,
                solution=ann.get("solution"),
                summary=ann.get("summary"),
                pseudocode=ann.get("pseudocode"),
            )

    def initialise_from_annotations(self, annotations: Dict[str, Dict]) -> None:
        for pid, ann in annotations.items():
            self.write_solution(pid, None, ann)
            for concept_ann in ann.get("concepts", []):
                self.write_concept(pid, concept_ann)

    # ------------------------------------------------------------------ #
    #  Persistence                                                       #
    # ------------------------------------------------------------------ #
    def save_to_file(self, path: Path) -> None:
        data = {
            "concepts": {n: asdict(c) for n, c in self.concepts.items()},
            "solutions": {pid: asdict(s) for pid, s in self.solutions.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    def load_from_file(self, path: Path) -> None:
        if not str(path).startswith("/"):
            path = REPO_ROOT / path

        mem = orjson.loads(path.read_bytes())
        self.concepts = {n: Concept(**c) for n, c in mem["concepts"].items()}
        self.solutions = {
            pid: ProblemSolution(**s) for pid, s in mem["solutions"].items()
        }
        self.categories.clear()
        for name, concept in self.concepts.items():
            self.categories[concept.kind].append(name)
