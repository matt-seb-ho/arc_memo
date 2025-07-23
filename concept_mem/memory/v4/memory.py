# concept_mem/memory.py
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import orjson
import yaml

from concept_mem.memory.v4.concept import Concept, ParameterSpec, maybe_parse_typedef
from concept_mem.constants import REPO_ROOT
from concept_mem.utils import extract_yaml_block

logger = logging.getLogger(__name__)


@dataclass
class ProblemSolution:
    problem_id: str
    solution: str | None = None
    summary: str | None = None
    pseudocode: str | None = None


class ConceptMemory:
    """
    Stores concepts, solutions, and custom type defs.

    Custom types are auto-extracted whenever we see a 'typing' string that
    looks like 'Name := annotation'.
    """

    DEFAULT_CATEGORY_ORDER = [
        "structure",
        "types",
        "grid manipulation",
        "other_routines",
    ]

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.categories: dict[str, list[str]] = defaultdict(
            list
        )  # 'structure'/'routine' ➜ names
        self.solutions: dict[str, ProblemSolution] = {}
        self.custom_types: dict[str, str] = {}  # name -> annotation

    # ---------------------------- Ingestion ----------------------------- #
    def write_concept(self, puzzle_id: str, ann: dict) -> None:
        name = ann.get("concept") or ann.get("name")
        if not name:
            logger.info(f"[{puzzle_id}] Skipping concept: missing 'concept' field.")
            return

        kind = ann.get("kind")
        if kind not in {"structure", "routine"}:
            logger.info(f"[{puzzle_id}] Concept '{name}' invalid kind '{kind}'.")
            return

        # normalise fields
        routine_subtype = ann.get("routine_subtype")
        if kind == "routine" and not routine_subtype:
            routine_subtype = None

        # Ensure parameters list is canonical
        params = []
        for p in ann.get("parameters", []):
            if not isinstance(p, dict):
                logger.info(f"non-dict parameter in concept {name}: {p}")
                p = {"name": str(p)}
            params.append(p)

        concept_exists = name in self.concepts
        if concept_exists:
            self.concepts[name].update(puzzle_id, ann)
        else:
            c = Concept(
                name=name,
                kind=kind,
                routine_subtype=routine_subtype,
                output_typing=ann.get("output_typing", None),
                parameters=[ParameterSpec(**p) for p in params],
                description=ann.get("description"),
                cues=ann.get("cues", []),
                implementation=ann.get("implementation", []),
            )
            c.update(puzzle_id, ann)
            self.concepts[name] = c
            self.categories[kind].append(name)

        # Extract & record any custom types from this concept
        self._harvest_typedefs(self.concepts[name])

    def write_solution(self, puzzle_id: str, solution: str | None, ann: dict) -> None:
        self.solutions[puzzle_id] = ProblemSolution(
            problem_id=puzzle_id,
            solution=solution,
            summary=ann.get("summary"),
            pseudocode=ann.get("pseudocode"),
        )

    def _harvest_typedefs(self, concept: Concept) -> None:
        # from output_typing
        for maybe in [concept.output_typing]:
            parsed = maybe_parse_typedef(maybe)
            if parsed:
                n, t = parsed
                self.custom_types[n] = t

        # from parameters
        for p in concept.parameters:
            parsed = maybe_parse_typedef(p.typing)
            if parsed:
                n, t = parsed
                self.custom_types[n] = t

    # ------------------------- Rendering -------------------------------- #
    def to_string(
        self,
        *,
        concept_names: list[str] | None = None,
        include_description: bool = True,
        indentation: int = 0,
        filter_concept: Callable[[Concept], bool] | None = None,
    ) -> str:
        """
        Master renderer following the new section order.

        Sections:
          1) structures
          2) types (custom typedefs)
          3) grid manipulation routines
          4) other routines grouped by routine_subtype
        """
        whitelist = set(concept_names) if concept_names else None

        blocks: list[str] = []

        # 1. Structures
        struct_block = self.to_string_structures(
            whitelist=whitelist,
            include_description=include_description,
            indentation=indentation,
            filter_concept=filter_concept,
        )
        if struct_block:
            blocks.append(struct_block)

        # 2. Types
        types_block = self.to_string_types(indentation=indentation)
        if types_block:
            blocks.append(types_block)

        # 3. Grid manipulation routines
        grid_block = self.to_string_grid_manip(
            whitelist=whitelist,
            include_description=include_description,
            indentation=indentation,
            filter_concept=filter_concept,
        )
        if grid_block:
            blocks.append(grid_block)

        # 4. Other routines
        other_block = self.to_string_other_routines(
            whitelist=whitelist,
            include_description=include_description,
            indentation=indentation,
            filter_concept=filter_concept,
        )
        if other_block:
            blocks.append(other_block)

        return "\n\n".join(blocks).rstrip()

    # ---- Individual sections ------------------------------------------ #
    def to_string_structures(
        self,
        *,
        whitelist: set | None = None,
        include_description: bool,
        indentation: int,
        filter_concept: Callable[[Concept], bool] | None,
    ) -> str:
        names = [
            n
            for n in self.categories.get("structure", [])
            if (not whitelist or n in whitelist)
        ]
        if not names:
            return ""
        lines = ["## structure concepts"]
        for n in names:
            c = self.concepts[n]
            if filter_concept and not filter_concept(c):
                continue
            lines.append(
                c.to_string(
                    include_description=include_description,
                    indentation=indentation,
                )
            )
        return "\n".join(lines)

    def to_string_types(self, *, indentation: int) -> str:
        if not self.custom_types:
            return ""
        ind = " " * indentation
        lines = ["## types"]
        for name, anno in sorted(self.custom_types.items()):
            lines.append(f"{ind}- {name} := {anno}")
        return "\n".join(lines)

    def to_string_grid_manip(
        self,
        *,
        whitelist: set | None,
        include_description: bool,
        indentation: int,
        filter_concept: Callable[[Concept], bool] | None,
    ) -> str:
        # subset of routines
        grid_names = [
            n
            for n in self.categories.get("routine", [])
            if (not whitelist or n in whitelist)
            and self.concepts[n].routine_subtype == "grid manipulation"
        ]
        if not grid_names:
            return ""
        lines = ["## grid manipulation routines"]
        for n in grid_names:
            c = self.concepts[n]
            if filter_concept and not filter_concept(c):
                continue
            lines.append(
                c.to_string(
                    include_description=include_description,
                    indentation=indentation,
                    skip_subtype=True,  # already under the group header
                )
            )
        return "\n".join(lines)

    def to_string_other_routines(
        self,
        *,
        whitelist: set | None,
        include_description: bool,
        indentation: int,
        filter_concept: Callable[[Concept], bool] | None,
    ) -> str:
        # collect routines with non-grid subtype
        routines = [
            self.concepts[n]
            for n in self.categories.get("routine", [])
            if (not whitelist or n in whitelist)
            and self.concepts[n].routine_subtype != "grid manipulation"
        ]
        if not routines:
            return ""

        # group by subtype
        buckets: dict[str, list[Concept]] = defaultdict(list)
        for c in routines:
            key = c.routine_subtype or "misc"
            buckets[key].append(c)

        lines: list[str] = ["## other routines"]
        for subtype in sorted(buckets.keys()):
            lines.append(f"### {subtype}")
            for c in buckets[subtype]:
                if filter_concept and not filter_concept(c):
                    continue
                lines.append(
                    c.to_string(
                        include_description=include_description,
                        indentation=indentation,
                        skip_subtype=True,
                    )
                )
        return "\n".join(lines)

    # ------------------------ Model output ingest ----------------------- #
    def update_from_model_output(self, puzzle_id: str, llm_output: str) -> None:
        yaml_block = extract_yaml_block(llm_output)
        try:
            parsed = yaml.safe_load(yaml_block)
        except yaml.YAMLError as e:
            logger.error(f"[{puzzle_id}] YAML parse error: {e}")
            return

        if not isinstance(parsed, list):
            logger.info(
                f"[{puzzle_id}] Expected a list of concepts, got {type(parsed)}."
            )
            return

        for concept_anno in parsed:
            self.write_concept(puzzle_id, concept_anno)

    # ----------------------- Initialisation ----------------------------- #
    def initialise_solutions(self, mapping: dict[str, dict]) -> None:
        for pid, ann in mapping.items():
            self.solutions[pid] = ProblemSolution(
                problem_id=pid,
                solution=ann.get("solution"),
                summary=ann.get("summary"),
                pseudocode=ann.get("pseudocode"),
            )

    def initialise_from_annotations(self, annotations: dict[str, dict]) -> None:
        for pid, ann in annotations.items():
            self.write_solution(pid, None, ann)
            for concept_ann in ann.get("concepts", []):
                self.write_concept(pid, concept_ann)

    # -------------------------- Persistence ----------------------------- #
    def save_to_file(self, path: Path) -> None:
        blob = {
            "concepts": {n: asdict(c) for n, c in self.concepts.items()},
            "solutions": {pid: asdict(s) for pid, s in self.solutions.items()},
            "custom_types": self.custom_types,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(blob, option=orjson.OPT_INDENT_2))

    def load_from_file(self, path: Path) -> None:
        if not str(path).startswith("/"):
            path = REPO_ROOT / path
        data = orjson.loads(path.read_bytes())

        self.concepts = {n: Concept(**c) for n, c in data["concepts"].items()}
        self.solutions = {
            pid: ProblemSolution(**s) for pid, s in data["solutions"].items()
        }
        self.custom_types = data.get("custom_types", {})

        self.categories.clear()
        for name, concept in self.concepts.items():
            self.categories[concept.kind].append(name)
