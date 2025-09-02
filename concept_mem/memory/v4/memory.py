# concept_mem/memory.py
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import orjson
import yaml

from concept_mem.constants import REPO_ROOT
from concept_mem.memory.v4.concept import (
    Concept,
    ParameterSpec,
    maybe_parse_typedef,
)
from concept_mem.utils import extract_yaml_block

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
#                               Dataclasses                             #
# --------------------------------------------------------------------- #
@dataclass
class ProblemSolution:
    problem_id: str
    solution: str | None = None
    summary: str | None = None
    pseudocode: str | None = None


# --------------------------------------------------------------------- #
#                            ConceptMemory                              #
# --------------------------------------------------------------------- #
class ConceptMemory:
    """
    Stores concepts, solutions, and custom type defs.
    """

    DEFAULT_CATEGORY_ORDER = [
        "structure",
        "types",
        "grid manipulation",
        "other_routines",
    ]

    def __init__(self) -> None:
        self.concepts: dict[str, Concept] = {}
        self.categories: dict[str, list[str]] = defaultdict(list)  # kind → names
        self.solutions: dict[str, ProblemSolution] = {}
        self.custom_types: dict[str, str] = {}  # typedef name → annotation

    # ----------------------------------------------------------------- #
    #                              Ingestion                            #
    # ----------------------------------------------------------------- #
    def write_concept(self, puzzle_id: str, ann: dict) -> None:
        name = ann.get("concept") or ann.get("name")
        if not name:
            logger.info(f"[{puzzle_id}] Skipping concept: missing 'concept' field.")
            return
        concept_exists = name in self.concepts

        kind = ann.get("kind", None)
        if kind not in {"structure", "routine"} and not concept_exists:
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

        if concept_exists:
            self.concepts[name].update(puzzle_id, ann)
        else:
            c = Concept(
                name=name,
                kind=kind,
                routine_subtype=routine_subtype,
                output_typing=ann.get("output_typing"),
                parameters=[ParameterSpec(**p) for p in params],
                description=ann.get("description"),
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

    # ----------------------------------------------------------------- #
    #                              Rendering                            #
    # ----------------------------------------------------------------- #
    def to_string(
        self,
        *,
        concept_names: list[str] | None = None,
        include_description: bool = True,
        # unified skip-* controls (all default False → show everything)
        skip_kind: bool = True,
        skip_routine_subtype: bool = True,
        skip_parameters: bool = False,
        skip_parameter_description: bool = True,
        skip_cues: bool = False,
        skip_implementation: bool = False,
        # usage-condensed rendering
        usage_threshold: int = 2,
        show_other_concepts: bool = False,
        indentation: int = 0,
        filter_concept: Callable[[Concept], bool] | None = None,
    ) -> str:
        """
        Render memory in four sections.  If `len(concept.used_in) < usage_threshold`
        the concept is listed only by name under a single summary bullet:
            - lower usage concepts: [name1, name2, ...]
        """
        whitelist = set(concept_names) if concept_names else None
        blocks: list[str] = []

        # 1) Structures -------------------------------------------------
        blk = self._to_string_structures(
            whitelist=whitelist,
            include_description=include_description,
            skip_kind=skip_kind,
            skip_routine_subtype=skip_routine_subtype,
            skip_parameters=skip_parameters,
            skip_parameter_description=skip_parameter_description,
            skip_cues=skip_cues,
            skip_implementation=skip_implementation,
            usage_threshold=usage_threshold,
            show_other_concepts=show_other_concepts,
            indentation=indentation,
            filter_concept=filter_concept,
        )
        if blk:
            blocks.append(blk)

        # 2) Types ------------------------------------------------------
        blk = self._to_string_types(indentation=indentation)
        if blk:
            blocks.append(blk)

        # 3) Grid-manipulation routines --------------------------------
        blk = self._to_string_grid_manip(
            whitelist=whitelist,
            include_description=include_description,
            skip_kind=skip_kind,
            skip_parameters=skip_parameters,
            skip_parameter_description=skip_parameter_description,
            skip_cues=skip_cues,
            skip_implementation=skip_implementation,
            usage_threshold=usage_threshold,
            show_other_concepts=show_other_concepts,
            indentation=indentation,
            filter_concept=filter_concept,
        )
        if blk:
            blocks.append(blk)

        # 4) Other routines --------------------------------------------
        blk = self._to_string_other_routines(
            whitelist=whitelist,
            include_description=include_description,
            skip_kind=skip_kind,
            skip_routine_subtype=skip_routine_subtype,
            skip_parameters=skip_parameters,
            skip_parameter_description=skip_parameter_description,
            skip_cues=True,
            skip_implementation=skip_implementation,
            usage_threshold=usage_threshold,
            show_other_concepts=show_other_concepts,
            indentation=indentation,
            filter_concept=filter_concept,
        )
        if blk:
            blocks.append(blk)

        return "\n\n".join(blocks).rstrip()

    # --------------------------- Sections ----------------------------- #
    def _to_string_structures(
        self,
        *,
        whitelist: set | None,
        include_description: bool,
        skip_kind: bool,
        skip_routine_subtype: bool,
        skip_parameters: bool,
        skip_parameter_description: bool,
        skip_cues: bool,
        skip_implementation: bool,
        usage_threshold: int,
        show_other_concepts: bool,
        indentation: int,
        filter_concept: Callable[[Concept], bool] | None,
    ) -> str:
        all_names = self.categories.get("structure", [])
        names_in = [n for n in all_names if (not whitelist or n in whitelist)]
        names_out = (
            [n for n in all_names if whitelist and n not in whitelist]
            if show_other_concepts
            else []
        )
        if not names_in and not names_out:
            return ""

        full_render: list[str] = []
        low_usage_names: list[str] = []

        for n in names_in:
            c = self.concepts[n]
            if filter_concept and not filter_concept(c):
                continue
            if len(c.used_in) < usage_threshold:
                low_usage_names.append(c.name)
            else:
                full_render.append(
                    c.to_string(
                        include_description=include_description,
                        indentation=indentation,
                        skip_kind=skip_kind,
                        skip_routine_subtype=skip_routine_subtype,
                        skip_parameters=skip_parameters,
                        skip_parameter_description=skip_parameter_description,
                        skip_cues=skip_cues,
                        skip_implementation=skip_implementation,
                    )
                )

        lines: list[str] = ["## structure concepts", *full_render]
        if low_usage_names:
            lines.append(
                f"- lower usage concepts: [{', '.join(sorted(low_usage_names))}]"
            )
        if names_out:
            lines.append(f"- other concepts: [{', '.join(sorted(names_out))}]")
        return "\n".join(lines)

    def _to_string_types(self, *, indentation: int) -> str:
        if not self.custom_types:
            return ""
        ind = " " * indentation
        lines = ["## types"]
        for name, anno in sorted(self.custom_types.items()):
            lines.append(f"{ind}- {name} := {anno}")
        return "\n".join(lines)

    def _to_string_grid_manip(
        self,
        *,
        whitelist: set | None,
        include_description: bool,
        skip_kind: bool,
        skip_parameters: bool,
        skip_parameter_description: bool,
        skip_cues: bool,
        skip_implementation: bool,
        usage_threshold: int,
        show_other_concepts: bool,
        indentation: int,
        filter_concept: Callable[[Concept], bool] | None,
    ) -> str:
        all_grid_names = [
            n
            for n in self.categories.get("routine", [])
            if self.concepts[n].routine_subtype == "grid manipulation"
        ]
        grid_in = [n for n in all_grid_names if (not whitelist or n in whitelist)]
        grid_out = (
            [n for n in all_grid_names if whitelist and n not in whitelist]
            if show_other_concepts
            else []
        )
        if not grid_in and not grid_out:
            return ""

        full_render: list[str] = []
        low_usage_names: list[str] = []

        for n in grid_in:
            c = self.concepts[n]
            if filter_concept and not filter_concept(c):
                continue
            if len(c.used_in) < usage_threshold:
                low_usage_names.append(c.name)
            else:
                full_render.append(
                    c.to_string(
                        include_description=include_description,
                        indentation=indentation,
                        skip_kind=skip_kind,
                        skip_routine_subtype=True,  # suppressed inside header
                        skip_parameters=skip_parameters,
                        skip_parameter_description=skip_parameter_description,
                        skip_cues=skip_cues,
                        skip_implementation=skip_implementation,
                    )
                )

        lines: list[str] = ["## grid manipulation routines", *full_render]
        if low_usage_names:
            lines.append(
                f"- lower usage concepts: [{', '.join(sorted(low_usage_names))}]"
            )
        if grid_out:
            lines.append(f"- other concepts: [{', '.join(sorted(grid_out))}]")
        return "\n".join(lines)

    def _to_string_other_routines(
        self,
        *,
        whitelist: set | None,
        include_description: bool,
        skip_kind: bool,
        skip_routine_subtype: bool,
        skip_parameters: bool,
        skip_parameter_description: bool,
        skip_cues: bool,
        skip_implementation: bool,
        usage_threshold: int,
        show_other_concepts: bool,
        indentation: int,
        filter_concept: Callable[[Concept], bool] | None,
    ) -> str:
        routines_all = [
            self.concepts[n]
            for n in self.categories.get("routine", [])
            if self.concepts[n].routine_subtype != "grid manipulation"
        ]

        # Partition into whitelisted and other buckets preserving subtype groupings
        routines_in = [
            c for c in routines_all if (not whitelist or c.name in whitelist)
        ]
        routines_out_names = (
            [c.name for c in routines_all if whitelist and c.name not in whitelist]
            if show_other_concepts
            else []
        )

        if not routines_in and not routines_out_names:
            return ""

        buckets: dict[str, list[Concept]] = defaultdict(list)
        for c in routines_in:
            key = c.routine_subtype or "misc"
            buckets[key].append(c)

        lines: list[str] = ["## other routines"]
        low_usage_overall: list[str] = []

        for subtype in sorted(buckets.keys()):
            lines.append(f"### {subtype}")
            for c in buckets[subtype]:
                if filter_concept and not filter_concept(c):
                    continue
                if len(c.used_in) < usage_threshold:
                    low_usage_overall.append(c.name)
                else:
                    lines.append(
                        c.to_string(
                            include_description=include_description,
                            indentation=indentation,
                            skip_kind=skip_kind,
                            skip_routine_subtype=skip_routine_subtype,
                            skip_parameters=skip_parameters,
                            skip_parameter_description=skip_parameter_description,
                            skip_cues=skip_cues,
                            skip_implementation=skip_implementation,
                        )
                    )

        if low_usage_overall:
            lines.append(
                f"- lower usage concepts: [{', '.join(sorted(low_usage_overall))}]"
            )
        if routines_out_names:
            lines.append(f"- other concepts: [{', '.join(sorted(routines_out_names))}]")
        return "\n".join(lines)

    # ----------------------------------------------------------------- #
    #                   Model-output ingest & initialization             #
    # ----------------------------------------------------------------- #
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

    def initialize_solutions(self, mapping: dict[str, dict]) -> None:
        for pid, ann in mapping.items():
            self.solutions[pid] = ProblemSolution(
                problem_id=pid,
                solution=ann.get("solution"),
                summary=ann.get("summary"),
                pseudocode=ann.get("pseudocode"),
            )

    def initialize_from_annotations(self, annotations: dict[str, dict]) -> None:
        for pid, ann in annotations.items():
            self.write_solution(pid, None, ann)
            for concept_ann in ann.get("concepts", []):
                self.write_concept(pid, concept_ann)

    # ----------------------------------------------------------------- #
    #                           Persistence                              #
    # ----------------------------------------------------------------- #
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
