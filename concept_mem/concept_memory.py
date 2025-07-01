import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import orjson
import yaml

import textwrap

from concept_mem.utils import extract_yaml_block, read_json, read_yaml
from concept_mem.constants import REPO_ROOT


logger = logging.getLogger(__name__)


# new format:
@dataclass
class Concept:
    name: str
    description: str | None = None
    # puzzle_id -> use case description
    use_cases: dict[str, str | None] = field(default_factory=dict)
    # relevance_cues: str | None = None
    relevance_cues: dict[str, str | None] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)

    # python helper functions that operationalise the concept
    helper_routines: list[str] = field(default_factory=list)

    def to_string(
        self,
        include_cues: bool = True,
        include_notes: bool = True,
        include_case_info: bool = False,
        include_helpers: bool = False,
        indentation: int = 0,
    ) -> str:
        lines: list[str] = [
            f"- concept: {self.name}",
            f"  description: {self.description or '—'}",
        ]
        if include_cues and self.relevance_cues:
            formatted_cues = self._format_per_puzzle_field(
                "relevance_cues", "relevance cues", indent=2
            )
            lines.append(formatted_cues)
        if include_helpers and self.helper_routines:
            lines.append("  helper routines:")
            for r in self.helper_routines:
                lines.append(f"  - `{r.splitlines()[0][:72]}`...")
        if include_notes and self.notes:
            lines.append(self._format_per_puzzle_field("notes", indent=2))
        if include_case_info and self.use_cases:
            lines.append("  use_cases:")
            for puzzle_id, uc in self.use_cases.items():
                lines.append(f"  - {puzzle_id}: {uc}")

        result = "\n".join([line for line in lines if line is not None])
        if indentation:
            result = textwrap.indent(result, " " * indentation)
        return result

    def _format_per_puzzle_field(
        self,
        field_name: str,
        display_name: str | None,
        indent: int = 2,
    ) -> str | None:
        seen = set()
        val = getattr(self, field_name, {})
        for _, values in val.items():
            if not values:
                continue
            for value in values:
                try:
                    seen.add(str(value).strip())
                except Exception as e:
                    logger.error(
                        f"Error processing {field_name} for concept {self.name}: value={value}, error={e}"
                    )
                    continue
        if not seen:
            return None
        lines = [f"{display_name or field_name}:"]
        for item in seen:
            lines.append(f"- {item}")
        result = "\n".join(lines)
        return textwrap.indent(result, " " * indent)


class ConceptMemory:
    def __init__(self):
        self.concepts: dict[str, Concept] = {}

    def add_concept(self, puzzle_id: str, concept: dict) -> None:
        name = concept["concept"]
        if name in self.concepts:
            entry = self.concepts[name]
        else:
            entry = Concept(name=name)
            self.concepts[name] = entry

        new_description = concept.get("description", None)
        if new_description and new_description.strip():
            entry.description = new_description.strip()
        new_cues = concept.get("relevance_cues", None)
        if new_cues and new_cues.strip():
            entry.relevance_cues = new_cues.strip()

        entry.use_cases[puzzle_id] = concept.get("use_case", None)
        # entry.notes[puzzle_id] = concept.get("notes", [])
        new_notes = []
        model_notes = concept.get("notes", [])
        for note in model_notes:
            try:
                processed_note = str(note).strip()
                new_notes.append(processed_note)
            except Exception as e:
                logger.error(f"Error processing note {note} for concept {name}: {e}")
                continue
        entry.notes[puzzle_id] = new_notes
        entry.helpers = concept.get("helper_routines", [])

    def to_string(
        self,
        include_cues: bool = True,
        include_notes: bool = True,
        include_case_info: bool = False,
        include_helpers: bool = False,
    ) -> str:
        # TODO: figure out how to include pseudocode solutions
        components = []
        for concept in self.concepts.values():
            components.append(
                concept.to_string(
                    include_cues=include_cues,
                    include_notes=include_notes,
                    include_case_info=include_case_info,
                    include_helpers=include_helpers,
                )
            )
        return "\n".join(components)

    def update_from_model_output(
        self,
        puzzle_id: str,
        model_output: str,
    ) -> None:
        # extract from triple backticks markdown block
        yaml_string = extract_yaml_block(model_output)
        # parse the yaml string
        try:
            parsed_output = yaml.safe_load(yaml_string)
            concepts = parsed_output.get("concepts", [])
            for concept in concepts:
                self.add_concept(puzzle_id, concept)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML (puzzle: {puzzle_id}): {e}")
            return
        except KeyError as e:
            print(f"KeyError: {e}")
            print(model_output)
            return

    def initialize_from_annotations(
        self,
        hand_annotations: dict[str, dict],
    ) -> None:
        """
        Initialize the concept memory from hand annotations.
        """
        for puzzle_id, annotation in hand_annotations.items():
            concepts = annotation.get("concepts", [])
            for concept in concepts:
                self.add_concept(puzzle_id, concept)

    def save_to_file(self, path: Path) -> None:
        serializable_concepts = {
            name: asdict(concept) for name, concept in self.concepts.items()
        }
        serialized_data = orjson.dumps(
            serializable_concepts, option=orjson.OPT_INDENT_2
        )
        with path.open("wb") as f:
            f.write(serialized_data)

    def load_from_file(self, path: Path) -> None:
        path_string = str(path)
        if not path_string.startswith("/"):
            path = REPO_ROOT / path_string
        with path.open("rb") as f:
            concept_data = orjson.loads(f.read())
        self.concepts = {name: Concept(**data) for name, data in concept_data.items()}


def merge_hand_and_machine_annotations(
    hand_annotations: dict | Path,
    machine_annotations: dict | Path,
    output_file: Path | None = None,
) -> dict[str, dict[str, str | list]]:
    merged = {}
    if isinstance(hand_annotations, Path):
        hand_annotations = read_yaml(hand_annotations)
    if isinstance(machine_annotations, Path):
        machine_annotations = read_json(machine_annotations)
        parsed_annotations = {}
        for puzzle_id, annotation in machine_annotations.items():
            if isinstance(annotation, str):
                yaml_string = extract_yaml_block(annotation)
                try:
                    yaml_data = yaml.safe_load(yaml_string)
                    parsed_annotations[puzzle_id] = yaml_data
                except yaml.YAMLError as e:
                    logger.info(f"Error parsing YAML for puzzle {puzzle_id}: {e}")
                    continue
            else:
                parsed_annotations[puzzle_id] = annotation
        machine_annotations = parsed_annotations

    for puzzle_id, seed_data in machine_annotations.items():
        annotation = {
            "general": seed_data.get("general", ""),
            "specific": seed_data.get("specific", seed_data.get("general", "")),
            "pseudocode": seed_data.get("pseudocode", []),
        }
        merged[puzzle_id] = annotation
    for puzzle_id, seed_data in hand_annotations.items():
        general = seed_data.get("general", "")
        merged[puzzle_id] = {
            "general": general,
            "specific": seed_data.get("specific", general),
            "pseudocode": seed_data.get("pseudocode", []),
        }
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # use orjson dump
        with open(output_file, "wb") as f:
            f.write(orjson.dumps(merged, option=orjson.OPT_INDENT_2))
    return merged


# local updated version (23.06.2025)
"""
import logging
from pathlib import Path
from dataclasses import asdict, dataclass, field
from detective.utils import read_json, write_json, read_yaml, extract_yaml_block


logger = logging.getLogger(__name__)


@dataclass
class Concept:
    title: str
    # concept notes:
    # - mapping puzzle_id -> note
    #   to retain note providence
    notes: dict[str, list[str]] = field(default_factory=dict)
    # use cases:
    # - mapping puzzle_id -> use case description
    use_cases: dict[str, str] = field(default_factory=dict)

    def to_string(self, include_notes: bool = True) -> str:
        components = [f"- concept: {self.title}"]
        if include_notes:
            formatted_notes = self._format_notes()
            if formatted_notes:
                components.append(formatted_notes)
        return "\n".join(components)

    def _format_notes(self, indent_size: int = 2) -> str | None:
        note_set = set()
        for puzzle_notes in self.notes.values():
            if not puzzle_notes:
                continue
            for note in puzzle_notes:
                note_set.add(note.strip())
        if not note_set:
            return None
        indent = " " * indent_size
        components = [indent + "notes:"]
        for note in note_set:
            components.append(f"- {note}")
        return "\n".join(indent + line for line in components)


class ConceptMemory:
    def __init__(self):
        self.concepts: dict[str, Concept] = {}

    def get_concept(self, title: str) -> Concept | None:
        return self.concepts.get(title, None)

    def add_concept(
        self, puzzle_id: str, title: str, notes: list[str], use_case: str | None
    ) -> None:
        if title in self.concepts:
            concept = self.concepts[title]
        else:
            concept = Concept(title=title)
            self.concepts[title] = concept
        puzzle_notes = concept.notes.setdefault(puzzle_id, [])
        puzzle_notes.extend(notes)
        concept.use_cases[puzzle_id] = use_case or ""

    def load_from_file(self, file_path: Path | str) -> None:
        data = read_json(file_path)
        for title, concept_data in data.items():
            concept = Concept(title=title)
            concept.notes = concept_data.get("notes", {})
            concept.use_cases = concept_data.get("use_cases", {})
            self.concepts[title] = concept

    def save_to_file(self, file_path: Path | str) -> None:
        data = {title: asdict(concept) for title, concept in self.concepts.items()}
        write_json(data, file_path)
        logger.info(f"concept memory saved to {file_path}")

    def initialize_from_annotations(self, file_path: Path | str) -> None:
        annotation_data = read_yaml(file_path)
        for puzzle_id, annotation in annotation_data.items():
            if "concepts" not in annotation:
                continue
            self.update_memory_from_annotation(puzzle_id, annotation)

    def update_memory_from_model_output(
        self, puzzle_id: str, model_output: str
    ) -> None:
        yaml_block = extract_yaml_block(model_output)
        try:
            yaml_data = read_yaml(yaml_block)
        except Exception:
            logger.info(
                f"Failed to parse YAML block from model output for puzzle {puzzle_id}. "
            )
        self.update_memory_from_annotation(puzzle_id, yaml_data)

    def update_memory_from_annotation(self, puzzle_id: str, annotation: dict) -> None:
        annotated_concepts = annotation["concepts"]
        for annotated_concept in annotated_concepts:
            title = annotated_concept["concept"]
            notes = annotated_concept.get("notes", [])
            use_case = annotated_concept.get("use_case", "")
            self.add_concept(puzzle_id, title, notes, use_case)

    def to_string(self, include_notes: bool = True) -> str:
        components = []
        for concept in self.concepts.values():
            components.append(concept.to_string(include_notes=include_notes))
        return "\n".join(components)

"""
