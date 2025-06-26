import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import orjson
import yaml

from concept_mem.utils import extract_yaml_block, read_json, read_yaml
from concept_mem.constants import REPO_ROOT


logger = logging.getLogger(__name__)


# new format:
@dataclass
class Concept:
    name: str
    # puzzle_id -> use case description
    use_cases: dict[str, str | None] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def to_string(
        self,
        include_notes: bool = True,
        include_case_info: bool = False,
        indentation: int = 0,
    ) -> str:
        components = [f"- concept: {self.name}"]
        if include_notes and self.notes:
            note_set = set()
            for note_list in self.notes.values():
                if note_list is None:
                    continue
                for note in note_list:
                    if isinstance(note, dict):
                        for k, v in note.items():
                            note_set.add(f"{k}: {v.strip}")
                    elif isinstance(note, str):
                        note_set.add(note.strip())
                    else:
                        logger.warning(
                            f"Unexpected note type: {type(note)} for concept {self.name}. Note: {note}"
                        )
            if note_set:
                components.append("  notes:")
                for note in note_set:
                    components.append(f"  - {note}")
        if include_case_info and self.use_cases:
            components.append("  concept usages:")
            for puzzle_id, use_case in self.use_cases.items():
                components.append(f"  - {puzzle_id}: {use_case}")
        if indentation:
            indent = " " * indentation
            components = [f"{indent}{line}" for line in components]
        text = "\n".join(components)
        return text


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
        entry.use_cases[puzzle_id] = concept.get("use_case", None)
        entry.notes[puzzle_id] = concept.get("notes", [])

    def to_string(
        self,
        include_notes: bool = True,
        include_case_info: bool = False,
    ) -> str:
        # TODO: figure out how to include pseudocode solutions
        components = []
        for concept in self.concepts.values():
            components.append(
                concept.to_string(
                    include_notes=include_notes,
                    include_case_info=include_case_info,
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
