import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import orjson
import yaml

from concept_mem.concept import Concept
from concept_mem.constants import REPO_ROOT
from concept_mem.utils import extract_yaml_block

logger = logging.getLogger(__name__)


@dataclass
class ProblemSolution:
    problem_id: str
    solution: str | None = None
    summary: str | None = None
    pseudocode: list[str | dict[str, str]] = field(default_factory=list)


class ConceptMemory:
    def __init__(self):
        self.concepts: dict[str, Concept] = {}
        self.solutions: dict[str, ProblemSolution] = {}

    def write_concept(self, puzzle_id: str, annotation: dict) -> None:
        # only strictly required field is `concept`
        if "concept" not in annotation:
            logger.info(
                f"Skipping annotation for puzzle {puzzle_id} as it does not contain a 'concept' key."
            )
            return

        # fetch entry from memory or create a new one
        concept_name = annotation["concept"]
        if concept_name in self.concepts:
            entry = self.concepts[concept_name]
        else:
            # new entry
            entry = Concept(name=concept_name)
            self.concepts[concept_name] = entry

        # update the entry with the annotation
        entry.update(problem_id=puzzle_id, annotation=annotation)

    def write_solution(
        self,
        puzzle_id: str,
        solution: str | None,
        annotation: dict,
    ) -> None:
        summary = annotation.get("summary", None)
        pseudocode = annotation.get("pseudocode", [])
        self.solutions[puzzle_id] = ProblemSolution(
            problem_id=puzzle_id,
            solution=solution,
            summary=summary,
            pseudocode=pseudocode,
        )

    def to_string(
        self,
        include_description: bool = True,
        include_parents: bool = True,
        include_associates: bool = True,
        include_cues: bool = True,
        include_notes: bool = True,
        indentation: int = 0,
    ) -> str:
        # TODO: figure out how to include pseudocode solutions
        components: list[str] = []
        problem_usage_info: dict[str, str] = {
            problem_id: problem_solution.summary
            for problem_id, problem_solution in self.solutions.items()
        }
        for concept in self.concepts.values():
            components.append(
                concept.to_string(
                    include_description=include_description,
                    include_parents=include_parents,
                    include_associates=include_associates,
                    include_cues=include_cues,
                    include_notes=include_notes,
                    problem_usage_info=problem_usage_info,
                    indentation=indentation,
                )
            )
        return "\n".join(components)

    def update_from_model_output(
        self,
        puzzle_id: str,
        solve_output: str,
        abstract_output: str,
    ) -> None:
        # extract from triple backticks markdown block
        yaml_string = extract_yaml_block(abstract_output)

        # parse the yaml string
        try:
            parsed_output = yaml.safe_load(yaml_string)
            # update solution records
            self.write_solution(
                puzzle_id,
                solve_output,
                parsed_output,
            )
            # update concept records
            concepts = parsed_output.get("concepts", [])
            for concept in concepts:
                self.write_concept(puzzle_id, concept)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML (puzzle: {puzzle_id}): {e}")
            return
        except KeyError as e:
            logger.info(f"KeyError: {e}, {puzzle_id} - {abstract_output}")
            return

    def initialize_from_annotations(
        self,
        annotations: dict[str, dict],
    ) -> None:
        """
        Initialize the concept memory from hand annotations.
        """
        for puzzle_id, annotation in annotations.items():
            # TODO: add full solution code
            self.write_solution(
                puzzle_id,
                None,
                annotation,
            )
            concepts = annotation.get("concepts", [])
            for concept in concepts:
                self.write_concept(puzzle_id, concept)

    def save_to_file(self, path: Path) -> None:
        serializable_concepts = {
            name: asdict(concept) for name, concept in self.concepts.items()
        }
        serializable_solutions = {
            name: asdict(solution) for name, solution in self.solutions.items()
        }
        serializable_mem = {
            "concepts": serializable_concepts,
            "solutions": serializable_solutions,
        }
        serialized_data = orjson.dumps(serializable_mem, option=orjson.OPT_INDENT_2)
        with path.open("wb") as f:
            f.write(serialized_data)

    def load_from_file(self, path: Path) -> None:
        # read data from file
        path_string = str(path)
        if not path_string.startswith("/"):
            path = REPO_ROOT / path_string
        with path.open("rb") as f:
            mem_data = orjson.loads(f.read())

        # reformat as dataclass objects
        self.concepts = {
            name: Concept(**data) for name, data in mem_data["concepts"].items()
        }
        self.solutions = {
            problem_id: ProblemSolution(**data)
            for problem_id, data in mem_data["solutions"].items()
        }


# def merge_hand_and_machine_annotations(
#     hand_annotations: dict | Path,
#     machine_annotations: dict | Path,
#     output_file: Path | None = None,
# ) -> dict[str, dict[str, str | list]]:
#     merged = {}
#     if isinstance(hand_annotations, Path):
#         hand_annotations = read_yaml(hand_annotations)
#     if isinstance(machine_annotations, Path):
#         machine_annotations = read_json(machine_annotations)
#         parsed_annotations = {}
#         for puzzle_id, annotation in machine_annotations.items():
#             if isinstance(annotation, str):
#                 yaml_string = extract_yaml_block(annotation)
#                 try:
#                     yaml_data = yaml.safe_load(yaml_string)
#                     parsed_annotations[puzzle_id] = yaml_data
#                 except yaml.YAMLError as e:
#                     logger.info(f"Error parsing YAML for puzzle {puzzle_id}: {e}")
#                     continue
#             else:
#                 parsed_annotations[puzzle_id] = annotation
#         machine_annotations = parsed_annotations

#     for puzzle_id, seed_data in machine_annotations.items():
#         annotation = {
#             "general": seed_data.get("general", ""),
#             "specific": seed_data.get("specific", seed_data.get("general", "")),
#             "pseudocode": seed_data.get("pseudocode", []),
#         }
#         merged[puzzle_id] = annotation
#     for puzzle_id, seed_data in hand_annotations.items():
#         general = seed_data.get("general", "")
#         merged[puzzle_id] = {
#             "general": general,
#             "specific": seed_data.get("specific", general),
#             "pseudocode": seed_data.get("pseudocode", []),
#         }
#     if output_file is not None:
#         output_file.parent.mkdir(parents=True, exist_ok=True)
#         # use orjson dump
#         with open(output_file, "wb") as f:
#             f.write(orjson.dumps(merged, option=orjson.OPT_INDENT_2))
#     return merged
