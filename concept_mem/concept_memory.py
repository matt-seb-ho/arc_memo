import logging
from collections import defaultdict
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
    pseudocode: str | None = None


class ConceptMemory:
    def __init__(self):
        # self.concepts: dict[str, Concept] = {}
        self.concepts: dict[str, Concept] = {}
        self.categories: dict[str, list[str]] = defaultdict(list)
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
            entry.update(problem_id=puzzle_id, annotation=annotation)
        else:
            # new entry
            # - update category buckets
            try:
                entry_type = annotation["kind"]
                entry = Concept(
                    name=concept_name,
                    kind=annotation["kind"],
                    description=annotation.get("description", None),
                    for_param=annotation.get("for_param", None),
                )
                entry.update(problem_id=puzzle_id, annotation=annotation)
                self.concepts[concept_name] = entry
                self.categories[entry_type].append(concept_name)
            except KeyError as e:
                logger.info(f"Missing field for new concept '{concept_name}': {e}")
            except Exception as e:
                logger.info(f"Error creating concept entry for {concept_name}: {e}")

    def write_solution(
        self,
        puzzle_id: str,
        solution: str | None,
        annotation: dict,
    ) -> None:
        summary = annotation.get("summary", None)
        pseudocode = annotation.get("pseudocode", None)
        self.solutions[puzzle_id] = ProblemSolution(
            problem_id=puzzle_id,
            solution=solution,
            summary=summary,
            pseudocode=pseudocode,
        )

    def to_string(
        self,
        categories: list[str] | None = None,
        include_description: bool = True,
        parameter_format: str = "none",
        include_usage: bool = False,
        indentation: int = 0,
    ) -> str:
        # TODO: figure out how to include pseudocode solutions
        components: list[str] = []
        if include_usage:
            problem_usage_info = {
                problem_id: problem_solution.summary
                for problem_id, problem_solution in self.solutions.items()
            }
        else:
            problem_usage_info = None
        if categories is None:
            categories = [
                "term definition",
                "grid manipulation",
                "intermediate operation",
                "parameter selection",
            ]
        for category in categories:
            concept_names = self.categories.get(category, [])
            if not concept_names:
                continue
            components.append(f"## {category} concepts")
            if category == "parameter selection":
                components.append(
                    self._format_selection_concepts(
                        include_description=include_description,
                        parameter_format=parameter_format,
                        problem_usage_info=problem_usage_info,
                        indentation=indentation,
                    )
                )
                continue
            for concept_name in concept_names:
                concept = self.concepts[concept_name]
                components.append(
                    concept.to_string(
                        include_description=include_description,
                        parameter_format=parameter_format,
                        problem_usage_info=problem_usage_info,
                        indentation=indentation,
                    )
                )
            # spacer line
            components.append("")

        concept_mem_string = "\n".join(components)
        return concept_mem_string

    def update_from_model_output(
        self,
        puzzle_id: str,
        abstract_output: str,
    ) -> None:
        # extract from triple backticks markdown block
        yaml_string = extract_yaml_block(abstract_output)

        # parse the yaml string
        try:
            parsed_output = yaml.safe_load(yaml_string)
            # update solution records
            # self.write_solution(
            #     puzzle_id,
            #     solve_output,
            #     parsed_output,
            # )
            # update concept records
            # concepts = parsed_output.get("concepts", [])
            if not isinstance(parsed_output, list):
                logger.info(
                    f"Expected a list of concepts for puzzle {puzzle_id}, "
                    f"got {type(parsed_output)}. Skipping update."
                )
                return
            for concept in parsed_output:
                self.write_concept(puzzle_id, concept)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML (puzzle: {puzzle_id}): {e}")
            return
        except KeyError as e:
            logger.info(f"KeyError: {e}, {puzzle_id} - {abstract_output}")
            return

    def initialize_solutions(
        self,
        solutions: dict[str, dict],
    ) -> None:
        for puzzle_id, annotation in solutions.items():
            problem_solution = ProblemSolution(
                problem_id=puzzle_id,
                solution=annotation.get("solution", None),
                summary=annotation.get("summary", None),
                pseudocode=annotation.get("pseudocode", None),
            )
            self.solutions[puzzle_id] = problem_solution

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
        for concept_name, concept in self.concepts.items():
            self.categories[concept.kind].append(concept_name)

    def _format_selection_concepts(
        self,
        include_description: bool = True,
        parameter_format: str = "none",
        problem_usage_info: dict[str, str] | None = None,
        indentation: int = 0,
    ) -> str:
        ps_concepts = [
            self.concepts[name]
            for name in self.categories.get("parameter selection", [])
        ]
        for_param_groups = defaultdict(list)
        for concept in ps_concepts:
            for_param_groups[concept.for_param].append(concept)
        components = []
        for param, concepts in for_param_groups.items():
            components.append(f"for parameter `{param}`:")
            for concept in concepts:
                components.append(
                    concept.to_string(
                        include_description=include_description,
                        include_for_param=False,
                        parameter_format=parameter_format,
                        problem_usage_info=problem_usage_info,
                        indentation=indentation,
                    )
                )
        return "\n".join(components)


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
