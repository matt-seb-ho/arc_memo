import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from llmplus import GenerationConfig, LLMClient

from concept_mem.data.arc_agi import Problem
from concept_mem.utils import extract_yaml_block, read_json, run_llm_job, write_json

logger = logging.getLogger(__name__)


LESSON_ENTRY_TEMPLATE = """\
lesson {num}.
- situation: {situation}
- suggestion: {situation}"""


SELECT_TOP_K_TEMPLATE = """\
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule.

We have a list of puzzle solving "lessons" or "rules" that provide a suggestion of how to solve the puzzle given a certain situation.

### Instructions
We will provide you with a numbered list of lessons and a puzzle description.
- Your task is to identify the most relevant {top_k} lessons for the given puzzle.
- Please output your final selection as a list of lesson numbers in a markdown yaml block, e.g.
```yaml
- 18
- 77
- 19
```

### Lessons
{concept_list}

### Puzzle Description
{description} 
"""


@dataclass
class LessonConcept:
    # concept payload
    situation: str
    suggestion: str

    # puzzle ID from which this concept was derived
    source: str


class LessonConceptMemory:
    def __init__(self):
        self.lessons: list[LessonConcept] = []

    def add_lesson(
        self,
        situation: str,
        suggestion: str,
        source: str,
    ) -> None:
        self.lessons.append(
            LessonConcept(
                situation=situation,
                suggestion=suggestion,
                source=source,
            )
        )

    async def select_concepts(
        self,
        puzzles: list[Problem],
        descriptions: list[str],
        top_k: int,
        llm_client: LLMClient,
        model: str,
        gen_cfg: GenerationConfig,
        output_dir: Path,
    ) -> dict[str, str]:
        """
        return
        - concepts: dict[str, str] = maps puzzle ID to formatted lesson concepts hint
        """
        metadata = [p.uid for p in puzzles]
        prompts = self.build_selection_prompts(puzzles, descriptions, top_k)
        model_output = await run_llm_job(
            prompts=prompts,
            metadata=metadata,
            llm_client=llm_client,
            model=model,
            gen_cfg=gen_cfg,
            output_dir=output_dir,
        )

        selected_idxs: dict[str, list[int]] = {}
        selected_concepts: dict[str, str] = {}
        parsing_error_count = 0
        for puzzle_id, completions in zip(metadata, model_output):
            completion = completions[0]
            try:
                concept_numbers = self._parse_top_k_yaml_list(completion)
                formatted_hint = self.format_retrieved_lesson_concept(concept_numbers)
                selected_idxs[puzzle_id] = concept_numbers
                selected_concepts[puzzle_id] = formatted_hint
            except Exception as e:
                logger.error(f"Error parsing completion for puzzle {puzzle_id}: {e}")
                parsing_error_count += 1
                continue
        logger.info(
            f"encountered {parsing_error_count} parsing errors while selecting concepts for {len(puzzles)} puzzles"
        )
        return selected_concepts

    def build_selection_prompts(
        self,
        puzzles: list[Problem],
        descriptions: list[str],
        top_k: int = 3,
    ) -> list[str]:
        concept_list = self._prep_concept_list_for_selection()
        prompts = []
        for puzzle, description in zip(puzzles, descriptions):
            prompt = SELECT_TOP_K_TEMPLATE.format(
                top_k=top_k,
                concept_list=concept_list,
                description=description,
            )
            prompts.append(prompt)
        return prompts

    def format_retrieved_lesson_concept(
        self,
        selected_idxs: list[int],
    ) -> str:
        components = []
        for selected_idx in selected_idxs:
            # -1 because when formatted in the prompt
            # the lesson numbers start from 1, but we use 0-based indexing
            lesson = self.lessons[selected_idx - 1]
            formatted = (
                f"- situation: {lesson.situation}\n  suggestion: {lesson.suggestion}"
            )
            components.append(formatted)
        return "\n".join(components)

    def initialize_from_abstraction_output(
        self,
        abstraction_output: dict,
    ) -> None:
        for puzzle_id, puzzle_lessons in abstraction_output.items():
            for lesson in puzzle_lessons:
                self.lessons.append(
                    LessonConcept(
                        situation=lesson["situation"],
                        suggestion=lesson["suggestion"],
                        source=puzzle_id,
                    )
                )

    def save_to_file(self, file_path: Path) -> None:
        serializable_form = [asdict(lesson) for lesson in self.lessons]
        write_json(serializable_form, file_path)
        logger.info(f"Saved lesson concepts to {file_path}")

    def load_from_file(self, file_path: Path) -> None:
        data = read_json(file_path)
        for lesson_dict in data:
            self.lessons.append(LessonConcept(**lesson_dict))
        logger.info(f"Loaded lesson concepts from {file_path}")

    def _prep_concept_list_for_selection(self) -> str:
        concept_entries = []
        for num, lesson in enumerate(self.lessons, start=1):
            entry = LESSON_ENTRY_TEMPLATE.format(
                num=num,
                situation=lesson.situation,
                suggestion=lesson.suggestion,
            )
            concept_entries.append(entry)
        concept_list = "\n".join(concept_entries)
        return concept_list

    def _parse_top_k_yaml_list(self, yaml_block: str) -> list[int]:
        # expected format:
        # ```yaml
        # - 18
        # - 77
        # ...
        # ```
        # step 1: remove markdown delimiters
        yaml_string = extract_yaml_block(yaml_block) or yaml_block
        # step 2: parse yaml string
        yaml_data = yaml.safe_load(yaml_string)
        top_k_list = []
        assert isinstance(yaml_data, list)
        for item in yaml_data:
            try:
                num = int(item)
                top_k_list.append(num)
            except Exception as e:
                logger.error(
                    f"Could not parse selected concept number: {item}, error: {e}"
                )
                continue
        return top_k_list
