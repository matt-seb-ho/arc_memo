from typing import Callable, Literal

import numpy as np
import yaml

from concept_mem.concept_memory import Concept, ConceptMemory
from concept_mem.constants import MAX_LINE_WIDTH
from concept_mem.data.arc_agi import Problem
from concept_mem.evaluation.solution_tree import SolutionStep, SolutionThread

# ------------------------------------------------------------------------------
# prompt constants
# ------------------------------------------------------------------------------

# -- system prompts ------------------------------------------------------------
# DEFAULT_SYSTEM_PROMPT = "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."
DEFAULT_SYSTEM_PROMPT_IND = "You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."
DEFAULT_SYSTEM_PROMPT_TRAN = "You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions."
CONCISE_SYSTEM_PROMPT = "You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions. Be concise and think through puzzles without being unnecessarily verbose."
SYSTEM_PROMPTS = {
    "default": DEFAULT_SYSTEM_PROMPT_IND,
    "concise": CONCISE_SYSTEM_PROMPT,
}

# -- section intro -------------------------------------------------------------
ARC_INTRO = """\
### Introduction
Given a puzzle containing input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Within a puzzle, each pair follows the same transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and is often the background color.
"""

ICL_SECTION_INTRO = 'We will provide a series of puzzles and their solutions to help you understand the problem. You will then be asked to solve a different puzzle ("Your Puzzle").'

EXAMPLE_GRIDS_INTRO = "Here are the input and output grids for the reference examples:"

CONCEPT_SECTION_INTRO = """\
### Concepts
# The following concepts may be relevant: {concepts}"""

LIBRARY_TEMPLATE = """### Common Library
Here is the common library function signature and docstring that you can use to solve the problem (skipping the implementation for brevity):
```python
{common_lib}
```"""

DESCRIPTION_TEMPLATE = """\
### External Puzzle Description
We queried other sources for descriptions of what they observe/speculate about this puzzle. These external sources are not authoritative (they may be wrong or incomplete), but they may provide useful context or insights. Here is the description(s):
{description}"""

# -- hint templates ------------------------------------------------------------
HINT_TEMPLATE_MIN = """\
### Hints
{hints}
"""

HINT_TEMPLATE_SELECTED = """\
### Hints
We distilled some lessons or takeaways from previously solved ARC puzzles. Here are some lessons we selected that may or may not be relevant to this puzzle:
{hints}
"""

HINT_TEMPLATE_ALL = """\
### Hints
We distilled some lessons or takeaways from previously solved ARC puzzles. Each lesson is formatted with a "situation" component describing when to apply this lesson, and a "suggestion" component describing what might be a good idea to do in this situation. Here are the lessons:
{hints}
"""

HINT_TEMPLATE_ANNOTATED_CONCEPT_LIST = """\
### Hints
We distilled some lessons and concepts from previously solved puzzles. Please note 2 special classes of concepts:
1. guide object: in a number of puzzles, there are "guide objects" in input grids which are used as a reference that informs output constructs. The guide object could indicate color, shape, position, directions, etc.
2. criteria: in a number of puzzles, there are conditional operations. These criteria can be thought of as predicates that help with these conditional operations.
Here are the lessons and concepts:
{hints}"""

HINT_TEMPLATE_ABSTRACT = """\
### Takeaways from Previously Solved Puzzles
Constructing a program solution requires choosing operations and their parameters. We recorded some grid manipulation, parameter selection, and other helper routines that were used in previously solved puzzles. Here are the takeaways:
{hints}"""

HINT_TEMPLATE_CURR3 = """\
### Takeaways from Previously Solved Puzzles
Constructing a program solution requires choosing operations and their parameters. We recorded some grid structures and routines that were used in previously solved puzzles. Here are the takeaways:
{hints}"""

HINT_TEMPLATES = {
    "basic": HINT_TEMPLATE_MIN,
    "default": HINT_TEMPLATE_MIN,
    "min": HINT_TEMPLATE_MIN,
    "selected": HINT_TEMPLATE_SELECTED,
    "all_hints": HINT_TEMPLATE_ALL,
    "abstract": HINT_TEMPLATE_ABSTRACT,
    "curr3": HINT_TEMPLATE_CURR3,
}

# -- code generation instructions ----------------------------------------------
CODE_INSTR_DEFAULT = """### Instructions
Write a Python function `transform` that converts a given input grid to its corresponding output grid based on the pattern observed in the reference examples. 
- Your function should have signature: `def transform(input_grid: np.ndarray) -> np.ndarray:`
- The input and output grids are 2D numpy arrays of integers.
- Each grid cell is assigned a **color**, represented by an **integer**. These integers **do not have numerical meaning**; they **should not** be used in arithmetic operations. Think of these integers as **labels** rather than numbers.
- Write your final code response in a markdown python code block (be sure to have "```python" in the line before your code and "```" in the line after your code).
- No need to provide examples of the function call, just the function definition is sufficient."""

CODE_INSTR_CHECK = """### Instructions
Write a Python function `transform` that converts a given input grid to its corresponding output grid based on the pattern observed in the reference examples. 
- Your function should have signature: `def transform(input_grid: np.ndarray) -> np.ndarray:`
- The input and output grids are 2D numpy arrays of integers.
- Each grid cell is assigned a **color**, represented by an **integer**. These integers **do not have numerical meaning**; they **should not** be used in arithmetic operations. Think of these integers as **labels** rather than numbers.
- Write your final code response in a markdown python code block (be sure to have "```python" in the line before your code and "```" in the line after your code).
- No need to provide examples of the function call, just the function definition is sufficient.
- Make sure to check your code by running against the reference example grids (i.e. run your code on reference example 1's input grid and compare the result to reference example 1's output grid). If your code does not pass the examples, please fix it before submitting.
"""

CODE_INSTR_CONCISE = """### Instructions
Write a Python function `transform` that converts a given input grid to its corresponding output grid based on the pattern observed in the reference examples. 
- Your function should have signature: `def transform(input_grid: np.ndarray) -> np.ndarray:`
- The input and output grids are 2D numpy arrays of integers.
- Each grid cell is assigned a **color**, represented by an **integer**. These integers **do not have numerical meaning**; they **should not** be used in arithmetic operations. Think of these integers as **labels** rather than numbers.
- Write your final code response in a markdown python code block (be sure to have "```python" in the line before your code and "```" in the line after your code).
- No need to provide examples of the function call, just the function definition is sufficient.
- Please reason concisely-- think your answers through, but try to skip filler words and phrases.
- Try not to be too verbose while thinking."""

CODE_INSTR_CITE = """### Instructions
Write a Python function `transform` that converts a given input grid to its corresponding output grid based on the pattern observed in the reference examples. 
- Your function should have signature: `def transform(input_grid: np.ndarray) -> np.ndarray:`
- The input and output grids are 2D numpy arrays of integers.
- Each grid cell is assigned a **color**, represented by an **integer**. These integers **do not have numerical meaning**; they **should not** be used in arithmetic operations. Think of these integers as **labels** rather than numbers.
- Write your final code response in a markdown python code block (be sure to have "```python" in the line before your code and "```" in the line after your code).
- No need to provide examples of the function call, just the function definition is sufficient.
- Please also indicate which hints were useful in the solution process
  - The hints are numbered, please provide a comma-separated list inside <hint_citations> </hint_citations> tags.
  - For example, if you found hints 1 and 3 useful, you would write: <hint_citations>1, 3</hint_citations>."""

HINT_CITATION_EXTRA_INSTRUCTION = """\
- Please also indicate which hints were useful in the solution process
  - The hints are numbered, please provide a comma-separated list inside <hint_citations> </hint_citations> tags.
  - For example, if you found hints 1 and 3 useful, you would write: <hint_citations>1, 3</hint_citations>."""

CODE_INSTR_DICT = {
    "default": CODE_INSTR_DEFAULT,
    "standard": CODE_INSTR_DEFAULT,  # deprecated key
    "check": CODE_INSTR_CHECK,
    "concise": CODE_INSTR_CONCISE,
    "cite": CODE_INSTR_CITE,
}

# -- long cot concept selection handling ----------------------------------------------
NOTES_SECTION_TEMPLATE = """\
### Notes
Here are notes from the first pass (may contain incomplete thoughts or errors):
{notes}

"""

SECOND_PASS_TEMPLATE = f"""\
{ARC_INTRO}

### Puzzle Grids
{{puzzle_grids}}

### Selected Concepts
- Here are concepts selected from the initial puzzle analysis, annotated with previously solved puzzles that used them. 
- Note two special concept classes:
1. guide object: reference objects informing output constructs' color, shape, position, direction, size, etc.
2. criteria: predicates that help with conditional operations (only executing under certain conditions or executing differently based on conditions).

Concepts List:
{{selected_concepts}}

{{notes_section}}
### Instructions
Write a Python function `transform` that converts a given input grid to its corresponding output grid based on the pattern observed in the reference examples. 
- Your function should have signature: `def transform(input_grid: np.ndarray) -> np.ndarray:`
- The input and output grids are 2D numpy arrays of integers.
- Each grid cell is assigned a **color**, represented by an **integer**. These integers **do not have numerical meaning**; they **should not** be used in arithmetic operations. Think of these integers as **labels** rather than numbers.
- Write your final code response in a markdown python code block (be sure to have "```python" in the line before your code and "```" in the line after your code).
- No need to provide examples of the function call, just the function definition is sufficient."""

# ------------------------------------------------------------------------------
# formatting functions
# ------------------------------------------------------------------------------


def format_grid_numpy(grid: np.ndarray, separator: str = ", ") -> str:
    grid_string = np.array2string(
        grid,
        separator=", ",
        max_line_width=MAX_LINE_WIDTH,
    )
    return grid_string


def format_puzzle_for_prompt(
    problem: Problem,
    format_grid: Callable[[np.ndarray], str] = format_grid_numpy,
    max_examples: int | None = None,
    include_intro: str | None = EXAMPLE_GRIDS_INTRO,
    include_dim: bool = False,
    include_test: bool = True,
    header: str | None = None,
) -> str:
    def format_grid_with_label(grid: np.ndarray, label: str, include_dim: bool = False):
        if not include_dim:
            return f"{label}:\n{format_grid(grid)}"
        else:
            return f"{label} ({grid.shape[0]}x{grid.shape[1]}):\n{format_grid(grid)}"

    formatted_examples_components = []
    for i, pair in enumerate(problem.train_pairs):
        if max_examples and i >= max_examples:
            break
        formatted_examples_components.append(f"Example {i + 1}")
        # formatted_examples_components.append(f"Input:\n{format_grid(pair.x)}")
        # formatted_examples_components.append(f"Output:\n{format_grid(pair.y)}\n")
        formatted_examples_components.append(
            format_grid_with_label(pair.x, "Input", include_dim)
        )
        formatted_examples_components.append(
            format_grid_with_label(pair.y, "Output", include_dim)
        )
        # add extra newline between examples
        formatted_examples_components.append("")
    formatted_examples = "\n".join(formatted_examples_components)

    if include_test:
        test_grid_intro = "Here is the input grid for the test example:\nInput:"
        formatted_test_grid = "\n".join(
            [format_grid(pair.x) for pair in problem.test_pairs]
        )
    else:
        test_grid_intro = None
        formatted_test_grid = None

    prompt_components = [
        header,
        include_intro,
        formatted_examples,
        test_grid_intro,
        formatted_test_grid,
    ]
    filtered_prompt_components = [pc for pc in prompt_components if pc]
    prompt = "\n".join(filtered_prompt_components)
    return prompt


def format_icl_demo_section(
    problems: list[Problem],
    solutions: dict[str, str] | None = None,
    intro: str = ICL_SECTION_INTRO,
) -> str:
    formatted_examples_components = [intro]
    for i, problem in enumerate(problems, start=1):
        # formatted_examples_components.append(f"## Example {i}")
        formatted_examples_components.append(
            format_puzzle_for_prompt(
                problem,
                format_grid=format_grid_numpy,
                header=f"### Example Puzzle {i} Grids",
            )
        )
        formatted_examples_components.append(f"### Example Puzzle {i} Solution")
        solution = problem.code or "# no solution provided"
        if solutions:
            solution = solutions.get(problem.uid, solution)
        formatted_examples_components.append(f"```python\n{solution}\n```")
    formatted_examples = "\n".join(formatted_examples_components)
    return formatted_examples


def make_prompt(
    problem: Problem,
    examples: list[Problem] | None = None,
    concepts: str | None = None,
    hint: str | None = None,
    description: str | None = None,
    common_lib: str | None = None,
    intro: str = ARC_INTRO,
    hint_template_key: str = "basic",
    require_hint_citations: bool = False,
    instruction_key: str = "standard",
) -> str:
    """
    makes prompt in the following format:

    ### Introduction
    ...
    ### Grids
    There are reference examples...
    Example 1
    Input:
    ...
    Output:
    ...
    This is the input grid for the test example:
    ...
    ### Instructions
    ...
    [Extras: concepts, hints, common lib]
    """
    icl_demo_section = format_icl_demo_section(examples) if examples else None
    grids = format_puzzle_for_prompt(
        problem, format_grid=format_grid_numpy, header="### Your Puzzle Grids"
    )
    instructions = CODE_INSTR_DICT[instruction_key]
    if hint and require_hint_citations:
        instructions = f"{instructions}\n{HINT_CITATION_EXTRA_INSTRUCTION}"
    formatted_concepts = (
        CONCEPT_SECTION_INTRO.format(concepts=concepts) if concepts else None
    )
    if hint:
        formatted_hint = HINT_TEMPLATES[hint_template_key].format(hints=hint)
    else:
        formatted_hint = None
    if description:
        formatted_desc = DESCRIPTION_TEMPLATE.format(description=description)
    else:
        formatted_desc = None
    formatted_common_lib = (
        LIBRARY_TEMPLATE.format(common_lib=common_lib) if common_lib else None
    )
    prompt_components = [
        intro,
        icl_demo_section,
        grids,
        instructions,
        formatted_concepts,
        formatted_desc,
        formatted_hint,
        formatted_common_lib,
    ]
    prompt = "\n".join(filter(None, prompt_components))
    return prompt


def make_retry_prompt(
    initial_prompt: str,
    solution_thread: SolutionThread,
    num_feedback_passes: int = 1,
    error_feedback: Literal["first", "all"] = "first",
    include_past_outcomes: bool = False,
    new_concepts: str | None = None,
) -> str:
    """
    Compose the final retry prompt.

    `previous_responses` should already contain any error / mismatch
    annotations that you want the model to see.
    """

    previous_responses = _format_previous_responses(
        solution_thread=solution_thread,
        num_passes=num_feedback_passes,
        error_feedback=error_feedback,
        include_past_outcomes=include_past_outcomes,
    )

    components = [
        initial_prompt,
        "",
        "### Your Previous Response(s) and Outcomes",
        previous_responses,
        "",
        "### New Instructions",
        (
            "Please reflect on the above issues (code formatting, code execution error, "
            "or grid outputs differing from the expected/correct example outputs), "
            "and revise your reasoning, transformation rule hypothesis, or code accordingly. "
            "Please reflect on the your previous response and consider whether your transformation rule hypothesis is incorrect "
            "or if the code implementation is flawed."
        ),
    ]
    if new_concepts:
        components.append("")
        components.append(
            "### Reselected Lessons\n"
            f"Here are reselected lessons that may or may not be helpful for solving this puzzle:\n{new_concepts}"
        )

    return "\n".join(components)


OUTPUT_MISMATCH_FEEDBACK_HEADER = """\
**Output Mismatches**
The puzzle provides reference examples containing input and output pairs. Here are the outputs from your previous attempt's code that did not match the expected output:"""


def _format_previous_responses(
    solution_thread: SolutionThread,
    num_passes: int,
    error_feedback: Literal["first", "all"],
    include_past_outcomes: bool,
) -> str:
    if num_passes == -1:
        attempts = solution_thread.steps
        first_attempt_num = 1
    else:
        attempts = solution_thread.steps[-num_passes:]
        first_attempt_num = len(solution_thread.steps) - num_passes + 1

    # gather entries for each pass/previous response
    blocks: list[str] = []
    for idx, att in enumerate(attempts, start=first_attempt_num):
        header = f"#### Attempt {idx}"
        body = att.completion
        footer_lines: list[str] = []
        # populate footer_lines with outcomes as necessary
        if include_past_outcomes or (idx == len(solution_thread.steps)):
            errs, mism = _extract_errors_and_mismatches(att)
            if error_feedback == "first":
                errs = errs[:1]
                mism = mism[:1]
            if errs:
                footer_lines.append("**Execution / Parsing Errors**")
                for e in errs:
                    footer_lines.append(f"- {e}")
            if mism:
                footer_lines.append(OUTPUT_MISMATCH_FEEDBACK_HEADER)
                for ex_idx, grid in mism:
                    if grid is None or isinstance(grid, str):
                        grid_txt = str(grid)
                    else:
                        grid_txt = np.array2string(
                            np.array(grid, dtype=int), separator=", "
                        )
                    footer_lines.append(f"- Example {ex_idx}:\n{grid_txt}")
        # add entry
        block = "\n".join([header, body] + footer_lines)
        blocks.append(block)
    formatted = "\n\n---\n\n".join(blocks)
    return formatted


def _extract_errors_and_mismatches(
    step: SolutionStep,
) -> tuple[list[str], list[tuple[int, np.ndarray | str | None]]]:
    """
    Given a solution step/attempt, return:
      errors      – list[str]            (parsing error or exec errors)
      mismatches  – list[(ex_idx, grid)] (index starts at 1)
    """

    errors: list[str] = []
    mismatches: list[tuple[int, np.ndarray]] = []

    if step.parsing_error:
        errors.append(step.parsing_error)
        return errors, mismatches

    # execution / scoring phase on TRAIN examples only
    if len(step.train_results) == 0:
        return errors, mismatches

    for result in step.train_results:
        if result.correct:
            continue
        if result.error:
            errors.append(result.error)
        else:
            # if no error, but not correct, it means grid mismatch
            mismatches.append((result.pair_idx + 1, result.output))

    return errors, mismatches


# example format for selected concept:
# - name: {name}
#   notes:
#    {notes list}
#   examples: # NOTE: for the first k examples, do the full thing
#   - puzzle_id: {puzzle_id}
#     solution_description: {one liner description}
#     pseudocode:
#        {pseudocode step list}
#   - puzzle_id: {puzzle_id}
#     solution_description: {one liner description}
#     use_case: {how the puzzle was used}
def format_concept_for_prompt(
    concept: Concept,
    annotations: dict[str, dict],
    detailed: int = 3,
    max_examples: int = 5,
) -> str:
    # no matter what, we want to include name and notes
    components = [
        concept.to_string(include_notes=True),
        "  examples of other puzzle solutions that use this concept:",
    ]
    # now we want to include info on the examples
    for i, (puzzle_id, usage) in enumerate(concept.use_cases.items()):
        if i >= max_examples:
            break
        annotation = annotations.get(puzzle_id, {})
        solution_description = annotation.get("specific", annotation.get("general", ""))
        components.append(f"  - puzzle_id: {puzzle_id}")
        components.append(f"    solution description: {solution_description}")
        if usage:
            components.append(f"    concept usage: {usage}")
        if i < detailed:
            pseudocode = annotation.get("pseudocode", None)
            if not pseudocode:
                continue
            pseudocode_dict = {"pseudocode": pseudocode}
            yaml_string = yaml.dump(pseudocode_dict)
            yaml_lines = yaml_string.splitlines()
            components.extend(f"    {line}" for line in yaml_lines)
    return "\n".join(components)


def make_lcs_puzzle_solving_prompt(
    puzzle: Problem,
    selected_concepts: list[str],
    notes: str,
    concept_memory: ConceptMemory,
    solution_annotations: dict[str, dict],
    detailed_examples: int = 3,
    max_examples: int = 5,
) -> str:
    puzzle_grids = format_puzzle_for_prompt(puzzle)

    selected_concepts_info = []
    for concept in selected_concepts:
        concept_obj = concept_memory.concepts.get(concept, None)
        if not concept_obj:
            concept_info = f"- {concept}: Concept not found in memory."
        else:
            concept_info = format_concept_for_prompt(
                concept_obj,
                solution_annotations,
                detailed=detailed_examples,
                max_examples=max_examples,
            )
        selected_concepts_info.append(concept_info)
    selected_concepts_str = "\n".join(selected_concepts_info)
    if notes:
        notes_section = NOTES_SECTION_TEMPLATE.format(notes=notes)
    else:
        notes_section = ""
    return SECOND_PASS_TEMPLATE.format(
        puzzle_grids=puzzle_grids,
        selected_concepts=selected_concepts_str,
        notes_section=notes_section,
    )
