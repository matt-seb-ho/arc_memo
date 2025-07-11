from pathlib import Path
from typing import Optional

from concept_mem.constants import REPO_ROOT
from concept_mem.evaluation.prompts import format_puzzle_for_prompt
from concept_mem.selection.description.image_helpers import (
    DEFAULT_IMAGE_DIR,
    create_barc_seed_img,
    encode_image,
)
from concept_mem.types import Problem

ARC_AGI_INTRO = """\
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the puzzle solver must predict the transformation rule.
"""

# from hypothesis search paper
STATIC_CONCEPT_LIST = """\
- Lines, rectangular shapes
- Symmetries rotations, translations.
- Shape upscaling or downscaling, elastic distortions.
- Containing / being contained / being inside or outside of a perimeter.
- Drawing lines, connecting points, orthogonal projections.
- Copying, repeating objects.
"""

OBS_SPEC_PROMPT = f"""\
# Background
{ARC_AGI_INTRO}
The transformation rule may involve counting or sorting objects (e.g. sorting by size), comparing numbers (e.g. which shape or symbol appears the most? Which is the largest object? Which objects are the same size?), or repeating a pattern for a fixed number of time.

{{concept_list_section}}

# Instructions
Your task is to analyze a provided puzzle {{input_format}} (containing reference example input-output grid pairs) and write a description of the puzzles' grids to help someone solve the puzzle. Feel free to think aloud before writing your responses. Try not to be too committal in your description. The logic involved in ARC puzzles may sometimes be involved, but the actual changes done to the grids are often simple. When possible, use phrasing consistent with the concepts listed above.

Please separate your description into 2 sections.
## Section 1: Observations (wrap inside <observations> and </observations> tags)
- Describe what you see in the input and output grids without speculating on what the transformation rule might be (e.g. "there are red blocks in the input grids, and blue and green blocks at those same positions in the output grids", "there are scattered pixels inside a yellow rectangular shape in the input grid").
- Feel free to comment on objects you see, the grid sizes, or visible patterns like symmetry, repetition, background colors, etc.
- Feel free to comment on relationships you see between grids (e.g. certain objects are preserved from input to output, or certain objects reoccur across examples, or new objects somehow match old ones, etc.)
- Do not be vague about shape changes (shapes are often copied or trivially extended with new pixels/lines/etc.)
- Write observations for each example.
- Write an observation summary subsection that summarizes across examples (wrap inside <summary> and </summary> tags after the last example).
## Section 2: Speculation (wrap inside <speculation> and </speculation> tags)
- Given your observations, speculate on what high level ideas we should consider for determining the transformation rule (e.g. if you noticed that that input objects appear in the output grids with different positions, the high level idea might be about sliding objects around).
- Also comment on what details about the rule need to be hashed out for this idea (e.g. even if you determine the rule might be about sliding objects idea, for example, you still need to hash out details like which objects/what direction/how far). Wrap these extra details inside <details> and </details> tags.
"""

CONCEPT_SECTION_TEMPLATE = """\
Here is a (non-exhaustive) list of possibly relevant concepts:
{concept_list}"""

# IMG_OBS_SPEC_PROMPT = OBS_SPEC_PROMPT.format(
#     input_format="image",
#     concept_list="{concept_list}",
# )

TXT_OBS_SPEC_PROMPT = f"""\
{OBS_SPEC_PROMPT.format(input_format="grid pairs", concept_list_section=CONCEPT_SECTION_TEMPLATE)}

# Puzzle
{{puzzle_input}}
"""


def build_image_caption_query_messages(
    problem: Problem,
    prompt_template: str = OBS_SPEC_PROMPT,
    image_dir: Path = (REPO_ROOT / DEFAULT_IMAGE_DIR),
    image_file_type: Optional[str] = "png",
    exclude_file_type: bool = False,
    concept_list: str | None = None,
    skip_concept_list: bool = False,
    include_puzzle_text: bool = False,
) -> list[dict]:
    # image processing
    image_path = image_dir / f"{problem.uid}.{image_file_type}"
    if not image_path.exists():
        create_barc_seed_img(problem, image_dir)

    # Getting the Base64 string
    base64_image = encode_image(image_path)
    if image_file_type is None or exclude_file_type:
        # https://github.com/QwenLM/Qwen2.5-VL/blob/main/README.md
        image_url = f"data:image;base64,{base64_image}"
    else:
        image_url = f"data:image/{image_file_type};base64,{base64_image}"

    # prompt formatting
    # - add an optionally dynamic concept list
    input_format = "image and text representations" if include_puzzle_text else "image"
    if skip_concept_list:
        concept_list_section = ""
    elif concept_list is None:
        concept_list_section = CONCEPT_SECTION_TEMPLATE.format(
            concept_list=STATIC_CONCEPT_LIST
        )
    else:
        concept_list_section = CONCEPT_SECTION_TEMPLATE.format(
            concept_list=concept_list
        )

    prompt = prompt_template.format(
        input_format=input_format,
        concept_list_section=concept_list_section,
    )
    # - optionally add the puzzle's text representation
    if include_puzzle_text:
        puzzle_text = format_puzzle_for_prompt(problem, include_dim=True)
        prompt += f"\n\n# Puzzle Grids Text Representation\n{puzzle_text}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]
    return messages


def build_text_caption_query_messages(
    problem: Problem,
    prompt: str = TXT_OBS_SPEC_PROMPT,
    grid_representation: str = "standard",
) -> list[dict]:
    # TODO: this doesn't inject concept list section
    if grid_representation == "standard":
        puzzle_input = format_puzzle_for_prompt(problem, include_dim=True)
    elif grid_representation == "objects":
        raise NotImplementedError(
            "Object representation is not implemented yet. Please use standard representation."
        )
    else:
        raise ValueError(
            f"Unknown grid representation: {grid_representation}. Use 'standard' or 'objects'."
        )
    formatted_prompt = prompt.format(puzzle_input=puzzle_input)
    messages = [
        {
            "role": "user",
            "content": formatted_prompt,
        }
    ]
    return messages
