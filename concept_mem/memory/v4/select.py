import asyncio
import logging
from pathlib import Path

import hydra
import yaml
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig
from tqdm import tqdm

from concept_mem.constants import DATA_DIR, DOTENV_PATH, HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.arc_agi import Problem, load_arc_data
from concept_mem.evaluation.prompts import format_puzzle_for_prompt
from concept_mem.utils import (
    extract_yaml_block,
    read_json,
    read_yaml,
    run_llm_job,
    write_json,
)

from .concept import Concept
from .memory import ConceptMemory

logger = logging.getLogger(__name__)


SELECT_PROMPT_TEMPLATE = """\
# Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and solving the puzzle means predicting the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

Your task is to analyze a puzzle's reference examples, examine a set of concepts recorded from previously solved puzzles, and determine which concepts are relevant to this puzzle. Your selected concepts and notes will be used for the puzzle solving phase, so emphasize problem solving helpfulness.


# Concepts from Previously Solved Puzzles
We recorded concepts about structures and routines we observed in previously solved puzzles. These concepts may or may not be relevant to this puzzle, but they provide useful context to show examples of what structures may appear in the grids, what operations may be used, and how they might be composed. Concepts are annotated with fields like:
- cues: (short for "relevance cues"), what to look for that might indicate this concept is relevant in this puzzle
- implementation: notes on how this concept was implemented in past solution programs
- output typing: what the output of this routine is (e.g. a grid, a list, a number, a bool, etc.)
- parameters: a list of parameters that describe ways the concept may vary
We also have some recommendations on how to approach problem solving with these concepts in mind:
- We label the grid manipulation routines separately-- these directly affect the grids so they are easier to spot (along with structure concepts)
- You might try to first identify which grid manipulation operations are used, then investigate their parameters
- The non-grid manipulation routines might describe ways we've seen previous puzzles set parameters, so you can look to these for inspiration
- There may not be exact matches to this list, so we encourage you to think about variations, novel ways to recombine existing ideas, as well as completely new concepts
- These concepts and this approach are only suggestions, use them as you see fit

{concepts}

# Instructions
Identify which concepts could be relevant to the given puzzle.
- We suggest first investigating more "visible" concepts first (e.g. structures and grid manipulation routines)
- After identifying these concepts, you can investigate what logic/criteria/intermediate routines might be useful for these initially identified concepts
- You can also select any other concept you think might be relevant even if it's not directly related to the grid manipulation routines
- Write your final selection of concepts as a yaml formatted list of concept names
- To allow us to match your selection to the concepts we have, please use the exact concept names as they appear in the above concept list
- Write your answer inside a markdown yaml code block (i.e. be sure to have "```yaml" in the line before your code and "```" in the line after your list)
- Here is a formatting example:
```yaml
- line drawing
- intersection of lines
...
```

# Your Given Puzzle
Analyze the following puzzle:
{puzzle}"""


async def select_concepts(
    problems: dict[str, Problem],
    mem_str: str,
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    metadata = []
    prompts = []
    for pzid, puzzle in problems.items():
        metadata.append(pzid)
        puzzle_string = format_puzzle_for_prompt(puzzle, include_dim=True)
        prompt = SELECT_PROMPT_TEMPLATE.format(
            concepts=mem_str,
            puzzle=puzzle_string,
        )
        prompts.append(prompt)

    model_output = await run_llm_job(
        prompts=prompts,
        metadata=metadata,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    selected_concepts = {}
    for pzid, output in zip(metadata, model_output):
        output = output[0]
        if not output:
            logger.warning(f"No output for puzzle {pzid}")
            continue
        selection_block = extract_yaml_block(output)
        try:
            selected_list = yaml.safe_load(selection_block)
            fixed_list = []
            for e in selected_list:
                if isinstance(e, str):
                    fixed_list.append(e.strip())
                elif isinstance(e, dict):
                    if len(e) == 1:
                        k, v = next(iter(e.items()))
                        fixed_list.append(f"{v.strip()}: {k.strip()}")
                    else:
                        logger.warning(
                            f"Unexpected dict format in selection for {pzid}: {e}"
                        )
                else:
                    logger.warning(
                        f"Unexpected type in selection for {pzid}: {type(e)}"
                    )
            selected_concepts[pzid] = [c.strip() for c in fixed_list]
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML for puzzle {pzid}: {e}")
            continue
    write_json(
        selected_concepts,
        output_dir / "selected_concepts.json",
    )
    return selected_concepts


async def async_main(cfg: DictConfig) -> None:
    # output directory setup
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # load instructions and ICL demos
    if cfg.selection.mem_str_path:
        mem_str_path = Path(cfg.selection.mem_str_path)
        if mem_str_path.is_absolute():
            mem_str_path = mem_str_path
        else:
            mem_str_path = REPO_ROOT / mem_str_path
        mem_str = mem_str_path.read_text()
    else:
        raise ValueError("No memory string path provided in config.")

    # prepare target puzzles
    pzids = read_json(cfg.selection.problems)
    target_puzzles = {pzid: Problem.from_puzzle_id(pzid) for pzid in pzids}

    # model related setup
    llm_client = LLMClient(
        provider=Provider(cfg.selection.model.provider),
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    gen_cfg = hydra.utils.instantiate(cfg.selection.generation)

    # initialize concept memory
    concept_mem = ConceptMemory()
    concept_mem.load_from_file(cfg.selection.mem_path)

    # run concept selection
    logger.info("Starting concept selection...")
    selected_concepts = await select_concepts(
        problems=target_puzzles,
        mem_str=mem_str,
        llm_client=llm_client,
        model=cfg.selection.model.name,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.dry_run,
    )

    if cfg.dry_run:
        return

    # create a prompt info artifact from this
    prompt_info = {}
    for pzid, selection in selected_concepts.items():
        sel_mem_str = concept_mem.to_string(
            concept_names=selection,
            skip_parameter_description=False,
            usage_threshold=0,
            show_other_concepts=True,
        )
        prompt_info[pzid] = {"op3f_sel": {"hint": sel_mem_str}}
    prompt_info_path = output_dir / "prompt_info.json"
    write_json(prompt_info, prompt_info_path)
    logger.info(f"prompt_info ready at: {prompt_info_path.relative_to(REPO_ROOT)}")
    logger.info(f"Wrote to output directory: {output_dir}")


@hydra.main(
    version_base=None,
    config_path=HYRDA_CONFIG_PATH,
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """
    Main function to run the annotation process.
    """
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
