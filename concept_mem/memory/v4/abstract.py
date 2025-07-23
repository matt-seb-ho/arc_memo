import asyncio
import logging
from pathlib import Path

import hydra
import yaml
from llmplus import GenerationConfig, LLMClient, Provider
from omegaconf import DictConfig
from tqdm import tqdm

from concept_mem.constants import DATA_DIR, DOTENV_PATH, HYRDA_CONFIG_PATH, REPO_ROOT
from concept_mem.data.arc_agi import load_arc_data
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

ABSTRACTION_INSTR_PATH = DATA_DIR / "abstract_anno/op3/concept_instr.txt"
ABSTRACTION_INSTR = ABSTRACTION_INSTR_PATH.read_text()
SKIP_EXAMPLE_IDS = []


CONCEPT_GEN_EX_TEMPLATE = """\
{header}
Puzzle Solution:
```
{solution}
```
Annotation:
```yaml
{annotation}
```"""


def format_concept_examples(
    problem_solutions: dict[str, str],
    annotations: dict[str, list[dict]],
    header_template: str = "## Example {example_number}",
    delimiter: str = "\n\n",
    skip_problems: list[str] | None = None,
) -> str:
    # format ICL examples for problem solution -> pseudocode task
    example_strings = []
    for i, (problem_id, annotation) in enumerate(annotations.items(), start=1):
        if skip_problems and problem_id in skip_problems:
            continue
        if problem_id not in problem_solutions:
            if "pseudocode" in annotation:
                solution = yaml.dump(annotation["pseudocode"], sort_keys=False).strip()
            else:
                logger.info(f"Missing solution for ICL example problem {problem_id}")
                continue
        else:
            solution = problem_solutions[problem_id]
        formatted_annotation = yaml.dump(annotation["concepts"], sort_keys=False)
        example = CONCEPT_GEN_EX_TEMPLATE.format(
            header=header_template.format(example_number=i),
            solution=solution,
            annotation=formatted_annotation.strip(),
        )
        example_strings.append(example)
    return delimiter.join(example_strings)


def parse_concept_model_output(
    model_output: str,
) -> list[dict]:
    # returns list[concept annotations in dict form]
    # first get yaml block
    yaml_string = extract_yaml_block(model_output)
    if not yaml_string:
        logger.info("No YAML block found in model output")
        return []

    # parse the yaml string
    try:
        yaml_data = yaml.safe_load(yaml_string)
        assert isinstance(yaml_data, list), "expected concept list"
        return yaml_data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error parsing concept YAML: {e}")
        return []


async def generate_concepts_batch(
    problems: list[str],
    solutions: dict[str, str],
    examples: dict[str, dict],
    concept_mem: ConceptMemory,
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, list[dict]]:
    # prepare ICL demo string
    formatted_examples = format_concept_examples(
        problem_solutions=solutions,
        annotations=examples,
    )
    # prepare prompts
    puzzle_ids = []
    prompts = []
    for puzzle_id in problems:
        if puzzle_id not in solutions:
            logger.warning(f"Missing solution for puzzle {puzzle_id}, skipping")
            continue
        prompt = ABSTRACTION_INSTR.format(
            examples=formatted_examples,
            concept_list=concept_mem.to_string(),
            pseudocode=solutions[puzzle_id],
        ).strip()
        puzzle_ids.append(puzzle_id)
        prompts.append(prompt)

    # run LLM job
    model_output = await run_llm_job(
        prompts=prompts,
        metadata=puzzle_ids,
        llm_client=llm_client,
        model=model,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=dry_run,
    )

    # parse results
    results = {}
    for puzzle_id, completions in zip(puzzle_ids, model_output):
        if not completions:
            logger.warning(f"No completions for puzzle {puzzle_id}")
            continue
        completion = completions[0]
        try:
            concept_list = parse_concept_model_output(completion)
            results[puzzle_id] = concept_list
        except Exception as e:
            logger.error(f"Error parsing output for puzzle {puzzle_id}: {e}")
            continue

    # save to output directory if specified
    if output_dir:
        output_file = output_dir / "concept_lists.json"
        write_json(results, output_file, indent=True)

    return results


async def generate_concepts(
    problems: list[str],
    solutions: dict[str, str],
    examples: dict[str, dict],
    concept_mem: ConceptMemory,
    batch_size: int,
    llm_client: LLMClient,
    model: str,
    gen_cfg: GenerationConfig,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    batch_num = 0
    for i in tqdm(range(0, len(problems), batch_size), desc="concept gen (batch)"):
        problem_batch = problems[i : i + batch_size]
        batch_output_dir = output_dir / f"batch_{batch_num}" if output_dir else None
        batch_num += 1
        concept_batch = await generate_concepts_batch(
            problems=problem_batch,
            solutions=solutions,
            examples=examples,
            concept_mem=concept_mem,
            llm_client=llm_client,
            model=model,
            gen_cfg=gen_cfg,
            output_dir=batch_output_dir,
            dry_run=dry_run,
        )
        for puzzle_id, concept_list in concept_batch.items():
            for concept in concept_list:
                concept_mem.write_concept(puzzle_id, concept)
        if batch_output_dir:
            output_file = batch_output_dir / "memory.json"
            concept_mem.save_to_file(output_file)
    if output_dir:
        mem_file = output_dir / "memory.json"
        concept_mem.save_to_file(mem_file)


async def async_main(cfg: DictConfig) -> None:
    # output directory setup
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.info(f"Output directory: {output_dir}")

    # load instructions and ICL demos
    hand_annotation_path = Path(cfg.annotate.hand_annotations_file)
    hand_annotations = read_yaml(hand_annotation_path)

    # prepare target puzzles
    limit = cfg.annotate.limit_problems
    target_puzzles = []
    if cfg.annotate.problem_ids is None:
        barc_seeds = load_arc_data("barc_seeds")
        for pzid in barc_seeds:
            if pzid in hand_annotations or (limit and len(target_puzzles) >= limit):
                continue
            target_puzzles.append(pzid)
    else:
        pzids = read_json(cfg.annotate.problem_ids)
        if limit:
            target_puzzles = pzids[:limit]
        else:
            target_puzzles = pzids

    # prepare pseudocode
    pseudocode = read_json(cfg.annotate.pseudocode)
    reformatted_pseudocode = {}
    for k, entry in pseudocode.items():
        if isinstance(entry, str):
            reformatted_pseudocode[k] = entry
        else:
            reformatted_pseudocode[k] = entry["pseudocode"]
    pseudocode = reformatted_pseudocode

    # model related setup
    llm_client = LLMClient(
        provider=Provider(cfg.annotate.model.provider),
        cache_dir=str(REPO_ROOT / "cache"),
        dotenv_path=DOTENV_PATH,
    )
    gen_cfg = hydra.utils.instantiate(cfg.annotate.generation)

    # initialize concept memory
    concept_mem = ConceptMemory()

    # - initialize preliminary concepts
    if cfg.annotate.preliminary_concepts_file:
        prelims = read_yaml(Path(cfg.annotate.preliminary_concepts_file))
        for concept in prelims["concepts"]:
            concept_mem.write_concept(puzzle_id="prelim", ann=concept)
    concept_mem.initialize_from_annotations(hand_annotations)
    # TODO (figure a way around this): remove a redundant example
    for problem_id in SKIP_EXAMPLE_IDS:
        hand_annotations.pop(problem_id, None)

    # run pseudocode generation
    await generate_concepts(
        problems=target_puzzles,
        solutions=pseudocode,
        examples=hand_annotations,
        concept_mem=concept_mem,
        batch_size=cfg.annotate.batch_size,
        llm_client=llm_client,
        model=cfg.annotate.model.name,
        gen_cfg=gen_cfg,
        output_dir=output_dir,
        dry_run=cfg.dry_run,
    )
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
