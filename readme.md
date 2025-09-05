# ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory

We introduce a lightweight LLM memory framework emphasizing higher-level abstraction and modularity to continually improve at compositional reasoning.

## Abstract
While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. On the challenging ARC-AGI benchmark, our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, we confirm that dynamically updating memory during test-time outperforms an otherwise identical fixed memory setting with additional attempts, supporting the hypothesis that solving more problems and abstracting more patterns to memory enables further solutions in a form of self-improvement.

## Repository Structure
- `concept_mem/`: the main package containing the implementation of the ArcMemo framework.
- `notebooks/` (under construction): contains python notebooks for supporting experiments, visualization, and analysis.
- `configs`/: We use [Hydra](https://hydra.cc/) for configuration management. The `configs` directory contains all the configuration files used in our experiments.
- `data/`: contains data files used in the experiments.
- `requirements.txt`: lists the Python dependencies required to run the code.

## Usage
Again we rely on hydra for managing experiments, the following examples highlight the main entry points and scripts to run:

### baseline
```bash
python -m concept_mem.evaluation.driver \
  data=val100 \
  model=o4_mini \
  generation=long_cot_defaults \
  generation.ignore_cache=true \
  puzzle_retry.max_passes=3 \
  generation.n=1
```

### ArcMemo
Here are some example commands to run the ArcMemo pipeline.

```
# preprocess seed solutions into pseudocode
python -m concept_mem.memory.v4.pseudocode \
	+annotate=default \
	annotate.limit_problems=10

# abstract memories
# - pseudocode output by the previous step
# - hand annotations as fewshot examples
python -m concept_mem.memory.v4.abstract \
	+annotate=default \
	annotate.pseudocode=".../initial_analysis.json" \
	annotate.hand_annotations_file="data/abstract_anno/op3/op3a.yaml" \
	annotate.batch_size=1

# select memories
python -m concept_mem.memory.v4.select \
	model@selection.model=o4_mini \
	generation@selection.generation=long_cot_defaults \
	selection.problems="data/testbeds/op3f_unsolved_ids.json" \
	selection.mem_str_path="data/abstract_anno/op3/barc_init_mem.txt" \
	selection.mem_path="data/memory/compressed_v1.json" 

# run inference (puzzle solving)
python -m concept_mem.evaluation.driver \
  data=val100 \
  prompt.problem_data=".../prompt_info.json" \
  model=o4_mini \
  generation=long_cot_defaults \
  generation.ignore_cache=true \
  prompt.hint_template_key="op3a" \
  puzzle_retry.max_passes=3 \
  generation.n=3
```

The key preprocessing step is formatting `problem_data` for the final inference step.
This is where concepts and other information to be included in context is organized.
The shape of the problem data json file is as follows:
```
{
    "[problem_uid]": {
        # note: we allow multiple different prompts to be run in parallel
        "[parallel_run_name]": {
            "hint": "[concepts string]",
            # description is optional
            "description": "[problem description",
        }
    }
    ...
}
```

## Dataset (Under Construction)
We release our concept annotations for difficult target puzzles in `data/dataset/`.
The current annotations support future work in reformatting the current concepts to determine an optimal representation to enable the model to solve the previously unsolved puzzles.

We are currently working on (1) additional annotations for these puzzles and (2) an automated pipeline to convert individual concepts into helper puzzles.
The goal here is to simulate the end-to-end setting of learning concepts from helper puzzles -> reusing the concept to solve target puzzles.
This in-progress dataset would enable future work in measuring model abilities to extract salient concepts from puzzles, with and without distractors.

We are continuing to improve and clean the puzzle generation pipeline, but it is currently functional and documented on this [branch](https://github.com/matt-seb-ho/arc_memo/tree/update_data).
