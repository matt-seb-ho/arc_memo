#!/bin/bash
# AIME pilot experiment
# Tests domain generalization to mathematical reasoning

set -e
cd "$(dirname "$0")/.."

AIME_DIR="data/aime"
EXP_DIR="experiments/aime_pilot"

# Step 1: Download and prepare dataset
echo "Step 1: Downloading AIME dataset"
cd $AIME_DIR
python download_and_prepare.py
cd ../..

# Step 2a: o4 solves training set (just get answers)
echo "Step 2a: o4 solving training problems"
python -m concept_mem.data.aime_simple_solver \
  data=aime_train \
  model=o4_mini \
  generation.max_tokens=512 \
  hydra.run.dir="$EXP_DIR/o4_solve"

# Step 2b: o4 generates thought processes (explain solutions)
echo "Step 2b: o4 generating thought processes"
python -m concept_mem.data.aime_thought_process \
  +abstraction=aime_thought_process \
  abstraction.solutions_file="$EXP_DIR/o4_solve/solutions.json" \
  data=aime_train \
  model=o4_mini \
  hydra.run.dir="$EXP_DIR/o4_reasoning"

# Step 3: gpt41 abstracts concepts (reuses analysis_concepts.py)
echo "Step 3: gpt41 abstracting concepts"
python -m concept_mem.abstraction.analysis_concepts \
  model=gpt41 \
  +abstraction=aime_lesson_from_trace \
  abstraction.problem_solutions="$EXP_DIR/o4_solve/solutions.json" \
  abstraction.thought_processes="$EXP_DIR/o4_reasoning/thought_processes.json" \
  hydra.run.dir="$EXP_DIR/abstraction"

# Step 4: Baseline - GPT 4.1 Mini without memory
echo "Step 4: Baseline (GPT 4.1 Mini, no memory)"
python -m concept_mem.data.aime_simple_solver \
  data=aime_val \
  model=gpt41_mini \
  generation.max_tokens=512 \
  hydra.run.dir="$EXP_DIR/baseline"

# Step 5: Select concepts for validation problems
echo "Step 5: Selecting concepts for validation"
python -m concept_mem.selection.description.select \
  selection.problems="$AIME_DIR/validation_ids.json" \
  selection.lesson_file="$EXP_DIR/abstraction/lessons.json" \
  model@selection.model=gpt41 \
  hydra.run.dir="$EXP_DIR/selection"

# Step 6: GPT 4.1 Mini with memory
echo "Step 6: With ArcMemo-OE concepts"
python -m concept_mem.data.aime_simple_solver \
  data=aime_val \
  model=gpt41_mini \
  prompt.problem_data="$EXP_DIR/selection/prompt_info.json" \
  generation.max_tokens=512 \
  hydra.run.dir="$EXP_DIR/with_memory"

echo "AIME pilot complete. Results in: $EXP_DIR"

