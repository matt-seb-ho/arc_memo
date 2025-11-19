#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# 400-puzzle baseline
for r in 0 1 2; do
  python -m concept_mem.evaluation.driver \
    data=arc_agi_val model=o4_mini generation=long_cot_defaults \
    puzzle_retry.max_passes=$((r+1)) generation.n=1 \
    hydra.run.dir="experiments/400pz/baseline_r$r"
done

# 400-puzzle ArcMemo-PS
for r in 0 1 2; do
  python -m concept_mem.evaluation.driver \
    data=arc_agi_val prompt.problem_data="data/curr2/c2_desc_hintm_pi.json" \
    model=o4_mini generation=long_cot_defaults prompt.hint_template_key="op3a" \
    puzzle_retry.max_passes=$((r+1)) generation.n=1 \
    hydra.run.dir="experiments/400pz/arcmemo_ps_r$r"
done

# Cross-model: Claude
python -m concept_mem.evaluation.driver \
  data=val100 model=claude_sonnet_4 generation=long_cot_defaults \
  puzzle_retry.max_passes=2 generation.n=1 \
  hydra.run.dir="experiments/claude"

# Cross-model: Gemini
python -m concept_mem.evaluation.driver \
  data=val100 model=gemini_2_5_flash generation=long_cot_defaults \
  puzzle_retry.max_passes=2 generation.n=1 \
  hydra.run.dir="experiments/gemini"

