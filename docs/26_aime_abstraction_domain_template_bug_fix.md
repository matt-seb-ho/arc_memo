# AIME Abstraction Domain Template Bug Fix

**Date**: 2025-11-19  
**Issue**: Lesson abstraction stage was using ARC puzzle prompt instead of AIME strict-uncertain prompt

## Problem

When running the AIME 2.5 pipeline with self-reflection and lesson abstraction, the abstraction stage was incorrectly using the ARC puzzle prompt template instead of the configured AIME strict-uncertain template. This resulted in prompts starting with:

```
### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule...
```

Instead of the expected AIME prompt:

```
### Introduction

You are analyzing a self-reflection on an AIME (American Invitational Mathematics Examination) problem...
```

## Root Cause

The bug was in `concept_mem/abstraction/analysis_concepts.py` at line 466:

```python
domain_template=cfg.abstraction.get('domain_template', 'arc'),
```

The config file structure for AIME abstraction (`configs/abstraction/aime_lesson_from_trace_uncertain.yaml`) has a nested `abstraction:` section:

```yaml
# Top level (loaded as cfg.abstraction)
problem_solutions: ...
thought_processes: ...
examples: ...

abstraction:  # Nested section (cfg.abstraction.abstraction)
  batch_size: 5
  domain_template: aime_strict_uncertain
  ...
```

When loaded with `+abstraction=aime_lesson_from_trace_uncertain`, this creates:
- `cfg.abstraction.domain_template` → doesn't exist
- `cfg.abstraction.abstraction.domain_template` → `aime_strict_uncertain`

The code was reading the wrong level, so it defaulted to `'arc'`.

## Solution

Modified `analysis_concepts.py` to handle the nested config structure:

```python
# Handle nested abstraction config structure
abstraction_cfg = cfg.abstraction.get('abstraction', cfg.abstraction)
await extract_lessons(
    ...
    use_barc_solution=abstraction_cfg.get('use_barc_solution', False),
    domain_template=abstraction_cfg.get('domain_template', 'arc'),
    ...
)
```

This checks if there's a nested `abstraction` key and uses that, otherwise falls back to the top-level config. This maintains backward compatibility with the original ARC configs.

## Additional Fixes

### 1. Cache Invalidation

Updated `RUN_AIME_2P5.md` to correctly pass `ignore_cache` to nested configs:

- **Step 3 (Abstraction)**: `abstraction.generation.ignore_cache=true`
- **Step 4 (Selection)**: `selection.generation.ignore_cache=true`

### 2. Debug Logging

Added logging to track:
- Number of thought processes loaded
- First problem's `thought_proc` status and `domain_template` value
- Missing problem IDs in thought processes dict

### 3. Debug Configuration

Created `configs/data/aime_train_debug.yaml` for single-problem testing:

```yaml
dataset: aime
split: train
num_problems: 1
problem_ids:
  - 2020-I-1
```

And `RUN_AIME_2P5_DEBUG.md` for quick pipeline verification.

## Verification

To verify the fix works:

```bash
cd arc_memo
ABSTRACT_RUN=$(ls -td experiments_2p5/aime_train/abstraction_uncertain_* | head -n1)

# Check domain_template in log
grep "First problem" "$ABSTRACT_RUN/analysis_concepts.log"
# Should show: domain_template: aime_strict_uncertain

# Check prompt starts correctly
cat "$ABSTRACT_RUN/prompts.json" | python3 -c "import json, sys; print(json.load(sys.stdin)[0][:200])"
# Should start with: "### Introduction\n\nYou are analyzing a self-reflection on an AIME..."
```

## Files Modified

1. `concept_mem/abstraction/analysis_concepts.py` - Domain template fix + LaTeX escaping fix
2. `concept_mem/data/aime_joint_solver.py` - Problem filtering support
3. `concept_mem/data/aime_self_reflection.py` - Problem filtering support
4. `concept_mem/data/aime_simple_solver_v3.py` - Problem filtering support
5. `arc_memo_rebuttal/RUN_AIME_2P5.md` - Cache invalidation fixes
6. `arc_memo_rebuttal/RUN_AIME_2P5_DEBUG.md` - Complete debug pipeline (6 steps)

## Files Created

1. `configs/data/aime_train_debug.yaml` - Single training problem (2020-I-1)
2. `configs/data/aime_val_debug.yaml` - Single validation problem (2024_I_1)
3. `data/aime/validation_ids_debug.json` - Debug validation IDs
4. `data/aime/validation_debug.json` - Debug validation data
5. `docs/26_aime_abstraction_domain_template_bug_fix.md` - This documentation

## Additional Fix: LaTeX Math in YAML

### Problem
Model outputs containing LaTeX math notation (e.g., `$180^\circ$`) in YAML strings caused parsing errors:
```
Error: while scanning a double-quoted scalar
```

This occurred because YAML treats `$` as a special character for variable references when inside double quotes.

### Solution
Added pre-processing in `parse_lessons()` to:
1. Escape `$` → `\$` before YAML parsing
2. Restore `$` after parsing using `_restore_latex_math()`

This allows the model to naturally use LaTeX math in lesson suggestions without YAML parsing failures.

## Additional Issue: Cache Not Being Invalidated

### Problem
Even with `ignore_cache=true` set at multiple levels, the LLM cache was not being cleared. Runs repeatedly returned identical cached responses.

### Root Cause
The llmplus library's `ignore_cache` flag doesn't actually clear the cache directory (`arc_memo/cache/`). It may only prevent writing new cache entries, not reading existing ones.

### Solution
Manually clear the cache directory before critical runs:
```bash
rm -rf cache/*
```

This ensures fresh API calls to the model without cached responses.

## Impact

This fix ensures that:
- AIME problems use AIME-specific prompts with appropriate mathematical context
- Self-reflection outputs are properly incorporated into lesson abstraction
- The strict-uncertain template is used when configured
- LaTeX math notation in lessons is preserved without YAML parsing errors
- Cache can be manually cleared when needed to force fresh model responses

