# Domain Categorization in Lesson Abstraction

**Date**: 2024-11-20  
**Status**: Implemented  
**Impact**: Improved lesson retrieval precision

## Overview

Added domain categorization to lesson abstraction to improve retrieval precision. Each lesson now includes its AIME domain (Algebra, Geometry, Number Theory, Combinatorics, or Probability) at the beginning of the situation text.

## Motivation

AIME problems span five distinct mathematical domains, and lessons from one domain may not be relevant to problems in another. By explicitly labeling lessons with their domain, we can:

1. **Improve retrieval precision**: The retrieval LLM can match on domain categories
2. **Enable domain filtering**: Can filter lessons by domain if needed
3. **Better organization**: Lessons are self-documenting with their domain

## Implementation Approach

Rather than explicitly instructing the model to include domain labels, we use **implicit learning through few-shot examples**. The model learns to include domain prefixes naturally from the examples.

### Format

Lessons now follow this format:

```yaml
- situation: "[Domain] When [pattern or condition]..."
  suggestion: "[technique or approach]..."
```

Example:

```yaml
- situation: "[Geometry] When a triangle with convenient side lengths (such as a scaled Pythagorean triple) has points marked at fixed distances along its sides..."
  suggestion: "Place the triangle in a coordinate system so key sides lie on the axes..."
```

## Changes Made

### 1. Updated Few-Shot Examples (`aime_icl_examples.yaml`)

Added domain prefixes to all lessons in the few-shot examples:

- **2019-I-1**: `[Number Theory]` - digit sums and repeated-digit patterns
- **2019-I-2**: `[Probability]` - random selection and counting outcomes
- **2019-I-3**: `[Geometry]` - coordinate geometry and area calculations
- **2019-I-4**: `[Combinatorics]` - sequential operations and counting
- **2019-I-5**: `[Probability]` - random walks and recurrence relations
- **2019-I-10**: `[Algebra]` - polynomial coefficients and symmetric sums

### 2. Expanded Few-Shot Example Set

Updated all abstraction configs to use **5 examples** (previously 3) to ensure all AIME domains are represented:

**Previous**: 2019-I-1, 2019-I-2, 2019-I-3  
**New**: 2019-I-1, 2019-I-2, 2019-I-3, 2019-I-4, 2019-I-10

This ensures coverage of all 5 AIME domains:
- Number Theory (2019-I-1)
- Probability (2019-I-2, 2019-I-5)
- Geometry (2019-I-3)
- Combinatorics (2019-I-4)
- Algebra (2019-I-10)

### 3. Updated Config Files

Modified all abstraction config files to use the expanded example set:

- `configs/abstraction/aime_lesson_from_trace.yaml`
- `configs/abstraction/aime_lesson_from_trace_strict.yaml` (used by Label-Guided pipeline)
- `configs/abstraction/aime_lesson_from_trace_uncertain.yaml` (used by Self-Reflective pipeline)
- `configs/abstraction/aime_lesson_from_trace_mistake.yaml`

### 4. Fixed Pipeline References

Corrected the self-reflective pipeline files to reference the correct config:
- Changed from: `+abstraction=aime_lesson_from_trace_strict_uncertain` (non-existent)
- Changed to: `+abstraction=aime_lesson_from_trace_uncertain` (correct)

Files updated:
- `arc_memo_rebuttal/RUN_AIME_SELF_REFLECTIVE.md`
- `arc_memo_rebuttal/RUN_AIME_SELF_REFLECTIVE_DEBUG.md`

## Domain Definitions

The five AIME domains are:

1. **Algebra**: Equations, inequalities, polynomials, sequences, functional equations, Vieta's formulas
2. **Geometry**: Triangles, circles, coordinate geometry, transformations, area/volume, similarity
3. **Number Theory**: Divisibility, primes, modular arithmetic, Diophantine equations, digit properties
4. **Combinatorics**: Counting, permutations, combinations, graph theory, recursion, sequential operations
5. **Probability**: Expected value, conditional probability, random processes, random walks

## Expected Benefits

1. **Better Retrieval**: The retrieval stage can now match lessons based on domain, reducing irrelevant lesson selection
2. **Domain-Aware Filtering**: Can implement domain-based filtering if needed (e.g., only retrieve geometry lessons for geometry problems)
3. **Improved Analysis**: Can analyze lesson quality and retrieval performance by domain
4. **Self-Documentation**: Lessons are more interpretable and easier to audit

## Testing

To test this change, run the debug pipeline:

```bash
# Self-Reflective approach (uncertain lessons with domain labels)
cd /Users/aaronfeng/Repo/ARC_AGI/arc_memo_rebuttal
bash RUN_AIME_SELF_REFLECTIVE_DEBUG.md

# Label-Guided approach (correct-only lessons with domain labels)
bash RUN_AIME_LABEL_GUIDED_DEBUG.md
```

Check the abstraction output to verify that lessons include domain prefixes:

```bash
# Check abstraction output
cat arc_memo/experiments_self_reflective/debug/abstraction_*/lessons.yaml | head -50
```

## Code Organization

### New File Structure

To improve code organization, AIME prompts have been separated from ARC prompts:

- **`concept_mem/abstraction/aime_concept_prompts.py`** (NEW)
  - Contains only AIME-specific prompts
  - 4 active prompts for the two pipelines
  - Clear documentation of which prompt is used by which pipeline
  
- **`concept_mem/abstraction/analysis_concept_prompts.py`** (ORIGINAL)
  - Now contains only ARC-specific prompts (grid puzzles)
  - Cleaner and more focused

- **`concept_mem/abstraction/analysis_concepts.py`**
  - Updated to import AIME prompts from `aime_concept_prompts.py`
  - Updated to import ARC prompts from `analysis_concept_prompts.py`
  - Simplified logic to only support `aime_strict` and `aime_strict_uncertain`
  - Removed support for legacy/experimental templates

## Related Files

- `/Users/aaronfeng/Repo/ARC_AGI/arc_memo/concept_mem/abstraction/aime_concept_prompts.py` (NEW)
- `/Users/aaronfeng/Repo/ARC_AGI/arc_memo/concept_mem/abstraction/analysis_concepts.py` (UPDATED)
- `/Users/aaronfeng/Repo/ARC_AGI/arc_memo/data/abstract_anno/aime_icl_examples.yaml`
- `/Users/aaronfeng/Repo/ARC_AGI/arc_memo/configs/abstraction/aime_lesson_from_trace*.yaml`

## Notes

- The domain label is part of the situation text, not a separate YAML field, to keep the format simple and backward-compatible
- The model learns the format implicitly from examples, requiring no prompt changes
- Domain labels use title case (e.g., "Number Theory", not "number_theory") for readability

