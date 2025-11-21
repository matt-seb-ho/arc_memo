# Self-Assessment Prompt for Lesson Usage

**Date**: 2025-11-20  
**Status**: Implemented  
**Related Files**: `arc_memo/concept_mem/data/aime_simple_solver_v3.py`

## Problem

Initial experiments with correct-only lessons showed that retrieval was **hurting performance** on easy problems:
- **Baseline (no memory)**: pass@1 = 73.33% (44/60)
- **With memory**: pass@1 = 68.33% (41/60)

Specifically, 6 problems went from **correct → wrong** when lessons were added:
- `2024_I_5`: 104✓ → 80✗
- `2024_I_10`: 113✓ → 2✗
- `2024_I_13`: 110✓ → 134✗
- `2024_II_9`: 902✓ → 900✗
- `2025_I_12`: 510✓ → 1✗
- `2025_II_3`: 82✓ → 58✗

## Root Cause

The model was **unconditionally applying lessons**, even when:
1. The problem was straightforward and didn't need extra guidance
2. The lessons were marginally relevant but added confusion
3. The model already had a correct intuition that got overridden

## Solution

Added a **self-assessment instruction** that gives the model agency to decide whether to use the lessons:

```
If you consider this problem straightforward to solve, you may disregard the following lessons retrieved from other similar problems.
```

This prompt is inserted **before** the lesson list in the validation solve stage.

## Implementation Details

**Modified Function**: `build_problem_prompt()` in `aime_simple_solver_v3.py` (lines 131-153)

**Before**:
```python
components.extend(
    [
        "",
        "Lessons distilled from similar problems:",
        hint.strip(),
    ]
)
```

**After**:
```python
components.extend(
    [
        "",
        "If you consider this problem straightforward to solve, you may disregard the following lessons retrieved from other similar problems.",
        "",
        "Lessons distilled from similar problems:",
        hint.strip(),
    ]
)
```

## Expected Benefits

1. **Preserves baseline performance on easy problems**: Model can ignore lessons when confident
2. **Still helps on hard problems**: Model will use lessons when uncertain
3. **Better calibration**: Model self-assesses difficulty rather than blindly following all hints
4. **Minimal overhead**: Single sentence doesn't significantly increase prompt length

## Testing

To test this change, run:
```bash
cd arc_memo_rebuttal
# Follow instructions in RUN_AIME_QUICK.md
```

Compare:
- **Baseline** (no lessons)
- **With lessons + self-assessment + top_k=2** (this change)

Key metrics to watch:
- **pass@1** on problems 1-8 (easier): Should not degrade
- **pass@1** on problems 9-15 (harder): Should improve with lessons
- **Overall pass@1**: Should be ≥ baseline (73.33%)

### Additional Refinement: Top 2 Lessons Only

After implementing self-assessment, we also reduced from 5 to 2 lessons per problem:
- Set via `prompt.hint_lessons_limit=2` in the solver command
- Keeps only the most relevant lessons, reducing noise
- Combined with self-assessment, gives model maximum flexibility

## Alternative Approaches Considered

1. **Reduce top_k from 5 to 2**: Less aggressive, but doesn't solve the underlying issue
2. **Problem-number-based filtering**: Skip lessons for problems 1-8, use for 9-15
   - Rejected: Too rigid, some "easy" problems are actually hard and vice versa
3. **Confidence threshold**: Only provide lessons if model's first-pass confidence is low
   - Rejected: Requires 2-pass approach, more expensive
4. **Better retrieval ranking**: Improve relevance scores
   - Complementary: Can do this in addition to self-assessment

## Future Improvements

1. Monitor which problems the model **actually ignores** lessons on (requires response analysis)
2. Consider A/B testing with different phrasings:
   - "If you are confident you can solve this problem, you may skip the lessons below"
   - "Use the following lessons only if you find them helpful"
   - "The following lessons are optional hints; use your judgment"
3. Add lesson quality metadata (e.g., "high confidence" vs "low confidence" lessons)

## Conclusion

This simple one-sentence change gives the model **metacognitive control** over lesson usage, potentially preventing the memory system from hurting on easy problems while still helping on hard ones.

