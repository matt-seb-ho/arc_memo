# Abstraction Configs

Configuration files for concept/lesson abstraction from solved problems.

---

## Available Configs

### ARC Domain

**`default_lesson_from_trace.yaml`**
- Domain: ARC grid puzzles
- Template: `domain_template: arc` (default)
- Uses: `EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE`
- Examples: ARC puzzle examples

---

### AIME Domain

**`aime_lesson_from_trace.yaml`** (Original)
- Domain: AIME mathematical reasoning
- Template: `domain_template: aime`
- Uses: `EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE`
- Prompt: ~250 tokens, simple instructions
- Style: Concise, gives model more freedom

**`aime_lesson_from_trace_strict.yaml`** (Strict)
- Domain: AIME mathematical reasoning
- Template: `domain_template: aime_strict`
- Uses: `EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT`
- Prompt: ~650 tokens, 6 explicit constraints
- Style: Highly structured, enforces quality standards

---

## Key Difference: `aime` vs `aime_strict`

| Aspect | `aime` (Original) | `aime_strict` (Strict) |
|--------|-------------------|------------------------|
| **Prompt length** | ~250 tokens | ~650 tokens (~2.5x) |
| **Instructions** | Brief, general | 6 numbered constraints |
| **Key step requirement** | Implicit | **Explicit**: Must encode main solution step |
| **Anti-overfitting** | Basic | **Explicit**: Multiple guards |
| **Suggestion concreteness** | Encouraged | **Required**: Must be executable |
| **Scope control** | Mentioned | **Explicit**: AIME-level only |
| **Output strictness** | Standard | **Ultra-strict**: "ONLY a markdown YAML block" |
| **Trivial advice prevention** | Implicit | **Explicit**: List of what NOT to do |

### Strict Version's 6 Constraints

1. **General but Detectable Situations** - Parameterize, use structural patterns
2. **Concrete, Executable Suggestions** - Must describe specific actions, encode main key step
3. **AIME-Level Methods Only** - No advanced university math
4. **Avoid Problem-Specific Overfitting** - Test: "Would this work with different numbers?"
5. **No Trivial or Meta Advice** - No "re-read problem" type advice
6. **Faithfulness to Given Solution** - Base on actual reasoning, don't invent new methods

---

## How to Choose

### Use `aime` (Original) if:
- ✅ Quick experimentation
- ✅ Want model creativity
- ✅ Token efficiency matters
- ✅ Model already produces good outputs

### Use `aime_strict` (Strict) if:
- ✅ Need consistent quality
- ✅ Want concrete, actionable lessons
- ✅ Concerned about overfitting
- ✅ Need explicit encoding of key steps
- ✅ Production/rebuttal quality required

---

## `domain_template` Parameter

**Purpose:** Select domain-appropriate prompt templates without adding new boolean flags

**Values:**
- `"arc"` - Default, for ARC grid puzzles
- `"aime"` - AIME with simple instructions
- `"aime_strict"` - AIME with 6 explicit constraints
- `"gpqa"` - Future: GPQA science reasoning (not yet implemented)

**Usage in config:**
```yaml
abstraction:
  domain_template: aime_strict  # Just set this value
```

**Extensibility:**
Future domains can be added without changing function signatures:
```yaml
abstraction:
  domain_template: gpqa  # Just add new template and update logic
```

---

## Example Usage

### Original AIME
```bash
python -m concept_mem.abstraction.analysis_concepts \
  +abstraction=aime_lesson_from_trace \
  ...
```

### Strict AIME
```bash
python -m concept_mem.abstraction.analysis_concepts \
  +abstraction=aime_lesson_from_trace_strict \
  ...
```

### The ONLY difference
Line 28 in configs:
- Original: `domain_template: aime`
- Strict: `domain_template: aime_strict`

Everything else in the configs is identical!

---

## A/B Testing

**Recommended workflow:**
1. Run both versions on test set (10 problems)
2. Manually review 3-5 sample lessons from each
3. Check: generalization, concreteness, key step encoding
4. Choose better version for full 120-problem training run

See `rebuttal/docs/14_aime_prompt_comparison.md` for detailed comparison guide.

---

**Status:** ✅ Both configs ready for use

