"""
AIME-specific prompts for lesson abstraction.

This module contains prompts used by the two AIME pipelines:
1. Label-Guided: Extracts lessons from verified correct solutions
2. Self-Reflective: Extracts lessons from uncertain self-reflections

Active Prompts:
- EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT: Label-Guided pipeline
- EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT_UNCERTAIN: Self-Reflective pipeline
- LESSON_FROM_TRACE_EXAMPLE_TEMPLATE: Few-shot formatting (with solution)
- LESSON_FROM_REFLECTION_EXAMPLE_TEMPLATE: Few-shot formatting (without solution)
"""

# ============================================================================
# ACTIVE PROMPT 1: Label-Guided Pipeline
# Used by: aime_lesson_from_trace_strict.yaml
# Domain Template: aime_strict
# Input: Solution + Thought Process (from verified correct solutions)
# Output: 2-5 lessons with situation-suggestion pairs
# Few-shot: Uses LESSON_FROM_TRACE_EXAMPLE_TEMPLATE (includes solution)
# ============================================================================

EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT = """### Introduction

You are analyzing AIME (American Invitational Mathematics Examination) problems to extract reusable mathematical reasoning patterns. AIME problems span algebra, geometry, number theory, combinatorics, and probability, and are typically solvable with high-school level contest techniques.

Your task is to analyze a solved problem and its reasoning process to extract reusable lessons for solving other AIME problems. These lessons should be phrased as problem-solving "rules" that describe:

- a **situation** (what structural pattern to recognize), and
- a **suggestion** (what concrete technique to try).

### Instructions

We will provide you with a problem answer and the solver's reasoning process.

Your job is to output ONLY a markdown YAML block containing 2–5 lessons, each of the form:

```yaml
- situation: [mathematical pattern or condition to recognize]
  suggestion: [technique or approach to try]
```

No extra prose, no headings, no explanation outside the YAML list.

Design the lessons with the following constraints:

1. **General but Detectable Situations**
   - Describe structural patterns, not specific numbers.
   - Parameterize values: e.g. say "when two overlapping products share a variable and their constants are coprime" instead of "when abc=70 and cde=71."
   - It is acceptable to mention qualitative properties like "prime," "perfect square," "consecutive integers," "equally spaced points," "similar triangles," etc.
   - Prefix each situation with the relevant mathematical domain in brackets: [Algebra], [Geometry], [Number Theory], [Combinatorics], or [Probability]. This categorization aids later retrieval of relevant lessons.

2. **Concrete, Executable Suggestions**
   - Each *suggestion* must describe specific mathematical actions a capable student could carry out: introduce a variable, write a recurrence, apply similarity, set up a system, use the shoelace formula, count lattice points, etc.
   - Avoid vague advice like "analyze carefully," "be systematic," or "consider symmetry" unless you immediately follow it with a concrete procedure.
   - At least **one** lesson must encode the **main key step** from the solution in an explicit way (for example, "define P(k) as … and write a recurrence P(k) = …", or "view each k-digit block of 9s as 10^k − 1 and sum these as a geometric series").

3. **AIME-Level Methods Only**
   - Use tools appropriate for AIME: algebraic manipulation, inequalities, modular arithmetic, recurrences, casework, combinatorial counting, similarity, Pythagorean triples, coordinate geometry, area/ratio arguments, parity, basic number theory (gcd/lcm, prime factorization, divisor-count formulas), etc.
   - Common algebra techniques: Vieta formulas and symmetric sums, polynomial factorization and roots (including simple roots-of-unity structure), functional equations, inequalities such as AM–GM or Cauchy, telescoping/series, and useful substitutions or variable changes.
   - Common number theory techniques: modular arithmetic and congruences, gcd–lcm identities, tracking prime exponents and valuations, using arithmetic functions like τ, σ, or φ in simple ways, orders/Fermat's little theorem, parity arguments, and divisibility or bounding via factorization.
   - Common combinatorics & probability techniques: bijections, stars-and-bars, inclusion–exclusion, recurrences and dynamic programming, Markov-chain style state transitions, expected value and linearity, symmetry/complement counting, and structured arrangements (e.g. circular seating).
   - Common geometry techniques: similar triangles, homothety, power of a point and radical axis, cyclic quadrilaterals and angle-chasing, coordinate or vector/complex-plane setups, and area/perimeter ratios induced by parallel lines.
   - Do NOT invoke heavy or unnecessary methods such as Fourier analysis, character sums, measure theory, advanced group theory, or anything beyond typical high-school olympiad/AIME level.

4. **Avoid Problem-Specific Overfitting**
   - Do not refer to problem labels, specific variable names from the statement, or the problem's numerical constants.
   - Do NOT restate the original problem in disguise.
   - Ask yourself: "Would this lesson still be useful if all the numbers were changed but the structure stayed the same?" If not, generalize it.

5. **No Trivial or Meta Advice**
   - Do not include lessons like "re-read the problem," "check your arithmetic," or "answer the question at the end."
   - Focus only on *mathematical structure* and *methods*.

6. **Faithfulness to the Given Solution**
   - Base your lessons on actual reasoning steps visible in the solver's process.
   - You may compress, reorganize, or generalize those steps, but do not invent fundamentally different solution methods that are not suggested by the reasoning.
   - If the reasoning describes a correction of a previous error, formulate the lesson as the correct strategy (and optionally a brief warning of the pitfall) in general terms.

### Examples
{examples}

### Problem Answer
{solution}

### Solver's Reasoning
{thought_process}
"""

# ============================================================================
# ACTIVE PROMPT 2: Self-Reflective Pipeline
# Used by: aime_lesson_from_trace_uncertain.yaml
# Domain Template: aime_strict_uncertain
# Input: Thought Process only (from self-reflections, may contain uncertainties)
# Output: 2-5 lessons with situation-suggestion pairs + safeguards
# Few-shot: Uses LESSON_FROM_REFLECTION_EXAMPLE_TEMPLATE (excludes solution)
# Key Difference: Adds constraint #7 for handling uncertain reasoning
# ============================================================================

EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT_UNCERTAIN = """### Introduction

You are analyzing a self-reflection on an AIME (American Invitational Mathematics Examination) problem to extract reusable mathematical reasoning patterns. AIME problems span algebra, geometry, number theory, combinatorics, and probability, and are typically solvable with high-school level contest techniques.

The reflection may contain unresolved uncertainties, so treat every step with a critical eye.

Your task is to derive problem-solving "rules" that capture:

- a **situation** (what structural pattern to recognize), and
- a **suggestion** (what concrete technique to try).

### Instructions

We will provide you with the solver's revised reasoning process (the reflection).

Your job is to output ONLY a markdown YAML block containing 2–5 lessons, each of the form:

```yaml
- situation: [mathematical pattern or condition to recognize]
  suggestion: [technique or approach to try]
```

No extra prose, no headings, no explanation outside the YAML list.

Design the lessons with the following constraints:

1. **General but Detectable Situations**
   - Describe structural patterns, not specific numbers.
   - Parameterize values: e.g. say "when two overlapping products share a variable and their constants are coprime" instead of "when abc=70 and cde=71."
   - It is acceptable to mention qualitative properties like "prime," "perfect square," "consecutive integers," "equally spaced points," "similar triangles," etc.
   - Prefix each situation with the relevant mathematical domain in brackets: [Algebra], [Geometry], [Number Theory], [Combinatorics], or [Probability]. This categorization aids later retrieval of relevant lessons.

2. **Concrete, Executable Suggestions**
   - Each *suggestion* must describe specific mathematical actions a capable student could carry out: introduce a variable, write a recurrence, apply similarity, set up a system, use the shoelace formula, count lattice points, etc.
   - Avoid vague advice like "analyze carefully," "be systematic," or "consider symmetry" unless you immediately follow it with a concrete procedure.
   - At least **one** lesson must encode the **main key step** from the reflection in an explicit way (for example, "define P(k) as … and write a recurrence P(k) = …", or "view each k-digit block of 9s as 10^k − 1 and sum these as a geometric series").

3. **AIME-Level Methods Only**
   - Use tools appropriate for AIME: algebraic manipulation, inequalities, modular arithmetic, recurrences, casework, combinatorial counting, similarity, Pythagorean triples, coordinate geometry, area/ratio arguments, parity, basic number theory (gcd/lcm, prime factorization, divisor-count formulas), etc.
   - Common algebra techniques: Vieta formulas and symmetric sums, polynomial factorization and roots (including simple roots-of-unity structure), functional equations, inequalities such as AM–GM or Cauchy, telescoping/series, and useful substitutions or variable changes.
   - Common number theory techniques: modular arithmetic and congruences, gcd–lcm identities, tracking prime exponents and valuations, using arithmetic functions like τ, σ, or φ in simple ways, orders/Fermat's little theorem, parity arguments, and divisibility or bounding via factorization.
   - Common combinatorics & probability techniques: bijections, stars-and-bars, inclusion–exclusion, recurrences and dynamic programming, Markov-chain style state transitions, expected value and linearity, symmetry/complement counting, and structured arrangements (e.g. circular seating).
   - Common geometry techniques: similar triangles, homothety, power of a point and radical axis, cyclic quadrilaterals and angle-chasing, coordinate or vector/complex-plane setups, and area/perimeter ratios induced by parallel lines.
   - Do NOT invoke heavy or unnecessary methods such as Fourier analysis, character sums, measure theory, advanced group theory, or anything beyond typical high-school olympiad/AIME level.

4. **Avoid Problem-Specific Overfitting**
   - Do not refer to problem labels, specific variable names from the statement, or the problem's numerical constants.
   - Do NOT restate the original problem in disguise.
   - Ask yourself: "Would this lesson still be useful if all the numbers were changed but the structure stayed the same?" If not, generalize it.

5. **No Trivial or Meta Advice**
   - Do not include lessons like "re-read the problem," "check your arithmetic," or "answer the question at the end."
   - Focus only on *mathematical structure* and *methods*.

6. **Faithfulness to the Reflection**
   - Base your lessons on actual reasoning steps visible in the reflection.
   - You may compress, reorganize, or generalize those steps, but do not invent fundamentally different solution methods that are not suggested by the reasoning.
   - If the reflection describes a correction of a previous error, formulate the lesson as the correct strategy (and optionally a brief warning of the pitfall) in general terms.

7. **Safeguards for Uncertain Reasoning**
   - Highlight verification steps or self-checks whenever the reflection signals doubt or unproven assumptions.
   - If a step hinges on an unchecked claim, pair it with a diagnostic action that would confirm or refute the claim.

### Examples
{examples}

### Refined Thought Process
{thought_process}
"""

# ============================================================================
# Example Formatting Templates - ACTIVE
# Used by both Label-Guided and Self-Reflective pipelines
# ============================================================================

# ACTIVE: Used by Label-Guided pipeline (includes solution in examples)
# Format: Solution code + Thought process + Lessons
LESSON_FROM_TRACE_EXAMPLE_TEMPLATE = """\
#### Example {example_num}
##### Example {example_num} Puzzle Solution
```python
{solution}
```
##### Example {example_num} Puzzle Solving Thought Process
{thought_process}
##### Example {example_num} Lesson(s)
{lessons}
"""

# ACTIVE: Used by Self-Reflective pipeline (excludes solution from examples)
# Format: Thought process (reflection) + Lessons only
LESSON_FROM_REFLECTION_EXAMPLE_TEMPLATE = """\
#### Example {example_num}
##### Example {example_num} Reflection
{thought_process}
##### Example {example_num} Lesson(s)
{lessons}
"""

