# prompts for abstracting analysis-oriented concepts

# ============================================================================
# AIME-Specific Prompts (Mathematical Reasoning)
# ============================================================================

EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE = """### Introduction
You are analyzing AIME (American Invitational Mathematics Examination) problems to extract reusable mathematical reasoning patterns. AIME problems span various domains including algebra, geometry, number theory, combinatorics, and probability.

Your task is to analyze a solved problem and its reasoning process to extract reusable lessons for solving other AIME problems. Write problem-solving "rules" that describe a **situation** (what patterns to recognize) and a **suggestion** (what techniques to try).

### Instructions
We will provide you with a problem answer and the solver's reasoning process.
- The "situation" should describe mathematical patterns, structures, or conditions to recognize
- The "suggestion" should recommend specific techniques or approaches
- Make lessons general and reusable across different problems
  - Focus on high-level mathematical insights
  - Parameterize specific values (use "when coefficients form a pattern" not "when a=5")
  - Ask: "Would this help with similar problems that use different numbers?"
- The reasoning may include corrections of an earlier mistake; base your lessons on the final, correct approach and note any pitfall as a general warning.
- Write lessons in a markdown YAML block in the following format:

```yaml
- situation: [mathematical pattern or condition to recognize]
  suggestion: [technique or approach to try]
```

### Examples
{examples}

### Problem Answer
{solution}

### Solver's Reasoning
{thought_process}
"""

EXTRACT_LESSON_FROM_AIME_ZS_TEMPLATE = """### Introduction
You are analyzing an AIME (American Invitational Mathematics Examination) problem to extract reusable mathematical reasoning patterns.

Your task is to analyze a solved problem and its reasoning process to extract reusable lessons for solving other AIME problems.

### Instructions
We will provide you with a problem answer and the solver's reasoning process.
- Extract the key mathematical insights and techniques used
- Write situation-suggestion pairs that are broadly applicable
- Focus on generalizable patterns, not problem-specific details
- If the reasoning mentions an earlier mistake, focus on the corrected method and phrase any caution as a general principle to remember.

Format:
```yaml
- situation: [mathematical pattern or condition]
  suggestion: [technique or approach]
```

### Problem Answer
{solution}

### Solver's Reasoning
{thought_process}
"""

# ============================================================================
# AIME-Specific Prompts - Strict Version (v2)
# More explicit constraints and anti-overfitting guards
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

2. **Concrete, Executable Suggestions**
   - Each *suggestion* must describe specific mathematical actions a capable student could carry out: introduce a variable, write a recurrence, apply similarity, set up a system, use the shoelace formula, count lattice points, etc.
   - Avoid vague advice like "analyze carefully," "be systematic," or "consider symmetry" unless you immediately follow it with a concrete procedure.
   - At least **one** lesson must encode the **main key step** from the solution in an explicit way (for example, "define P(k) as … and write a recurrence P(k) = …", or "view each k-digit block of 9s as 10^k − 1 and sum these as a geometric series").

3. **AIME-Level Methods Only**
   - Use tools appropriate for AIME: algebraic manipulation, inequalities, modular arithmetic, recurrences, casework, combinatorial counting, similarity, Pythagorean triples, coordinate geometry, area/ratio arguments, parity, basic number theory (gcd/lcm, prime factorization, divisor-count formulas), etc.
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

EXTRACT_LESSON_FROM_AIME_FS_TEMPLATE_STRICT_UNCERTAIN = """### Introduction

You are analyzing a self-reflection on an AIME (American Invitational Mathematics Examination) problem to extract reusable mathematical reasoning patterns. The reflection may contain unresolved uncertainties, so treat every step with a critical eye.

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

2. **Concrete, Executable Suggestions**
   - Each *suggestion* must describe specific mathematical actions a capable student could carry out: introduce a variable, write a recurrence, apply similarity, set up a system, use the shoelace formula, count lattice points, etc.
   - Avoid vague advice like "analyze carefully," "be systematic," or "consider symmetry" unless you immediately follow it with a concrete procedure.
   - At least **one** lesson must encode the **main key step** from the reflection in an explicit way (for example, "define P(k) as … and write a recurrence P(k) = …", or "view each k-digit block of 9s as 10^k − 1 and sum these as a geometric series").

3. **AIME-Level Methods Only**
   - Use tools appropriate for AIME: algebraic manipulation, inequalities, modular arithmetic, recurrences, casework, combinatorial counting, similarity, Pythagorean triples, coordinate geometry, area/ratio arguments, parity, basic number theory (gcd/lcm, prime factorization, divisor-count formulas), etc.
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

EXTRACT_LESSON_FROM_AIME_MISTAKE_TEMPLATE = """### Introduction
You are analyzing reflective notes that explain *why a previous AIME solution attempt failed*. Your role is to extract reusable lessons that warn about the pitfall and provide preventative checks.

### Instructions
We will provide you with a solver's reflection about their incorrect attempt (the correct answer is not emphasized).
- Each lesson should include a **situation** describing when the pitfall can appear.
- The **suggestion** should recommend diagnostic checks, sanity tests, or guardrails that help avoid making the same mistake.
- Keep lessons general; avoid copying literal numbers or narrative details from the reflection.
- Do **not** reveal or rely on the final correct answer. Focus on avoiding the error.

### Solver Reflection
{thought_process}
"""

# ============================================================================
# ARC-Specific Prompts (Grid Puzzles)
# ============================================================================

EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

We are trying to learn from puzzles solutions to improve our puzzle solving capabilities. Your task is to analyze a puzzle and its solution to extract reusable lessons for solving other puzzles. Write problem solving "rules" that can be applied to other puzzles. The "rule" format describes a **situation** where it might be useful and includes a **suggestion** for what to try out/consider in that situation.

### Instructions
We will provide you with a puzzle and its solution.
- The "situation" component of the lesson should be about what to look for in the puzzle (shapes, patterns, observations)
- Make the lesson general and reusable for other puzzles.
  - Focus on high level ideas.
  - If there are hardcoded values (colors, number, orientation, shape), try to generealize into a broader statement that parameterizes these hardcoded values.
- Write your lessons in a markdown yaml block (have a "```yaml" line before and "```" line after) in the following format:
```yaml
- situation: [description of the conditions/situations/observations where this rule applies]
  suggestion: [suggestion of what to try out/consider in that situation]
```

### Examples
{examples}

### Your Puzzle
{puzzle}

### Your Puzzle Solution
{solution}
"""

# Contains hints that the examples are highly related
EXTRACT_LESSON_FROM_PUZZLE_FS_TEMPLATE_RETRIEVAL = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

We are trying to learn from puzzle solutions to improve our puzzle-solving capabilities. Your task is to analyze a puzzle **and its solution** to extract reusable lessons for solving _other_ puzzles.

> **Why generalize?**  
> The examples you see below were **intentionally retrieved** because **they share important similarities with the current puzzle** and **they are already verified as valid lessons**.  
> Aim to write lessons that cut across these related puzzles rather than tailoring them to a single instance.
> You may consider refine or improve these already existed lessons.

### Instructions
We will provide you with a puzzle and its solution.
- For each lesson, the **situation** should describe what to look for (shapes, patterns, relationships, observations).
- Make each lesson broadly applicable:
  - Focus on high-level ideas that could help with _multiple_ puzzles, not just the one at hand.
  - If you notice a hard-coded value (color, exact count, orientation, shape), rephrase it into a parameterized or conditional statement.
  - Ask yourself: “Would this hint still be useful if the same idea appeared with different colors or sizes?”
- Structure your output as a markdown YAML block (start with "`yaml`" and end with "` `") in the format:

yaml
- situation: [conditions where the rule applies]
  suggestion: [what to try or consider in that situation]

### Examples
{examples}

### Your Puzzle
{puzzle}

### Your Puzzle Solution
{solution}
"""

EXTRACT_LESSON_FROM_TRACE_ZS_TEMPLATE = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

We are trying to learn from puzzles solutions to improve our puzzle solving capabilities. Your task is to analyze a puzzle solution and the puzzle solving thought process to extract reusable lessons for solving other puzzles. Write problem solving "rules" that can be applied to other puzzles. The "rule" format describes a **situation** where it might be useful and includes a **suggestion** for what to try out/consider in that situation. The given thought process (a sequence of observations and thoughts) demonstrates the reasoning process of solving this particular puzzle. Please try to generalize the lessons from this puzzle to be broadly useful for other puzzles that may have similar or related concepts.

### Instructions
We will provide you with a puzzle solution and a thought process.
- The "situation" component of the lesson should be about what to look for in the puzzle that suggests that a certain concept is in play.
  - Please consider generalizing from the specific observations such that the situation description can handle a class of related puzzles and not just this one.
- Make the lesson general and reusable for other puzzles.
  - Focus on high level ideas.
  - If there are hardcoded values (colors, number, orientation, shape), try to generealize into a broader statement that parameterizes these hardcoded values.
- Write your lessons in a markdown yaml block (have a "```yaml" line before and "```" line after) in the following format:
```yaml
- situation: [description of the conditions/situations/observations where this rule applies]
  suggestion: [suggestion of what to try out/consider in that situation]
```
- Please limit the number of lessons to the most important or broadly useful ones.

### Puzzle Solution
{solution}

### Puzzle Solving Thought Process
{thought_process}
"""

EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

We are trying to learn from puzzles solutions to improve our puzzle solving capabilities. Your task is to analyze a puzzle solution and the puzzle solving thought process to extract reusable lessons for solving other puzzles. Write problem solving "rules" that can be applied to other puzzles. The "rule" format describes a **situation** where it might be useful and includes a **suggestion** for what to try out/consider in that situation. The given thought process (a sequence of observations and thoughts) demonstrates the reasoning process of solving this particular puzzle. Please try to generalize the lessons from this puzzle to be broadly useful for other puzzles that may have similar or related concepts.

### Instructions
We will provide you with a puzzle solution and a thought process.
- The "situation" component of the lesson should be about what to look for in the puzzle that suggests that a certain concept is in play.
  - Please consider generalizing from the specific observations such that the situation description can handle a class of related puzzles and not just this one.
- Make the lesson general and reusable for other puzzles.
  - Focus on high level ideas.
  - If there are hardcoded values (colors, number, orientation, shape), try to generealize into a broader statement that parameterizes these hardcoded values.
- Write your lessons in a markdown yaml block (have a "```yaml" line before and "```" line after) in the following format:
```yaml
- situation: [description of the conditions/situations/observations where this rule applies]
  suggestion: [suggestion of what to try out/consider in that situation]
```
- Please limit the number of lessons to the most important or broadly useful ones.

### Examples
{examples}

### Puzzle Solution
{solution}

### Puzzle Solving Thought Process
{thought_process}
"""


# Contains hints that the examples are highly related
EXTRACT_LESSON_FROM_TRACE_FS_TEMPLATE_RETRIEVAL = """### Introduction
Consider a class of "ARC" puzzles where each puzzle has a hidden transformation rule that maps input grids to output grids. Each puzzle presents several input-output grid pairs as reference examples and the task is to predict the transformation rule. Grids are 2D numpy integer arrays with integers representing colors. 0 represents black and should be treated as the background.

We are trying to learn from puzzle solutions to improve our puzzle-solving capabilities. Your task is to analyze a puzzle **solution** together with its recorded **thought process** and extract reusable lessons for solving _other_ puzzles.

> **Why generalize?**  
> The examples you see below were **intentionally retrieved** because **they share important similarities with the current puzzle** and **they are already verified as valid lessons**.  
> Aim to write lessons that cut across these related puzzles rather than tailoring them to a single instance.
> You may consider refine or improve these already existed lessons.

### Instructions
We will provide you with a puzzle solution **and** a thought-process trace.
- For each lesson, the **situation** should describe what signals (shapes, relationships, repeated structures, etc.) suggest a particular strategy.
  - Generalize the description so it applies to a _family_ of puzzles, not just this one.
- Make every lesson broadly reusable:
  - Focus on high-level ideas.
  - Re-phrase hard-coded values (exact color, count, shape) into parameterized or conditional statements.
- Output a markdown YAML block (start with "`yaml`" and end with "` `"):

```yaml
- situation: [conditions/situations/observations where this rule applies]
  suggestion: [what to try or consider in that situation]
"""


LESSON_FROM_PUZZLE_EXAMPLE_TEMPLATE = """\
#### Example {example_num}
##### Example {example_num} Puzzle
{puzzle}
##### Example {example_num} Solution
```python
{solution}
```
##### Example {example_num} Lesson(s)
{lessons}
"""
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

LESSON_FROM_REFLECTION_EXAMPLE_TEMPLATE = """\
#### Example {example_num}
##### Example {example_num} Reflection
{thought_process}
##### Example {example_num} Lesson(s)
{lessons}
"""
