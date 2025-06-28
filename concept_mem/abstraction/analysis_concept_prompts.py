# prompts for abstracting analysis-oriented concepts

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
