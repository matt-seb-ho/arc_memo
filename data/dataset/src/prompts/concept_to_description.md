You are an ARC-AGI puzzle author. ARC-AGI (ARC) problems are small 2D color-grid tasks (values 0..9) where a deterministic transformation maps input to output.

Task: Given ONE internal concept (YAML), write a BARC-style description block. Output a single Markdown Python block that contains exactly:

```python
# concepts: <comma-separated concept tags suitable for BARC>
# description: <concise, deterministic I/O transformation in ARC terms>
```

Guidelines:
- Keep it puzzle-facing: state the typical input contents and the deterministic output transformation.
- Use ARC vocabulary (connected components, 4/8-connectivity, colors 0..9, holes/enclosed regions, symmetry, copy/move/scale, blitting, bounding boxes, etc.).
- Deterministic only; no randomness or ambiguity.
- Formatting: headers must appear exactly as shown; content may be inline after the colon or on subsequent commented lines.

Difference vs our internal concept description: our YAML’s description is an internal note; your BARC-style description must be a self-contained puzzle spec that can drive codegen (generate_input + main).

Target concept (full YAML; may include parameters/extras):
```yaml
{concept_yaml}
```

Here are a few BARC-style examples to emulate (do not copy; follow the style):
{fewshot_examples}

