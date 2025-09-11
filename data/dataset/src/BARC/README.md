### Embedded BARC subset

This folder contains a minimal subset of the BARC codebase, sourced from the upstream repository and included here to support our finalized pipeline (Concept → Description → Code → Problems).

- Upstream source: [BARC on GitHub](https://github.com/xu3kev/BARC.git)
- Scope: only the files needed by our pipeline are included (for example: `generate_code.py`, `generate_problems.py`, `execution.py`, `utils.py`, `prompt.py`, `parse_batch_description_samples.py`, `llm.py`, plus `prompts/` and `seeds/`).
- Note: minor adjustments were made to fit our pipeline integration under `arc_memo/data/dataset/src/BARC`.

If you need the full project, refer to the upstream repository: `https://github.com/xu3kev/BARC.git`.


