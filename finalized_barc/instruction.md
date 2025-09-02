# Finalized BARC Pipeline

This folder contains a self-contained environment for the finalized BARC two-stage pipeline (Concept → Description → Code → Problems) and visualization.

## Layout
- `config.yaml`: Pipeline configuration (paths relative to this folder)
- `prompts/concept_to_description.md`: Stage A prompt template
- `data/clean_concepts_filled.csv` (and `.yaml`): Concept source
- `scripts/pipeline.py`: Entrypoint to run Stage A/B/C
- `scripts/render.py`: Renderer for generated problems PNGs
- `BARC/`: Local copy of the BARC codebase used by Stage B/C
- `outputs/`: Artifacts written here (`descriptions`, `code`, `problems`, `viz`, `logs`)
- `setup_api_key.sh`: Helper to export API keys

## Prereqs
- Python 3.11
- Install repository requirements from project root:
  ```bash
  pip install -r requirements.txt
  ```
- Set API keys (OpenAI, etc.). Example:
  ```bash
  source finalized_barc/setup_api_key.sh
  ```

## Usage
Run from the project root with module-style commands.

- Stage A: Concept → Description
  ```bash
  python -m finalized_barc.scripts.pipeline --stage descriptions
  ```
  Outputs JSONL under `outputs/descriptions/` (and a padded version if needed).

- Stage B: Description → Code (BARC)
  ```bash
  python -m finalized_barc.scripts.pipeline --stage code
  ```
  Writes code JSONL to `outputs/code/`.

- Stage C: Code → Problems (BARC)
  ```bash
  python -m finalized_barc.scripts.pipeline --stage problems
  ```
  Writes `*_generated_problems.jsonl` to `outputs/problems/`.

- Visualization
  ```bash
  python -m finalized_barc.scripts.render
  ```
  Saves per-problem and stacked PNGs under `outputs/viz/`.

## Configuration
Edit `config.yaml`:
- `finalized_barc.concepts_csv`: path to concepts
- `finalized_barc.stage_a`: model, prompt, outdir
- `finalized_barc.stage_b`: BARC codegen options
- `finalized_barc.stage_c`: problem generation controls (e.g., `num_input_grids`)
- `finalized_barc.viz`: visualization output dir
- `finalized_barc.logging`: logs and metadata

Notes:
- Paths in `config.yaml` are relative to this folder.
- Stage C parameters include reduced grids and optional color-invariance checks.
