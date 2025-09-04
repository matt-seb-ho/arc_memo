# Finalized BARC Pipeline

This folder contains a self-contained environment for the finalized BARC two-stage pipeline (Concept → Description → Code → Problems) and visualization.

## Layout
- `config.yaml`: Pipeline configuration (paths relative to this folder)
- `prompts/concept_to_description.md`: Stage A prompt template
- `data/clean_concepts_filled.csv` (and `.yaml`): Concept source
- `scripts/pipeline.py`: Entrypoint to run Stage A/B/C
- `scripts/render.py`: Renderer for generated problems PNGs
- `BARC/`: Local copy of the BARC codebase used by Stage B/C
- `outputs/`: Created on first run (`descriptions`, `code`, `problems`, `viz`, `logs`)
- `setup_api_key.sh`: Helper to export API keys

## Prereqs
- Python 3.11
- Activate your virtual environment (if using a local venv):
  ```bash
  source .venv/bin/activate
  ```
- Install repository requirements from project root:
  ```bash
  pip install -r requirements.txt
  ```
- Set API keys (OpenAI, etc.). Example (run from repo root):
  ```bash
  source data/dataset/src/setup_api_key.sh
  ```

## Usage
Run from the repo root: `arc_memo` (cd into it first), with module-style commands.

- Stage A: Concept → Description
  ```bash
  # from arc_memo/
  python -m data.dataset.src.scripts.pipeline --stage descriptions
  ```
  Outputs JSONL under `data/dataset/src/outputs/descriptions/` (and a padded version if needed).

- Stage B: Description → Code (BARC)
  ```bash
  # from arc_memo/
  python -m data.dataset.src.scripts.pipeline --stage code
  ```
  Writes code JSONL to `data/dataset/src/outputs/code/`.

- Stage C: Code → Problems (BARC)
  ```bash
  # from arc_memo/
  python -m data.dataset.src.scripts.pipeline --stage problems
  ```
  Writes `*_generated_problems.jsonl` to `data/dataset/src/outputs/problems/`.

- Save: write helper paths to CSV after running A, B, C
  ```bash
  # from arc_memo/
  python -m data.dataset.src.scripts.pipeline --stage save
  ```
  Scans `data/dataset/src/outputs/problems/` for the latest `*_generated_problems.jsonl` and writes its path into the CSV helper column
  (configured via `src.csv_schema.helper_column`, default `helper_puzzle`).
  If `sample_num` is an integer, fills up to that many NA rows; if `sample_num: "all"`, fills all NA rows.

- Visualization
  ```bash
  # from arc_memo/
  python -m data.dataset.src.scripts.render
  ```
  Saves per-problem and stacked PNGs under `data/dataset/src/outputs/viz/`.

- Progress (skip finished concepts, run A→B→C once)
  ```bash
  # from arc_memo/
  python -m data.dataset.src.scripts.pipeline --stage progress
  ```
  Selects up to `sample_num` unfinished concepts (skipping those with
  `data/dataset/src/outputs/problems/by_concept/csv_XXXX.jsonl`) and runs A→B→C once. It tops
  up to at least 10 descriptions if needed to satisfy BARC codegen, but it
  does not change Stage A/B/C logic.

- Retry: Per-row retries until success or limit
  ```bash
  # from arc_memo/
  python -m data.dataset.src.scripts.pipeline --stage retry
  ```
  Iterates over CSV rows with NA in the helper column and repeatedly runs A→B→C for one concept at a time
  (ignoring cache if configured) until a valid problems JSONL is produced or `src.retry.limit`
  is reached. Existing non-NA entries are never modified.

## Configuration
Edit `config.yaml`:
- `src.concepts_csv`: path to concepts
- `src.stage_a`: model, prompt, outdir
- `src.stage_b`: BARC codegen options
- `src.stage_c`: problem generation controls (e.g., `num_input_grids`)
- `src.viz`: visualization output dir
- `src.logging`: logs and metadata

Notes:
- Paths in `config.yaml` are relative to this folder.
- Stage C parameters include reduced grids and optional color-invariance checks.
