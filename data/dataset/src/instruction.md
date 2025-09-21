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
- Activate your virtual environment (if using a local venv):
  ```bash
  source .venv/bin/activate
  ```
- Install repository requirements from project root:
  ```bash
  pip install -r requirements.txt
  ```
- Set API keys (OpenAI, etc.). Example:
  ```bash
  source data/dataset/src/setup_api_key.sh
  ```

## Usage
Run from the project root with module-style commands.

- Stage A: Concept → Description
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage descriptions
  ```
  Outputs JSONL under `outputs/descriptions/` (and a padded version if needed).

- Stage B: Description → Code (BARC)
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage code
  ```
  Writes code JSONL to `outputs/code/`.

- Stage C: Code → Problems (BARC)
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage problems
  ```
  Writes `*_generated_problems.jsonl` to `outputs/problems/`.

- Save: write helper paths to CSV after running A, B, C
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage save
  ```
  Uses a pandas-based saver to read per-concept files under `outputs/problems/by_concept/*.jsonl`, extract examples JSON,
  and write those examples into the CSV helper column
  (configured via `src.csv_schema.helper_column`, default `helper_puzzle`).
  If `sample_num` is an integer, fills up to that many NA rows; if `sample_num: "all"`, fills all NA rows.

- Visualization
  ```bash
  python -m data.dataset.src.scripts.render
  ```
  Saves per-problem and stacked PNGs under `outputs/viz/`.

- Visualize helpers (per-concept by_concept/*.jsonl)
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage viz_helpers
  ```
  Renders each `outputs/problems/by_concept/csv_XXXX.jsonl` into PNGs under `outputs/viz_by_concept/`.
  Controlled by `src.viz_helpers` in `config.yaml` (outdir, start, limit, scale).

- Consolidate problems into by_concept
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage consolidate
  ```
  Scans `outputs/problems/*_generated_problems.jsonl`, maps them back to concepts using Stage A outputs,
  and writes/updates `outputs/problems/by_concept/csv_XXXX.jsonl`. Controlled by `src.consolidate.replace_existing`
  (default false, so existing files are not overwritten).

- Progress (skip finished concepts, run A→B→C once)
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage progress
  ```
  Selects up to `sample_num` unfinished concepts (skipping those with
  `outputs/problems/by_concept/csv_XXXX.jsonl`) and runs A→B→C once. It tops
  up to at least 10 descriptions if needed to satisfy BARC codegen, but it
  does not change Stage A/B/C logic.

- Retry: Per-row or mini-batch retries until success or limit
  ```bash
  python -m data.dataset.src.scripts.pipeline --stage retry
  ```
  Strategy is controlled by `src.retry.strategy` ("per_concept" or "mini_batch").
  Respects `start`, `num_sample`, `limit`, `ignore_cache`, and `k_descriptions`.
  Skips concepts already completed (presence of `outputs/problems/by_concept/csv_XXXX.jsonl`).

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
