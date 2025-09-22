from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_jsonl(p: Path) -> list[dict]:
    items: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def _is_na_val(v: Any, na_values: list[str]) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if s == "":
        return True
    return s in {nv.strip() for nv in na_values}


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    fb = cfg.get("src", {})

    target_csv = (REPO_ROOT / fb.get("target_csv", fb.get("concepts_csv", "data/target.csv"))).resolve()
    name_col = fb.get("csv_schema", {}).get("name_column", "ConceptName")
    helper_col = fb.get("csv_schema", {}).get("helper_column", "helper_puzzle")
    na_values = fb.get("na_values", ["", "NA", "N/A", "None", "null"])

    by_concept_dir = (REPO_ROOT / fb.get("stage_c", {}).get("outdir", "outputs/problems")).resolve() / "by_concept"
    if not by_concept_dir.exists():
        raise FileNotFoundError(f"by_concept directory not found: {by_concept_dir}")

    # Load target CSV with pandas
    if not target_csv.exists():
        raise FileNotFoundError(f"target CSV not found: {target_csv}")
    df = pd.read_csv(target_csv)
    if helper_col not in df.columns:
        df[helper_col] = ""

    # Build concept_name -> examples mapping from by_concept files
    mapping: dict[str, Any] = {}
    files = sorted(by_concept_dir.glob("*.jsonl"))
    for p in files:
        try:
            rows = _load_jsonl(p)
            if not rows:
                continue
            rec = rows[0]
            concept_meta = rec.get("concept") or {}
            concept_name = str((concept_meta.get("concept") or "")).strip()
            if not concept_name:
                continue
            problems = rec.get("problems") or []
            examples = None
            for prob in problems:
                if prob.get("examples"):
                    examples = prob["examples"]
                    break
            if examples is None:
                continue
            mapping[concept_name] = examples
        except Exception:
            continue

    # Determine how many to fill
    raw_sn = fb.get("sample_num", 20)
    max_fill = None if (isinstance(raw_sn, str) and str(raw_sn).lower() == "all") else int(raw_sn)

    # Fill NA helper cells by concept name
    filled = 0
    for idx, row in df.iterrows():
        if max_fill is not None and filled >= max_fill:
            break
        concept_name = str((row.get(name_col) or "")).strip()
        if not concept_name:
            continue
        if concept_name not in mapping:
            continue
        if _is_na_val(row.get(helper_col), na_values):
            try:
                df.at[idx, helper_col] = json.dumps(mapping[concept_name], ensure_ascii=False)
                filled += 1
            except Exception:
                continue

    df.to_csv(target_csv, index=False)
    print(f"[save] Filled {filled} rows in {target_csv}")


if __name__ == "__main__":
    main()


