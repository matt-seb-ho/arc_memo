from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import uuid
from llmplus import LLMClient, GenerationConfig, Provider
from concept_mem.utils.llm_job import run_llm_job

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"
PROMPTS_DIR = REPO_ROOT / "prompts"
CACHE_DIR = REPO_ROOT / "cache"

DESC_TPL = PROMPTS_DIR / "concept_to_description.md"


def _clear_caches(cfg: dict, attempt_tag: str) -> None:
    # Clear BARC cache folder if present
    barc_cache = REPO_ROOT / "BARC" / "cache"
    try:
        if barc_cache.exists():
            import shutil
            shutil.rmtree(barc_cache)
    except Exception:
        pass
    # Optionally, adjust CACHE_DIR per attempt; create a unique subdir
    try:
        attempt_cache = REPO_ROOT / "cache" / f"attempt_{attempt_tag}"
        attempt_cache.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _run_single_concept(cfg: dict, row: dict, csv_index: int, attempt: int, ignore_cache: bool, code_out_override: Path | None = None, k_descriptions: int = 1) -> str | None:
    """Run A->B->C for a single concept row. Return problems JSONL path string on success, else None."""
    fb = cfg["src"]
    logs_dir = (REPO_ROOT / fb.get("logging", {}).get("outdir", "outputs/logs")).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    if ignore_cache:
        _clear_caches(cfg, attempt_tag=f"row{csv_index}_try{attempt}")

    # Stage A: build prompt from single row
    name_col = fb["csv_schema"]["name_column"]
    desc_col = fb["csv_schema"]["description_column"]
    concept = {"concept": (row.get(name_col) or "").strip(), "description": (row.get(desc_col) or "").strip()}
    concept_yaml = yaml.safe_dump(concept, sort_keys=False)

    # fewshot
    fs_cfg = fb["stage_a"]
    fewshot = ""
    fs_dir = fs_cfg.get("fewshot_dir")
    if fs_dir:
        fs_path = (REPO_ROOT / fs_dir)
        if fs_path.exists():
            files = sorted([p for p in fs_path.glob("*.txt") if not p.name.lower().startswith("readme")])
            limit = int(fs_cfg.get("fewshot_limit", 0))
            if limit and limit > 0:
                files = files[:limit]
            parts = []
            for p in files:
                try:
                    parts.append(p.read_text(encoding="utf-8").strip())
                except Exception:
                    pass
            if parts:
                fewshot = "\n\n".join(parts)

    tpl = (REPO_ROOT / fb["stage_a"]["prompt_template"]).read_text(encoding="utf-8")
    base_prompt = tpl.replace("{concept_yaml}", concept_yaml).replace("{fewshot_examples}", fewshot)
    _dump_text(base_prompt, logs_dir / f"row_{csv_index:04d}.desc.prompt.try{attempt}.txt")

    # LLM call
    provider = Provider("openai")
    # Retry-specific overrides
    rcfg = fb.get("retry", {})
    retry_temp = float(rcfg.get("temperature", fs_cfg["temperature"]))
    k = max(1, int(k_descriptions))
    rows_for_jsonl: list[dict] = []
    from external.llmplus.llmplus.sync_adapter import run_sync as _run_sync
    for i in range(k):
        # Unique cache per K when ignoring cache
        llm_cache = (REPO_ROOT / "cache" / (f"attempt_row{csv_index}_try{attempt}_k{i}" if ignore_cache else "default")).resolve()
        llm = LLMClient(provider=provider, cache_dir=str(llm_cache))
        gen_cfg = GenerationConfig(temperature=retry_temp, max_tokens=int(fs_cfg["max_output_tokens"]), n=1)
        nonce = uuid.uuid4().hex
        p = base_prompt + f"\n\n# nonce: row={csv_index} attempt={attempt} k={i} id={nonce}"
        _dump_text(p, logs_dir / f"row_{csv_index:04d}.desc.prompt.try{attempt}.k{i}.txt")
        coro = run_llm_job(prompts=[p], metadata=[f"row:{csv_index}:k:{i}"], llm_client=llm, model=fs_cfg["model"], gen_cfg=gen_cfg, output_dir=(REPO_ROOT / fb["stage_a"]["outdir"]).resolve(), dry_run=False)
        outs: list[list[str]] = _run_sync(coro)
        raw_i = outs[0][0] if outs and outs[0] else ""
        _dump_text(raw_i, logs_dir / f"row_{csv_index:04d}.desc.raw.try{attempt}.k{i}.txt")
        ccsv, desc = _extract_block(raw_i)
        if ccsv and desc:
            rows_for_jsonl.append({
                "uid": f"csv_{csv_index:04d}",
                "concept": concept,
                "concepts": ccsv,
                "description": desc
            })
    if not rows_for_jsonl:
        return None

    # Write temp JSONL
    tmp_dir = (REPO_ROOT / fb["stage_a"]["outdir"] / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    desc_jsonl = tmp_dir / f"single_row_{csv_index:04d}_try{attempt}.jsonl"
    _dump_jsonl(rows_for_jsonl, desc_jsonl)

    # No padding: ensure k_descriptions >= 10 in config to satisfy BARC's sampling

    # Stage B
    code_dir = stage_b(cfg, desc_jsonl, code_out_override=code_out_override)
    # Stage C
    problems_dir = stage_c(cfg, code_dir)
    # Evaluate success: look for generated problems file non-empty
    files = sorted(Path(problems_dir).glob("*_generated_problems.jsonl"))
    if not files:
        return None
    # naive check: at least 1 json line exists
    try:
        with files[-1].open("r", encoding="utf-8") as f:
            for _ in f:
                return str(files[-1])
    except Exception:
        return None
    return None


def _dump_text(text: str, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _dump_jsonl(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_csv_all(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        return [dict(r) for r in reader], headers


def _write_csv_all(path: Path, rows: list[dict], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in headers})


def _is_na(val: Any, na_values: list[str]) -> bool:
    s = "" if val is None else str(val).strip()
    return s in {v.strip() for v in na_values}


def _load_jsonl(p: Path) -> list[dict]:
    items: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def _extract_block(sample: str) -> tuple[str, str]:
    """Extract (concepts_csv, description_text) from a BARC-style block.
    Accepts fenced blocks or raw text; supports inline values after the header colons.
    """
    text = sample or ""
    # Prefer fenced content if present
    start = text.find("```")
    if start != -1:
        # skip possible language token after ```
        nl = text.find("\n", start)
        start = nl + 1 if nl != -1 else start + 3
        end = text.find("```", start)
        content = text[start:end] if end != -1 else text[start:]
        lines = content.strip().splitlines()
    else:
        lines = text.splitlines()

    concepts_parts: list[str] = []
    description_parts: list[str] = []
    mode: str | None = None

    def _maybe_header(line: str) -> tuple[str | None, str]:
        """Return (section, remainder) where section in {concepts, description} if line is a header."""
        s = line.strip()
        if not s.startswith("#"): return None, ""
        body = s[1:].strip()
        lowered = body.lower()
        # Accept both "concepts" and "concept" headers
        if lowered.startswith("concepts") or lowered.startswith("concept"):
            key = "concepts" if lowered.startswith("concepts") else "concept"
            rem = body[len(key):].lstrip()
            if rem.startswith(":"): rem = rem[1:].lstrip()
            return "concepts", rem
        if lowered.startswith("description"):
            rem = body[len("description"):].lstrip()
            if rem.startswith(":"): rem = rem[1:].lstrip()
            return "description", rem
        return None, ""

    for raw in lines:
        sec, rem = _maybe_header(raw)
        if sec == "concepts":
            mode = "concepts"
            if rem:
                concepts_parts.append(rem)
            continue
        if sec == "description":
            mode = "description"
            if rem:
                description_parts.append(rem)
            continue
        # Non-header comment lines continue the current section
        s = raw.strip()
        if s.startswith("#"):
            val = s[1:].strip()
            if mode == "concepts":
                concepts_parts.append(val)
            elif mode == "description":
                description_parts.append(val)

    # Normalize concepts to CSV
    ccsv = ", ".join(c.strip() for c in ",".join(concepts_parts).split(",") if c.strip())
    desc = "\n".join(description_parts).strip()
    return ccsv, desc


def _read_concepts_csv(path: Path, name_col: str, desc_col: str, extras: list[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for r in reader:
            name = (r.get(name_col) or "").strip()
            desc = (r.get(desc_col) or "").strip()
            if not name and not desc:
                continue
            concept = {"concept": name, "description": desc}
            # extras: "*" means include all other columns except name/description
            to_copy = []
            if extras and extras == ["*"]:
                to_copy = [h for h in headers if h not in (name_col, desc_col)]
            else:
                to_copy = extras or []
            for ex in to_copy:
                if ex in r:
                    concept[ex] = r[ex]
            rows.append(concept)
    return rows


def stage_a(cfg: dict) -> Path:
    fb = cfg["src"]
    # support sample_num == "all"
    pool = _read_concepts_csv((REPO_ROOT / fb["concepts_csv"]).resolve(), fb["csv_schema"]["name_column"], fb["csv_schema"]["description_column"], fb["csv_schema"].get("extras", []))
    raw_sn = fb.get("sample_num", 20)
    if isinstance(raw_sn, str) and raw_sn.lower() == "all":
        selected = pool
    else:
        sample_num = int(raw_sn)
        do_random = bool(fb.get("random", False))
        selected = random.sample(pool, k=min(sample_num, len(pool))) if do_random else pool[:sample_num]

    outdir = (REPO_ROOT / fb["stage_a"]["outdir"]).resolve()
    logs = (REPO_ROOT / fb["logging"]["outdir"]).resolve() if fb.get("logging", {}).get("enabled", True) else None
    if logs: logs.mkdir(parents=True, exist_ok=True)

    provider = Provider("openai")
    llm = LLMClient(provider=provider, cache_dir=str(CACHE_DIR))
    gen_cfg = GenerationConfig(temperature=float(fb["stage_a"]["temperature"]), max_tokens=int(fb["stage_a"]["max_output_tokens"]), n=1)
    model = fb["stage_a"]["model"]

    # Load optional few-shot examples from folder
    fewshot = ""
    fs_cfg = fb["stage_a"]
    fs_dir = fs_cfg.get("fewshot_dir")
    if fs_dir:
        fs_path = (REPO_ROOT / fs_dir)
        if fs_path.exists():
            files = sorted([p for p in fs_path.glob("*.txt") if not p.name.lower().startswith("readme")])
            limit = int(fs_cfg.get("fewshot_limit", 0))
            if limit and limit > 0:
                files = files[:limit]
            parts: list[str] = []
            for p in files:
                try:
                    parts.append(p.read_text(encoding="utf-8").strip())
                except Exception:
                    pass
            if parts:
                fewshot = "\n\n".join(parts)
    tpl = (REPO_ROOT / fb["stage_a"]["prompt_template"]).read_text(encoding="utf-8")

    prompts = []
    metas = []
    for i, c in enumerate(selected):
        concept_yaml = yaml.safe_dump(c, sort_keys=False)
        prompt = tpl.replace("{concept_yaml}", concept_yaml).replace("{fewshot_examples}", fewshot)
        prompts.append(prompt)
        metas.append(f"csv:{i}")
        if logs: _dump_text(prompt, logs / f"csv_{i:04d}.desc.prompt.txt")

    from external.llmplus.llmplus.sync_adapter import run_sync as _run_sync
    coro = run_llm_job(prompts=prompts, metadata=metas, llm_client=llm, model=model, gen_cfg=gen_cfg, output_dir=outdir, dry_run=False)
    raws: list[list[str]] = _run_sync(coro)

    rows = []
    for i, (c, outs) in enumerate(zip(selected, raws)):
        raw = outs[0] if outs else ""
        if logs: _dump_text(raw, logs / f"csv_{i:04d}.desc.raw.txt")
        ccsv, desc = _extract_block(raw)
        if ccsv and desc:
            # Override the YAML concept name with the generated one (first token if multiple)
            concept_name = ccsv.split(",")[0].strip() if ccsv else (c.get("concept", "") or "")
            concept_obj = dict(c)
            if concept_name:
                concept_obj["concept"] = concept_name
            rows.append({"uid": f"csv_{i:04d}", "concept": concept_obj, "concepts": ccsv, "description": desc})

    outdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = outdir / "concept_descriptions.jsonl"
    _dump_jsonl(rows, out_jsonl)

    pad_to = int(fb["stage_a"].get("pad_descriptions_to", 10))
    if pad_to and len(rows) < pad_to and len(rows) > 0:
        padded = rows.copy()
        i = 0
        while len(padded) < pad_to:
            padded.append(rows[i % len(rows)])
            i += 1
        padded_path = (outdir / "concept_descriptions.padded.jsonl").resolve()
        _dump_jsonl(padded, padded_path)
        return padded_path
    return out_jsonl.resolve()


def stage_b(cfg: dict, desc_jsonl: Path, code_out_override: Path | None = None) -> Path:
    fb = cfg["src"]
    code_out_dir = (code_out_override or (REPO_ROOT / fb["stage_b"]["outdir"]).resolve())
    code_out_dir.mkdir(parents=True, exist_ok=True)
    barc_codegen = REPO_ROOT / fb["stage_b"]["barc_generate_code_py"]
    logs_dir = (REPO_ROOT / fb.get("logging", {}).get("outdir", "outputs/logs")).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", str(barc_codegen),
        "--jsonl", str(Path(desc_jsonl).resolve()),
        "--outdir", str(code_out_dir),
        "--prompt_model", fb["stage_b"]["prompt_model"],
        "--embedding_model", fb["stage_b"]["embedding_model"],
        "-n", str(fb["stage_b"]["num_samples"]),
        "-s", str(fb["stage_b"]["num_seeds"]),
        "--max_tokens", str(fb["stage_b"]["max_tokens"]),
        "-t", str(fb["stage_b"]["temperature"]),
        "--nohtml",
        "--logdir", str(logs_dir),
    ]
    if fb["stage_b"].get("batch_request", True): cmd.append("--batch_request")
    if fb["stage_b"].get("ignore_cache_samples", True): cmd.append("--ignore_cache_samples")
    sp = int(fb["stage_b"].get("sample_parallel", 1))
    if sp > 1: cmd.extend(["--sample_parallel", str(sp)])
    subprocess.run(cmd, check=True, cwd=str(barc_codegen.parent))
    return code_out_dir


def _check_grid(grid: Any) -> bool:
    try:
        import numpy as np
        return isinstance(grid, np.ndarray) and len(grid.shape) == 2 and np.all((grid >= 0) & (grid <= 9))
    except Exception:
        return False


def stage_c(cfg: dict, code_dir: Path) -> Path:
    fb = cfg["src"]
    problems_out = (REPO_ROOT / fb["stage_c"]["outdir"]).resolve()
    problems_out.mkdir(parents=True, exist_ok=True)
    barc_genprob = REPO_ROOT / "BARC" / "generate_problems.py"
    jsonls = sorted(Path(code_dir).glob("*.jsonl"))
    if not jsonls:
        raise FileNotFoundError(f"No code JSONLs in {code_dir}")
    for jf in jsonls:
        cmd = [
            "python", str(barc_genprob),
            "--jsonl", str(jf.resolve()),
            "--outdir", str(problems_out),
            "--total_timeout", str(int(fb["stage_c"].get("total_timeout", 60))),
            "--num_input_grids", str(int(fb["stage_c"].get("num_input_grids", 30))),
            "--num_deterministic_check", str(int(fb["stage_c"].get("num_deterministic_check", 20))),
            "--num_color_permute_check", str(int(fb["stage_c"].get("num_color_permute_check", 20))),
        ]
        subprocess.run(cmd, check=True, cwd=str(barc_genprob.parent))

    # After generation, write per-concept files: csv_xxxx.jsonl with concept -> description -> code -> problems
    # 1) Build uid/description -> concept metadata map from Stage A outputs
    desc_dir = (REPO_ROOT / fb["stage_a"]["outdir"]).resolve()
    desc_files: list[Path] = []
    p_primary = desc_dir / "concept_descriptions.jsonl"
    p_padded = desc_dir / "concept_descriptions.padded.jsonl"
    if p_primary.exists(): desc_files.append(p_primary)
    if p_padded.exists(): desc_files.append(p_padded)
    uid_to_concept_meta: dict[str, dict] = {}
    desc_to_concept_meta: dict[str, dict] = {}
    for df in desc_files:
        for item in _load_jsonl(df):
            try:
                meta = {
                    "uid": item.get("uid"),
                    "concept": (item.get("concept") or {}),
                    "concepts": item.get("concepts"),
                    "description": item.get("description"),
                }
                if item.get("uid") is not None:
                    uid_to_concept_meta[str(item.get("uid"))] = meta
                if item.get("description"):
                    desc_to_concept_meta[str(item.get("description"))] = meta
            except Exception:
                pass

    # 2) Collect codes by uid/description from Stage B jsonls
    code_map: dict[str, list[str]] = {}
    for cj in jsonls:
        try:
            for rec in _load_jsonl(cj):
                codes = rec.get("code") or []
                uid_val = rec.get("uid")
                seed_list = rec.get("seeds") or []
                desc_text = seed_list[-1] if isinstance(seed_list, list) and seed_list else None
                key = None
                if uid_val is not None:
                    key = f"uid::{uid_val}"
                elif desc_text:
                    key = f"desc::{str(desc_text).strip()}"
                if key:
                    code_map.setdefault(key, [])
                    # codes is list of strings (possibly nested lists); flatten first elements
                    for c in codes:
                        try:
                            code_map[key].append(c[0] if isinstance(c, list) else c)
                        except Exception:
                            pass
        except Exception:
            pass

    # 3) Read problems jsonl results, bucket by uid/description and write per-concept file
    problems_files = sorted(problems_out.glob("*_generated_problems.jsonl"))
    out_per_concept_dir = problems_out / "by_concept"
    out_per_concept_dir.mkdir(parents=True, exist_ok=True)
    items_by_key: dict[str, dict] = {}
    for pf in problems_files:
        try:
            for rec in _load_jsonl(pf):
                examples = rec.get("examples")
                if not examples:
                    continue
                uid_val = rec.get("uid")
                seeds = rec.get("seeds") or []
                desc_text = seeds[-1] if isinstance(seeds, list) and seeds else None
                key = None
                if uid_val is not None:
                    key = f"uid::{uid_val}"
                    meta = uid_to_concept_meta.get(str(uid_val))
                else:
                    key = f"desc::{str(desc_text).strip()}" if desc_text else None
                    meta = desc_to_concept_meta.get(str(desc_text).strip()) if desc_text else None
                if not key or not meta:
                    continue
                bucket = items_by_key.setdefault(key, {"meta": meta, "codes": [], "problems": []})
                # enrich per-problem with minimal metadata for ordering/debug
                try:
                    import time as _t
                    rec.setdefault("_meta", {})
                    rec["_meta"]["timestamp"] = int(_t.time())
                except Exception:
                    pass
                bucket["problems"].append(rec)
        except Exception:
            pass

    # attach codes
    for key, bucket in items_by_key.items():
        if key in code_map:
            bucket["codes"] = code_map[key]

    # write files named by uid if present, else by csv index if detectable
    for key, bucket in items_by_key.items():
        meta = bucket.get("meta", {})
        uid_val = meta.get("uid")
        # prefer uid-based filenames; fallback to concept name
        if uid_val is not None:
            fname = f"{uid_val}.jsonl"
        else:
            cname = (meta.get("concept") or {}).get("concept") or "concept"
            fname = f"{cname}.jsonl"
        target = out_per_concept_dir / fname
        # Compose one json record per file (not multiple lines), but keep .jsonl for simplicity
        record = {
            "concept": meta.get("concept"),
            "concepts_csv": meta.get("concepts"),
            "description": meta.get("description"),
            "codes": bucket.get("codes", []),
            "problems": bucket.get("problems", []),
        }
        try:
            import json as _json
            with target.open("w", encoding="utf-8") as f:
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return problems_out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["descriptions", "code", "problems", "save", "retry", "progress"], default="descriptions")
    args = parser.parse_args()
    if args.stage == "descriptions":
        desc = stage_a(cfg)
        print(f"[finalized] descriptions -> {desc}")
    elif args.stage == "code":
        fb = cfg["src"]
        base = (REPO_ROOT / fb["stage_a"]["outdir"]).resolve() 
        padded = base / "concept_descriptions.padded.jsonl"
        primary = base / "concept_descriptions.jsonl"
        # Prefer padded if it exists (avoids BARC's 10-sample debug sampling crash)
        desc_jsonl = padded if padded.exists() else primary
        out = stage_b(cfg, desc_jsonl)
        print(f"[finalized] code -> {out}")
    elif args.stage == "problems":
        fb = cfg["src"]
        code_dir = (REPO_ROOT / fb["stage_b"]["outdir"]).resolve() 
        out = stage_c(cfg, code_dir)
        print(f"[finalized] problems -> {out}")
    elif args.stage == "save":
        fb = cfg["src"]
        target_csv = (REPO_ROOT / fb.get("target_csv", fb["concepts_csv"]))
        rows, headers = _read_csv_all(target_csv)
        helper_col = fb["csv_schema"].get("helper_column", "helper_puzzle")
        if helper_col not in headers:
            headers = headers + [helper_col]
        na_values = fb.get("na_values", ["", "NA", "N/A", "None", "null"])

        # Read per-concept files for direct mapping
        by_concept_dir = (REPO_ROOT / fb["stage_c"]["outdir"] / "by_concept").resolve()
        name_col = fb["csv_schema"]["name_column"]
        if not by_concept_dir.exists():
            print(f"No by_concept directory found at {by_concept_dir}")
            _write_csv_all(target_csv, rows, headers)
            return
        files = sorted(by_concept_dir.glob("*.jsonl"))
        # Determine how many rows to fill based on sample_num
        raw_sn = fb.get("sample_num", 20)
        max_fill = None if (isinstance(raw_sn, str) and raw_sn.lower() == "all") else int(raw_sn)
        filled = 0
        import json as _json
        for per_file in files:
            try:
                # one json per file
                data = _load_jsonl(per_file)
                if not data:
                    continue
                rec = data[0]
                meta = rec.get("concept") or {}
                concept_name = (meta.get("concept") or "").strip()
                problems = rec.get("problems") or []
                if not concept_name or not problems:
                    continue
                # Use examples from the first problem that exists
                examples = None
                for prob in problems:
                    if prob.get("examples"):
                        examples = prob["examples"]
                        break
                if not examples:
                    continue
                # Fill matching row if NA
                for r in rows:
                    if max_fill is not None and filled >= max_fill:
                        break
                    if (r.get(name_col) or "").strip() == concept_name and _is_na(r.get(helper_col), na_values):
                        r[helper_col] = _json.dumps(examples, ensure_ascii=False)
                        filled += 1
                        break
            except Exception:
                continue
        _write_csv_all(target_csv, rows, headers)
    elif args.stage == "retry":
        fb = cfg["src"]
        rcfg = fb.get("retry", {})
        # Separate retry config
        start_index = int(rcfg.get("start", 0))
        num_sample = rcfg.get("num_sample", "all")
        limit = int(rcfg.get("limit", 10))
        ignore_cache = bool(rcfg.get("ignore_cache", True))
        k_desc = int(rcfg.get("k_descriptions", 10))
        strategy = str(rcfg.get("strategy", "per_concept")).lower()
        batch_size = int(rcfg.get("batch_size", 10))

        target_csv = (REPO_ROOT / fb.get("target_csv", fb["concepts_csv"]))
        rows, headers = _read_csv_all(target_csv)
        name_col = fb["csv_schema"]["name_column"]
        helper_col = fb["csv_schema"].get("helper_column", "helper_puzzle")
        if helper_col not in headers:
            headers = headers + [helper_col]
        na_values = fb.get("na_values", ["", "NA", "N/A", "None", "null"])

        # Detect already completed by presence of by_concept file
        by_concept_dir = (REPO_ROOT / fb["stage_c"]["outdir"] / "by_concept").resolve()
        by_concept_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.stem for p in by_concept_dir.glob("*.jsonl")}

        # Build the work list
        total = len(rows)
        end_index = total if (isinstance(num_sample, str) and num_sample.lower() == "all") else min(total, start_index + int(num_sample))
        indices = list(range(start_index, end_index))

        if strategy == "per_concept":
            for idx in indices:
                row = rows[idx]
                uid = f"csv_{idx:04d}"
                out_path = by_concept_dir / f"{uid}.jsonl"
                if out_path.stem in existing and out_path.exists():
                    continue
                attempt = 0
                while attempt <= limit:
                    attempt += 1
                    code_override = (REPO_ROOT / fb["stage_b"]["outdir"] / f"per_concept_{uid}").resolve()
                    _run_single_concept(cfg, row, idx, attempt=attempt, ignore_cache=ignore_cache, code_out_override=code_override, k_descriptions=k_desc)
                    if out_path.exists():
                        break
        else:
            # mini_batch strategy
            attempts: dict[int, int] = {i: 0 for i in indices}
            def is_done(i: int) -> bool:
                return (by_concept_dir / f"csv_{i:04d}.jsonl").exists()
            unfinished = [i for i in indices if not is_done(i)]
            # iterate until all done or limits reached
            while unfinished:
                # select up to batch_size unfinished
                batch = unfinished[:batch_size]
                # if less than batch_size, top up by repeating from start of batch (will generate extra descs for those)
                while len(batch) < batch_size and batch:
                    batch.append(batch[len(batch) % max(1, len(batch))])
                # Build descriptions JSONL for the batch (one per concept occurrence)
                tmp_dir = (REPO_ROOT / fb["stage_a"]["outdir"] / "tmp").resolve()
                tmp_dir.mkdir(parents=True, exist_ok=True)
                desc_jsonl = tmp_dir / f"minibatch_{start_index:04d}_{batch_size}_{attempts.get(batch[0],0)}.jsonl"
                desc_rows: list[dict] = []
                for pos, idx in enumerate(batch):
                    row = rows[idx]
                    # generate 1 description per selected idx occurrence
                    # reuse _run_single_concept's Stage A core inline
                    name_col_l = fb["csv_schema"]["name_column"]
                    desc_col_l = fb["csv_schema"]["description_column"]
                    concept = {"concept": (row.get(name_col_l) or "").strip(), "description": (row.get(desc_col_l) or "").strip()}
                    concept_yaml = yaml.safe_dump(concept, sort_keys=False)
                    fs_cfg = fb["stage_a"]
                    tpl = (REPO_ROOT / fs_cfg["prompt_template"]).read_text(encoding="utf-8")
                    base_prompt = tpl.replace("{concept_yaml}", concept_yaml).replace("{fewshot_examples}", "")
                    provider = Provider("openai")
                    llm_cache = (REPO_ROOT / "cache" / (f"attempt_batch_row{idx}_n{attempts[idx]}_pos{pos}" if ignore_cache else "default")).resolve()
                    llm = LLMClient(provider=provider, cache_dir=str(llm_cache))
                    gen_cfg = GenerationConfig(temperature=float(fs_cfg["temperature"]), max_tokens=int(fs_cfg["max_output_tokens"]), n=1)
                    from external.llmplus.llmplus.sync_adapter import run_sync as _run_sync
                    nonce = uuid.uuid4().hex
                    p = base_prompt + f"\n\n# nonce: idx={idx} pos={pos} n={attempts[idx]} id={nonce}"
                    coro = run_llm_job(prompts=[p], metadata=[f"batch:{idx}:{pos}"], llm_client=llm, model=fs_cfg["model"], gen_cfg=gen_cfg, output_dir=(REPO_ROOT / fs_cfg["outdir"]).resolve(), dry_run=False)
                    outs: list[list[str]] = _run_sync(coro)
                    raw_i = outs[0][0] if outs and outs[0] else ""
                    ccsv, dtext = _extract_block(raw_i)
                    if ccsv and dtext:
                        desc_rows.append({"uid": f"csv_{idx:04d}", "concept": concept, "concepts": ccsv, "description": dtext})
                if not desc_rows:
                    # increment attempts and continue
                    for i in set(batch):
                        attempts[i] += 1
                    unfinished = [i for i in indices if not is_done(i) and attempts[i] < limit]
                    continue
                # ensure at least 10
                if len(desc_rows) < 10 and batch:
                    # top-up by duplicating first entries (acceptable since they are distinct concepts; rare case)
                    fill = 10 - len(desc_rows)
                    desc_rows.extend(desc_rows[:fill])
                _dump_jsonl(desc_rows, desc_jsonl)
                # Run B and C on this batch
                code_dir = stage_b(cfg, desc_jsonl)
                stage_c(cfg, code_dir)
                # update attempts and unfinished
                for i in set(batch):
                    attempts[i] += 1
                unfinished = [i for i in indices if not is_done(i) and attempts[i] < limit]
    elif args.stage == "progress":
        # One-pass A->B->C over unfinished concepts to meet sample_num, skipping those with by_concept files
        fb = cfg["src"]
        name_col = fb["csv_schema"]["name_column"]
        desc_col = fb["csv_schema"]["description_column"]
        # Determine finished
        by_concept_dir = (REPO_ROOT / fb["stage_c"]["outdir"] / "by_concept").resolve()
        by_concept_dir.mkdir(parents=True, exist_ok=True)
        finished = {p.stem for p in by_concept_dir.glob("*.jsonl")}
        # Pool of all concepts
        rows_all = _read_concepts_csv((REPO_ROOT / fb["concepts_csv"]).resolve(), name_col, desc_col, fb["csv_schema"].get("extras", []))
        # Select unfinished according to sample_num
        raw_sn = fb.get("sample_num", 20)
        # Map concept index to unfinished
        unfinished_indices: list[int] = []
        for i, c in enumerate(rows_all):
            uid = f"csv_{i:04d}"
            if uid not in finished:
                unfinished_indices.append(i)
        if isinstance(raw_sn, str) and raw_sn.lower() == "all":
            selected_indices = unfinished_indices
        else:
            n = int(raw_sn)
            do_random = bool(fb.get("random", False))
            pool = unfinished_indices
            if do_random:
                selected_indices = random.sample(pool, k=min(n, len(pool)))
            else:
                selected_indices = pool[:n]
        if not selected_indices:
            print("[progress] No unfinished concepts to process.")
            return
        # Build prompts (at least 10)
        fs_cfg = fb["stage_a"]
        provider = Provider("openai")
        llm = LLMClient(provider=provider, cache_dir=str(CACHE_DIR))
        gen_cfg = GenerationConfig(temperature=float(fs_cfg["temperature"]), max_tokens=int(fs_cfg["max_output_tokens"]), n=1)
        tpl = (REPO_ROOT / fs_cfg["prompt_template"]).read_text(encoding="utf-8")
        # Fewshot
        fewshot = ""
        fs_dir = fs_cfg.get("fewshot_dir")
        if fs_dir:
            fs_path = (REPO_ROOT / fs_dir)
            if fs_path.exists():
                files = sorted([p for p in fs_path.glob("*.txt") if not p.name.lower().startswith("readme")])
                limit_fs = int(fs_cfg.get("fewshot_limit", 0))
                if limit_fs and limit_fs > 0:
                    files = files[:limit_fs]
                parts: list[str] = []
                for p in files:
                    try:
                        parts.append(p.read_text(encoding="utf-8").strip())
                    except Exception:
                        pass
                if parts:
                    fewshot = "\n\n".join(parts)
        prompts: list[str] = []
        metas: list[str] = []
        uids: list[str] = []
        for idx in selected_indices:
            c = rows_all[idx]
            concept_yaml = yaml.safe_dump(c, sort_keys=False)
            base = tpl.replace("{concept_yaml}", concept_yaml).replace("{fewshot_examples}", fewshot)
            prompts.append(base)
            metas.append(f"progress:{idx}")
            uids.append(f"csv_{idx:04d}")
        # Ensure >=10 by adding duplicates with nonce
        while len(prompts) < 10:
            i = len(prompts) % max(1, len(selected_indices))
            idx = selected_indices[i]
            c = rows_all[idx]
            concept_yaml = yaml.safe_dump(c, sort_keys=False)
            base = tpl.replace("{concept_yaml}", concept_yaml).replace("{fewshot_examples}", fewshot)
            import uuid as _uuid
            prompts.append(base + f"\n\n# nonce: progress fill idx={idx} id={_uuid.uuid4().hex}")
            metas.append(f"progress:{idx}:dup")
            uids.append(f"csv_{idx:04d}")
        from external.llmplus.llmplus.sync_adapter import run_sync as _run_sync
        outdir = (REPO_ROOT / fs_cfg["outdir"]).resolve()
        coro = run_llm_job(prompts=prompts, metadata=metas, llm_client=llm, model=fs_cfg["model"], gen_cfg=gen_cfg, output_dir=outdir, dry_run=False)
        outs: list[list[str]] = _run_sync(coro)
        # Compose descriptions JSONL
        tmp_dir = (REPO_ROOT / fs_cfg["outdir"] / "tmp").resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        desc_jsonl = tmp_dir / f"progress_fill_{uuid.uuid4().hex}.jsonl"
        rows_jsonl: list[dict] = []
        for raw, uid, idx_meta in zip((o[0] if o else "" for o in outs), uids, metas):
            ccsv, dtext = _extract_block(raw)
            if not (ccsv and dtext):
                continue
            # retrieve concept data again for completeness
            try:
                idx_str = idx_meta.split(":")[1]
                idx_int = int(idx_str)
                c = rows_all[idx_int]
            except Exception:
                c = {}
            rows_jsonl.append({"uid": uid, "concept": c, "concepts": ccsv, "description": dtext})
        if not rows_jsonl:
            print("[progress] No valid descriptions generated.")
            return
        _dump_jsonl(rows_jsonl, desc_jsonl)
        # Stage B and C
        code_dir = stage_b(cfg, desc_jsonl)
        stage_c(cfg, code_dir)


if __name__ == "__main__":
    main()


