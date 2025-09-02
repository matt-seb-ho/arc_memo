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


def _run_single_concept(cfg: dict, row: dict, idx: int, attempt: int, ignore_cache: bool) -> str | None:
    """Run A->B->C for a single concept row. Return problems JSONL path string on success, else None."""
    fb = cfg["finalized_barc"]
    logs_dir = (REPO_ROOT / fb.get("logging", {}).get("outdir", "outputs/logs")).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    if ignore_cache:
        _clear_caches(cfg, attempt_tag=f"row{idx}_try{attempt}")

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
    prompt = tpl.replace("{concept_yaml}", concept_yaml).replace("{fewshot_examples}", fewshot)
    _dump_text(prompt, logs_dir / f"row_{idx:04d}.desc.prompt.try{attempt}.txt")

    # LLM call
    provider = Provider("openai")
    # Use a unique cache dir on retries to minimize cache hits
    llm_cache = (REPO_ROOT / "cache" / (f"attempt_row{idx}_try{attempt}" if ignore_cache else "default")).resolve()
    llm = LLMClient(provider=provider, cache_dir=str(llm_cache))
    gen_cfg = GenerationConfig(temperature=float(fs_cfg["temperature"]), max_tokens=int(fs_cfg["max_output_tokens"]), n=1)
    from external.llmplus.llmplus.sync_adapter import run_sync as _run_sync
    coro = run_llm_job(prompts=[prompt], metadata=[f"row:{idx}"], llm_client=llm, model=fs_cfg["model"], gen_cfg=gen_cfg, output_dir=(REPO_ROOT / fb["stage_a"]["outdir"]).resolve(), dry_run=False)
    outs: list[list[str]] = _run_sync(coro)
    raw = outs[0][0] if outs and outs[0] else ""
    _dump_text(raw, logs_dir / f"row_{idx:04d}.desc.raw.try{attempt}.txt")
    ccsv, desc = _extract_block(raw)
    if not (ccsv and desc):
        return None

    # Write temp JSONL
    tmp_dir = (REPO_ROOT / fb["stage_a"]["outdir"] / "tmp").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    desc_jsonl = tmp_dir / f"single_row_{idx:04d}_try{attempt}.jsonl"
    _dump_jsonl([{ "uid": f"row_{idx:04d}", "concept": concept, "concepts": ccsv, "description": desc }], desc_jsonl)

    # Stage B
    code_dir = stage_b(cfg, desc_jsonl)
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
    fb = cfg["finalized_barc"]
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


def stage_b(cfg: dict, desc_jsonl: Path) -> Path:
    fb = cfg["finalized_barc"]
    code_out_dir = (REPO_ROOT / fb["stage_b"]["outdir"]).resolve()
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
    fb = cfg["finalized_barc"]
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
    return problems_out


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["descriptions", "code", "problems", "all", "retry_fill"], default="all")
    args = parser.parse_args()
    if args.stage == "descriptions":
        desc = stage_a(cfg)
        print(f"[finalized] descriptions -> {desc}")
    elif args.stage == "code":
        fb = cfg["finalized_barc"]
        base = (REPO_ROOT / fb["stage_a"]["outdir"]).resolve() 
        padded = base / "concept_descriptions.padded.jsonl"
        primary = base / "concept_descriptions.jsonl"
        # Prefer padded if it exists (avoids BARC's 10-sample debug sampling crash)
        desc_jsonl = padded if padded.exists() else primary
        out = stage_b(cfg, desc_jsonl)
        print(f"[finalized] code -> {out}")
    elif args.stage == "problems":
        fb = cfg["finalized_barc"]
        code_dir = (REPO_ROOT / fb["stage_b"]["outdir"]).resolve() 
        out = stage_c(cfg, code_dir)
        print(f"[finalized] problems -> {out}")
    elif args.stage == "all":
        # Stage A
        desc_path = stage_a(cfg)
        print(f"[finalized] descriptions -> {desc_path}")
        # Stage B
        fb = cfg["finalized_barc"]
        base = (REPO_ROOT / fb["stage_a"]["outdir"]).resolve()
        padded = base / "concept_descriptions.padded.jsonl"
        primary = base / "concept_descriptions.jsonl"
        desc_jsonl = padded if padded.exists() else primary
        code_dir = stage_b(cfg, desc_jsonl)
        print(f"[finalized] code -> {code_dir}")
        # Stage C
        problems_dir = stage_c(cfg, code_dir)
        print(f"[finalized] problems -> {problems_dir}")
    elif args.stage == "retry_fill":
        fb = cfg["finalized_barc"]
        if not bool(fb.get("retry", {}).get("enabled", False)):
            print("Retry is not enabled in config.")
            return
        target_csv = (REPO_ROOT / fb.get("target_csv", fb["concepts_csv"]))
        rows, headers = _read_csv_all(target_csv)
        helper_col = fb["csv_schema"].get("helper_column", "helper_puzzle")
        if helper_col not in headers:
            headers = headers + [helper_col]
        na_values = fb.get("na_values", ["", "NA", "N/A", "None", "null"])

        # First pass: fill missing once
        for idx, row in enumerate(rows):
            if not _is_na(row.get(helper_col), na_values):
                continue
            success_ref = _run_single_concept(cfg, row, idx, attempt=0, ignore_cache=False)
            if success_ref:
                row[helper_col] = success_ref
                _write_csv_all(target_csv, rows, headers)

        # Retry pass
        max_retries = int(fb.get("retry", {}).get("max_retries", 10))
        for idx, row in enumerate(rows):
            if not _is_na(row.get(helper_col), na_values):
                continue
            for attempt in range(1, max_retries + 1):
                success_ref = _run_single_concept(cfg, row, idx, attempt=attempt, ignore_cache=True)
                if success_ref:
                    row[helper_col] = success_ref
                    _write_csv_all(target_csv, rows, headers)
                    break
        # Final save
        _write_csv_all(target_csv, rows, headers)


if __name__ == "__main__":
    main()


