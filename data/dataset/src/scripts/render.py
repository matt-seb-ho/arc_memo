from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _color_map(v: int) -> tuple[int, int, int]:
    palette = [
        (0, 0, 0),
        (0, 0, 255),
        (0, 128, 0),
        (255, 0, 0),
        (255, 255, 0),
        (255, 165, 0),
        (128, 0, 128),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 128),
    ]
    try:
        iv = int(v)
    except Exception:
        iv = 0
    if not (0 <= iv < len(palette)):
        iv = 0
    return palette[iv]


def _draw_grid(grid: list[list[int]], scale: int = 24, label: str | None = None) -> Image.Image:
    h = max(1, len(grid or []))
    w = max(1, max((len(r or []) for r in (grid or [])), default=1))
    header = 24 if label else 0
    img = Image.new("RGB", (w * scale, h * scale + header), (30, 30, 30))
    dr = ImageDraw.Draw(img)
    y0 = header
    for r in range(h):
        row = grid[r] if r < len(grid) else []
        for c in range(w):
            val = row[c] if c < len(row) else 0
            col = _color_map(val)
            x1, y1 = c * scale, y0 + r * scale
            dr.rectangle([x1, y1, x1 + scale - 1, y1 + scale - 1], fill=col)
    gl = (55, 55, 55)
    for r in range(h + 1):
        y = y0 + r * scale
        dr.line([(0, y), (w * scale, y)], fill=gl)
    for c in range(w + 1):
        x = c * scale
        dr.line([(x, y0), (x, y0 + h * scale)], fill=gl)
    if label:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        dr.text((4, 2), label, fill=(255, 255, 255), font=font)
    return img


def _compose_io_panel(examples: list[Any], title: str | None = None) -> Image.Image:
    cols: list[Image.Image] = []
    for i, ex in enumerate(examples):
        gi, go = None, None
        if isinstance(ex, dict):
            gi = ex.get("input")
            go = ex.get("output")
        elif isinstance(ex, (list, tuple)) and len(ex) == 2:
            gi, go = ex[0], ex[1]
        else:
            continue
        im_in = _draw_grid(gi, label=f"in {i}")
        im_out = _draw_grid(go, label=f"out {i}")
        pad = 8
        col = Image.new("RGB", (max(im_in.width, im_out.width), im_in.height + pad + im_out.height), (20, 20, 20))
        col.paste(im_in, (0, 0))
        col.paste(im_out, (0, im_in.height + pad))
        cols.append(col)
    if not cols:
        cols = [Image.new("RGB", (200, 200), (20, 20, 20))]
    gap = 8
    header = 48 if title else 0
    w = sum(im.width for im in cols) + gap * (len(cols) - 1)
    h = header + max(im.height for im in cols)
    canvas = Image.new("RGB", (w, h), (15, 15, 15))
    dr = ImageDraw.Draw(canvas)
    if title:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        dr.text((6, 8), title, fill=(255, 255, 255), font=font)
    x = 0
    for im in cols:
        canvas.paste(im, (x, header))
        x += im.width + gap
    return canvas


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    fb = (cfg or {}).get("src", {})
    problems_dir = REPO_ROOT / fb.get("stage_c", {}).get("outdir", "outputs/problems")
    viz_dir = REPO_ROOT / fb.get("viz", {}).get("outdir", "outputs/viz")
    viz_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(problems_dir).glob("*_generated_problems.jsonl"))
    if not files:
        raise FileNotFoundError(f"No problems JSONL found in {problems_dir}")
    for jf in files:
        rows = _load_jsonl(jf)
        panels: list[Image.Image] = []
        for idx, r in enumerate(rows):
            examples = r.get("examples") or []
            title = r.get("seeds") or r.get("concepts") or f"sample {idx}"
            if isinstance(title, list):
                title = ",".join(title)
            panel = _compose_io_panel(examples, title=str(title)[:96])
            panels.append(panel)
            single_name = viz_dir / (jf.stem + f"__problem_{idx:02d}.png")
            panel.save(single_name)
        if not panels:
            continue
        W = max(p.width for p in panels)
        H = sum(p.height for p in panels) + 8 * (len(panels) - 1)
        canvas = Image.new("RGB", (W, H), (10, 10, 10))
        y = 0
        for p in panels:
            x = (W - p.width) // 2
            canvas.paste(p, (x, y))
            y += p.height + 8
        out_png = viz_dir / (jf.stem + ".png")
        canvas.save(out_png)


if __name__ == "__main__":
    main()



