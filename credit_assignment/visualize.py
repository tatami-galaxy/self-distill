"""Render per-token credit (A_t) heatmaps to a standalone HTML file.

Reads an advantages JSONL (from score_teacher.py) and renders a selected subset
of rollouts as token heatmaps: each completion token is a colored span on a
diverging scale centered at 0 — green for A_t > 0 (teacher endorses the token
more than the student) and red for A_t < 0 (teacher blames it). This mirrors the
signed per-token credit figures in the OPD/SDPO papers.

Per-sequence symmetric scaling (clip at the p95 of |A_t|) keeps a few large
"forking" tokens from washing out the map. No external plotting deps.

Run from the repo root, e.g.:
    python -m credit_assignment.visualize \
        --advantages data/credit_assignment/advantages_opd_..._deepmath.jsonl \
        --n-correct 4 --n-incorrect 4
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render A_t token heatmaps to HTML.")
    p.add_argument("--advantages", required=True, help="JSONL from score_teacher.py")
    p.add_argument("--n-correct", type=int, default=4)
    p.add_argument("--n-incorrect", type=int, default=4)
    p.add_argument("--problem-ids", nargs="*", default=None,
                   help="Explicit problem_ids to render (overrides n-correct/n-incorrect).")
    p.add_argument("--color-by", default="A_t",
                   choices=["A_t", "Abar_t", "reweight_t"],
                   help="Per-token field to color by: A_t (sampled-token advantage), "
                        "Abar_t (expected advantage = -KL_t, dense per-position pull), "
                        "reweight_t (dense-training reweight of the sampled token).")
    p.add_argument("--clip-percentile", type=float, default=95.0,
                   help="Per-sequence percentile of |signal| mapped to full color saturation.")
    p.add_argument("--output", default=None, help="HTML path (default alongside advantages).")
    return p.parse_args()


def load_records(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def select(records: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.problem_ids:
        by_id = {r["problem_id"]: r for r in records}
        return [by_id[pid] for pid in args.problem_ids if pid in by_id]
    correct = [r for r in records if r.get("correct")]
    incorrect = [r for r in records if not r.get("correct")]
    return correct[: args.n_correct] + incorrect[: args.n_incorrect]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 1.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return s[idx]


def _blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


WHITE = (255, 255, 255)
GREEN = (34, 150, 60)
RED = (200, 45, 45)


def color_for(a: float, scale: float) -> str:
    """Diverging color: white at 0, green for +, red for -, saturating at +/-scale."""
    if scale <= 0:
        return "rgb(255,255,255)"
    t = max(-1.0, min(1.0, a / scale))
    r, g, b = _blend(WHITE, GREEN if t >= 0 else RED, abs(t))
    return f"rgb({r},{g},{b})"


def _fmt(x) -> str:
    return f"{x:+.3f}" if isinstance(x, (int, float)) else "n/a"


def render_tokens(record: dict, clip_q: float, field: str) -> str:
    tokens = record["tokens"]
    vals = [t.get(field) for t in tokens]
    scale = _percentile([abs(v) for v in vals if isinstance(v, (int, float))], clip_q)

    spans = []
    for t in tokens:
        v = t.get(field)
        bg = color_for(v, scale) if isinstance(v, (int, float)) else "rgb(245,245,245)"
        surf = t["token_str"]
        # Show newlines as a visible glyph + an actual break so layout is readable.
        display = html.escape(surf).replace("\n", "↵\n")
        tip = (
            f"A_t={_fmt(t.get('A_t'))}  Abar_t={_fmt(t.get('Abar_t'))}  "
            f"reweight_t={_fmt(t.get('reweight_t'))}  teacher_lp={t['teacher_lp']:.3f}  "
            f"student_lp={t['student_lp']:.3f}  id={t['token_id']}"
        )
        spans.append(
            f'<span class="tok" style="background:{bg}" title="{html.escape(tip)}">{display}</span>'
        )
    return "".join(spans)


def render_record(record: dict, clip_q: float, field: str) -> str:
    badge = "✓ correct" if record.get("correct") else "✗ incorrect"
    badge_cls = "ok" if record.get("correct") else "bad"
    header = (
        f'<div class="meta"><span class="badge {badge_cls}">{badge}</span>'
        f'<b>problem_id</b> {html.escape(str(record["problem_id"]))} · '
        f'<b>level</b> {record.get("level", "")} · '
        f'<b>gold</b> {html.escape(str(record.get("gold_answer", "")))} · '
        f'<b>pred</b> {html.escape(str(record.get("pred_answer")))} · '
        f'<b>tokens</b> {record.get("num_completion_tokens", len(record["tokens"]))}</div>'
    )
    problem = f'<div class="problem">{html.escape(record["problem"])}</div>'
    body = f'<div class="trace">{render_tokens(record, clip_q, field)}</div>'
    return f'<section>{header}{problem}{body}</section>'


HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<title>OPD token credit (A_t)</title><style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#111;}
h1{font-size:20px;} .legend{margin:8px 0 24px;font-size:13px;color:#444;}
.legend .sw{display:inline-block;width:140px;height:14px;vertical-align:middle;
 background:linear-gradient(to right,rgb(200,45,45),rgb(255,255,255),rgb(34,150,60));
 border:1px solid #ccc;margin:0 6px;}
section{border:1px solid #e2e2e2;border-radius:8px;padding:14px;margin:0 0 22px;}
.meta{font-size:13px;color:#333;margin-bottom:8px;}
.badge{padding:1px 7px;border-radius:10px;color:#fff;margin-right:8px;font-size:12px;}
.badge.ok{background:#2a9d4a;} .badge.bad{background:#c62d2d;}
.problem{background:#f7f7f8;border-radius:6px;padding:10px;margin-bottom:10px;
 white-space:pre-wrap;font-size:14px;}
.trace{white-space:pre-wrap;line-height:1.9;font-family:ui-monospace,Menlo,Consolas,monospace;
 font-size:13px;}
.tok{border-radius:2px;}
</style></head><body>"""


def main() -> None:
    args = parse_args()
    adv_path = Path(args.advantages)
    output_path = (
        Path(args.output) if args.output
        else adv_path.with_suffix("").with_name(adv_path.stem + "_heatmap.html")
    )

    records = load_records(adv_path)
    chosen = select(records, args)
    if not chosen:
        raise SystemExit("No records selected to render.")

    field = args.color_by
    field_desc = {
        "A_t": "<code>A_t = log π_T(y_t) − log π_S(y_t)</code> (sampled-token advantage): "
               "green = teacher endorses (A_t&gt;0), red = blames (A_t&lt;0)",
        "Abar_t": "<code>Ābar_t = Σ_v π_S(v)[log π_T(v) − log π_S(v)] = −KL_t</code> "
                  "(dense per-position pull; ≤0 so shades of red = stronger pull)",
        "reweight_t": "<code>reweight_t = π_S(y_t)·(A_t − Ābar_t)</code> "
                      "(dense-training reweight of y_t): green = up-weighted, red = down-weighted",
    }[field]
    legend = (
        f'<div class="legend">Coloring by <b>{field}</b>: {field_desc}. '
        '<span class="sw"></span> '
        f'Color saturates at the per-trace p{args.clip_percentile:g} of |{field}|.</div>'
    )
    sections = "\n".join(render_record(r, args.clip_percentile, field) for r in chosen)
    doc = (
        HTML_HEAD
        + f"<h1>OPD token credit ({field}) — {html.escape(adv_path.name)}</h1>"
        + legend
        + sections
        + "</body></html>"
    )
    output_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {len(chosen)} heatmaps -> {output_path}")


if __name__ == "__main__":
    main()
