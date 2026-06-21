"""Static SVG figures for the public v2 report."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .schemas import DIMENSIONS


PALETTE = {
    "blue": "#2f4fd8",
    "orange": "#cc6f47",
    "green": "#177245",
    "red": "#b3261e",
    "purple": "#7657c8",
    "muted": "#6f7a86",
    "line": "#d9dee7",
}


def _pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.1f}%"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    text = f"{float(value):.{digits}f}"
    return text.rstrip("0").rstrip(".")


def _figure_shell(width: int, height: int, body: str, title: str, subtitle: str | None = None) -> str:
    subtitle_svg = (
        f'<text class="muted" x="24" y="52" font-size="13">{escape(subtitle)}</text>'
        if subtitle
        else ""
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">
<style>
.bg{{fill:#fff}} .ink{{fill:#11151c}} .muted{{fill:#5b6470}} .grid{{stroke:#d9dee7}} .axis{{stroke:#8b95a1}}
text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}}
@media (prefers-color-scheme:dark){{.bg{{fill:#161b22}} .ink{{fill:#e9edf3}} .muted{{fill:#9aa4b1}} .grid{{stroke:#303844}} .axis{{stroke:#717b88}}}}
</style>
<rect class="bg" x="0" y="0" width="{width}" height="{height}" rx="10"/>
<text class="ink" x="24" y="30" font-size="18" font-weight="700">{escape(title)}</text>
{subtitle_svg}
{body}
</svg>
"""


def _write_svg(out_dir: Path, filename: str, svg: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(svg, encoding="utf-8")
    return f"assets/figures/v2/{filename}"


def _short_model(name: str) -> str:
    return name.replace("xai:", "").replace("llama3.2:", "llama ").replace("deepseek-r1:", "deepseek ")


def _bar_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    label_key: str,
    series: list[tuple[str, str, str]],
    value_max: float | None = None,
    pct_values: bool = True,
    width: int = 920,
) -> str:
    top = 76
    row_h = 34
    left = 190
    right = 130
    height = top + 34 * len(rows) + 32
    values = [float(row.get(key) or 0) for row in rows for _label, key, _color in series]
    max_value = value_max if value_max is not None else max(values + [0.01])
    max_value = max(max_value, 0.01)
    bar_h = 10 if len(series) > 1 else 14
    chart_w = width - left - right
    body = [
        f'<line class="axis" x1="{left}" y1="{top - 10}" x2="{left}" y2="{height - 34}" stroke-width="1"/>',
        f'<line class="grid" x1="{left + chart_w}" y1="{top - 10}" x2="{left + chart_w}" y2="{height - 34}" stroke-width="1"/>',
    ]
    for i, row in enumerate(rows):
        y = top + i * row_h
        label = _short_model(str(row.get(label_key, "")))
        body.append(f'<text class="ink mono" x="24" y="{y + 13}" font-size="12">{escape(label[:28])}</text>')
        for j, (_name, key, color) in enumerate(series):
            value = float(row.get(key) or 0)
            w = round((value / max_value) * chart_w, 1)
            by = y + 1 + j * (bar_h + 3)
            body.append(f'<rect x="{left}" y="{by}" width="{w}" height="{bar_h}" rx="3" fill="{color}"/>')
        value_text = " / ".join(_pct(row.get(key)) if pct_values else _num(row.get(key)) for _name, key, _color in series)
        body.append(f'<text class="muted mono" x="{left + chart_w + 12}" y="{y + 13}" font-size="12">{escape(value_text)}</text>')
    legend_x = left
    for name, _key, color in series:
        body.append(f'<rect x="{legend_x}" y="{height - 20}" width="10" height="10" rx="2" fill="{color}"/>')
        body.append(f'<text class="muted" x="{legend_x + 15}" y="{height - 11}" font-size="12">{escape(name)}</text>')
        legend_x += 120
    return _figure_shell(width, height, "\n".join(body), title, subtitle)


def _quality_heatmap(rows: list[dict[str, Any]]) -> str:
    width = 920
    cell_w = 82
    cell_h = 34
    left = 165
    top = 88
    height = top + cell_h * len(rows) + 36
    body: list[str] = []
    for j, dim in enumerate(DIMENSIONS):
        x = left + j * cell_w
        body.append(f'<text class="muted mono" x="{x + cell_w / 2}" y="72" text-anchor="middle" font-size="12">{dim}</text>')
    for i, row in enumerate(rows):
        y = top + i * cell_h
        body.append(f'<text class="ink mono" x="24" y="{y + 22}" font-size="12">{escape(str(row.get("role", "")))}</text>')
        for j, dim in enumerate(DIMENSIONS):
            value = row.get(dim)
            opacity = max(0.08, min(0.86, float(value or 0) * 0.9))
            x = left + j * cell_w
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="5" fill="rgba(47,79,216,{opacity:.3f})"/>')
            body.append(f'<text class="ink mono" x="{x + cell_w / 2}" y="{y + 20}" text-anchor="middle" font-size="12">{escape(_num(value, 2))}</text>')
    return _figure_shell(
        width,
        height,
        "\n".join(body),
        "Non-refusal quality by role",
        "Refused outputs are excluded; darker cells are higher primary-judge quality scores.",
    )


def _metric_cards(metrics: list[dict[str, Any]], title: str, subtitle: str) -> str:
    width = 920
    height = 240
    card_w = 200
    gap = 18
    start_x = 24
    y = 86
    body: list[str] = []
    for i, metric in enumerate(metrics):
        x = start_x + i * (card_w + gap)
        body.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="116" rx="8" fill="rgba(47,79,216,0.08)" stroke="{PALETTE["line"]}"/>')
        body.append(f'<text class="muted" x="{x + 14}" y="{y + 28}" font-size="12">{escape(metric["label"])}</text>')
        body.append(f'<text class="ink mono" x="{x + 14}" y="{y + 68}" font-size="28" font-weight="700">{escape(metric["value"])}</text>')
        if metric.get("note"):
            body.append(f'<text class="muted" x="{x + 14}" y="{y + 94}" font-size="11">{escape(metric["note"])}</text>')
    return _figure_shell(width, height, "\n".join(body), title, subtitle)


def _role_design_chart(rows: list[dict[str, Any]]) -> str:
    chart_rows = [{"dim": row.get("dim"), "corr": row.get("expected_agency_correlation")} for row in rows]
    width = 920
    top = 78
    row_h = 34
    left = 220
    chart_w = 450
    height = top + row_h * len(chart_rows) + 34
    body = [f'<line class="axis" x1="{left + chart_w / 2}" y1="{top - 10}" x2="{left + chart_w / 2}" y2="{height - 30}" stroke-width="1"/>']
    for i, row in enumerate(chart_rows):
        y = top + i * row_h
        corr = row["corr"]
        value = float(corr or 0)
        zero = left + chart_w / 2
        end = zero + value * (chart_w / 2)
        x = min(zero, end)
        w = abs(end - zero)
        color = PALETTE["green"] if value >= 0 else PALETTE["red"]
        body.append(f'<text class="ink mono" x="24" y="{y + 13}" font-size="12">{escape(str(row["dim"]))}</text>')
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="14" rx="3" fill="{color}"/>')
        body.append(f'<text class="muted mono" x="{left + chart_w + 16}" y="{y + 13}" font-size="12">{escape(_num(corr))}</text>')
    return _figure_shell(
        width,
        height,
        "\n".join(body),
        "Role-card design correlations",
        "Expected role profiles are not a single monotone agency ladder across all dimensions.",
    )


def generate_v2_figures(summary: dict[str, Any], docs_dir: Path) -> list[dict[str, str]]:
    """Generate public v2 SVG figures and return site-relative metadata."""
    v2 = summary.get("v2", {})
    if not v2.get("available"):
        return []

    out_dir = docs_dir / "assets" / "figures" / "v2"
    figures: list[dict[str, str]] = []

    refusal_rows = sorted(v2.get("refusal_by_model", []), key=lambda row: row.get("refusal_rate") or 0, reverse=True)
    if refusal_rows:
        rel = _write_svg(
            out_dir,
            "local_refusal_by_model.svg",
            _bar_chart(
                refusal_rows,
                title="Local refusal by model",
                subtitle="DeepSeek is the refusal outlier; most refusals are over-refusals under primary-judge scoring.",
                label_key="model",
                series=[
                    ("refusal", "refusal_rate", PALETTE["blue"]),
                    ("over-refusal", "over_refusal_rate", PALETTE["orange"]),
                ],
                value_max=max([row.get("refusal_rate") or 0 for row in refusal_rows] + [0.01]),
            ),
        )
        figures.append({"title": "Local refusal by model", "path": rel, "caption": "Refusal is reported separately from answer quality."})

    quality_rows = v2.get("quality_non_refusal_by_role", [])
    if quality_rows:
        rel = _write_svg(out_dir, "local_quality_by_role.svg", _quality_heatmap(quality_rows))
        figures.append({"title": "Non-refusal quality by role", "path": rel, "caption": "Quality claims exclude refusals by construction."})

    profile_rows = sorted(v2.get("role_profile_by_role", []), key=lambda row: row.get("profile_fit_mean") or 0, reverse=True)
    if profile_rows:
        rel = _write_svg(
            out_dir,
            "local_profile_fit_by_role.svg",
            _bar_chart(
                profile_rows,
                title="Role-profile fit by role",
                subtitle="Higher is closer to the role the output was supposed to enact.",
                label_key="role",
                series=[("profile fit", "profile_fit_mean", PALETTE["green"])],
                value_max=1.0,
                pct_values=False,
            ),
        )
        figures.append({"title": "Role-profile fit", "path": rel, "caption": "Profile fit checks whether outputs match the role they were asked to perform."})

    robust = v2.get("judge_robustness", {})
    if robust.get("available"):
        post = robust.get("poststratified", {})
        metrics = [
            {
                "label": "sample rows",
                "value": str(robust.get("n_common")),
                "note": "stratified Qwen sensitivity",
            },
            {
                "label": "raw refusal mismatch",
                "value": _pct(robust.get("refusal_mismatch_rate")),
                "note": "balanced sample",
            },
            {
                "label": "post-strat mismatch",
                "value": _pct(post.get("refusal_mismatch_rate") if post.get("available") else None),
                "note": "weighted to full refusal rate",
            },
            {
                "label": "role-profile delta",
                "value": _num(robust.get("role_profile_mean_abs_delta")),
                "note": "mean absolute judge difference",
            },
        ]
        rel = _write_svg(
            out_dir,
            "qwen_sample_agreement.svg",
            _metric_cards(
                metrics,
                "Stratified Qwen sensitivity sample",
                "The sample over-represents refusal cases; post-stratified metrics estimate full-run rates.",
            ),
        )
        figures.append({"title": "Alternate-judge sample", "path": rel, "caption": "Judge robustness is sampled and post-stratified, not claimed as a full alternate-judge rescoring."})

    design_rows = v2.get("role_profile_design", {}).get("by_dimension", [])
    if design_rows:
        rel = _write_svg(out_dir, "role_profile_design.svg", _role_design_chart(design_rows))
        figures.append({"title": "Role-card design", "path": rel, "caption": "The role cards are multi-dimensional; a single role-discretion slope is too blunt."})

    frontier = summary.get("v2_frontier", {})
    if frontier.get("available"):
        rows = [
            {
                "population": "local clean",
                "refusal_rate": v2.get("overall", {}).get("refusal_rate"),
                "over_refusal_rate": v2.get("overall", {}).get("over_refusal_rate"),
            },
            {
                "population": "frontier exploratory",
                "refusal_rate": frontier.get("overall", {}).get("refusal_rate"),
                "over_refusal_rate": frontier.get("overall", {}).get("over_refusal_rate"),
            },
        ]
        rel = _write_svg(
            out_dir,
            "frontier_vs_local_refusal.svg",
            _bar_chart(
                rows,
                title="Local citable vs frontier exploratory",
                subtitle="Frontier results are separated because Grok is also the primary judge.",
                label_key="population",
                series=[
                    ("refusal", "refusal_rate", PALETTE["blue"]),
                    ("over-refusal", "over_refusal_rate", PALETTE["orange"]),
                ],
                value_max=max([row.get("refusal_rate") or 0 for row in rows] + [0.01]),
            ),
        )
        figures.append({"title": "Local vs frontier", "path": rel, "caption": "The frontier arm is useful for exploration, not pooled evidence."})

    return figures
