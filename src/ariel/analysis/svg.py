"""SVG line and bar charts for analysis reports.

Charts use light/dark CSS palettes, a shared y axis and `<title>` tooltips.
Multiple series get a legend.
"""

from __future__ import annotations

from html import escape
import math

SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70")

_STYLE = """<style>
svg{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;{light}}}
@media (prefers-color-scheme: dark){{svg{{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#383835;{dark}}}}}
text{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;fill:var(--ink2);font-size:11px}}
.title{{fill:var(--ink);font-size:13px;font-weight:600}}
.grid{{stroke:var(--grid);stroke-width:1}}
.axis{{stroke:var(--ink2);stroke-width:1}}
</style>"""


def _style() -> str:
    light = "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES_LIGHT))
    dark = "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES_DARK))
    return _STYLE.format(light=light, dark=dark)


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return []
    if high <= low:
        return [low]
    raw = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(low / step) * step
    ticks, value = [], start
    while value <= high + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt(value: float) -> str:
    if abs(value) >= 1000 and float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.3g}"


def _frame(width, height, title, xlabel, ylabel, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{escape(title)}">', _style(),
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>',
        f'<text class="title" x="12" y="20">{escape(title)}</text>',
        f'<text x="{width / 2}" y="{height - 6}" text-anchor="middle">{escape(xlabel)}</text>',
        f'<text transform="translate(12 {height / 2}) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>',
        *body, "</svg>", ""])


def _empty(width, height, title, xlabel, ylabel, message="no data") -> str:
    return _frame(width, height, title, xlabel, ylabel,
                  [f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle">{escape(message)}</text>'])


def line_chart(series: list[dict], *, title: str, xlabel: str, ylabel: str, width: int = 640,
               height: int = 300, y_range: tuple[float, float] | None = None) -> str:
    """series: [{"label", "x": [...], "y": [...]}] (None y values break the line). At most three series."""
    if len(series) > len(SERIES_LIGHT):
        raise ValueError(f"at most {len(SERIES_LIGHT)} series per chart; split into separate charts")
    pairs = [[(float(x), float(y)) for x, y in zip(s["x"], s["y"]) if y is not None and math.isfinite(y)]
             for s in series]
    flat = [p for ps in pairs for p in ps]
    if not flat:
        return _empty(width, height, title, xlabel, ylabel)
    left, right, top, bottom = 56, 16, 34 + (18 if len(series) > 1 else 0), 40
    xs, ys = [p[0] for p in flat], [p[1] for p in flat]
    x0, x1 = min(xs), max(xs)
    y0, y1 = y_range or (min(ys), max(ys))
    if x1 == x0:
        x0, x1 = x0 - 1, x1 + 1
    if y1 == y0:
        y0, y1 = y0 - 1, y1 + 1
    yticks = _ticks(y0, y1)
    y0, y1 = min(y0, yticks[0]), max(y1, yticks[-1])
    plot_w, plot_h = width - left - right, height - top - bottom
    sx = lambda v: left + (v - x0) / (x1 - x0) * plot_w                       # noqa: E731
    sy = lambda v: top + (1 - (v - y0) / (y1 - y0)) * plot_h                  # noqa: E731
    body = []
    for t in yticks:
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        body.append(f'<text x="{left - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
    for t in _ticks(x0, x1, 6):
        if x0 <= t <= x1:
            body.append(f'<text x="{sx(t):.1f}" y="{height - bottom + 16}" text-anchor="middle">{_fmt(t)}</text>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    for i, (s, ps) in enumerate(zip(series, pairs)):
        color = f"var(--s{i})"
        segments, current = [], []
        for x, y in zip(s["x"], s["y"]):
            if y is None or not math.isfinite(y):
                if current:
                    segments.append(current)
                current = []
            else:
                current.append((sx(float(x)), sy(float(y))))
        if current:
            segments.append(current)
        for seg in segments:
            if len(seg) == 1:
                body.append(f'<circle cx="{seg[0][0]:.1f}" cy="{seg[0][1]:.1f}" r="3" fill="{color}"/>')
            else:
                d = " ".join(f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}" for j, (x, y) in enumerate(seg))
                body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" '
                            'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in ps:
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="transparent">'
                        f'<title>{escape(s["label"])}: {_fmt(y)} at {xlabel} {_fmt(x)}</title></circle>')
        if len(series) > 1:
            lx = left + i * 170
            body.append(f'<line x1="{lx}" x2="{lx + 16}" y1="{top - 14}" y2="{top - 14}" stroke="{color}" '
                        'stroke-width="2"/>')
            body.append(f'<text x="{lx + 22}" y="{top - 10}">{escape(s["label"])}</text>')
    return _frame(width, height, title, xlabel, ylabel, body)


def bar_chart(bars: list[dict], *, title: str, ylabel: str, xlabel: str = "", width: int = 640,
              height: int = 300) -> str:
    """bars: [{"label", "value", "lo", "hi", "n"}]; lo/hi draw an interval whisker when present."""
    values = [b["value"] for b in bars if b.get("value") is not None]
    if not values:
        return _empty(width, height, title, xlabel, ylabel)
    extent = values + [b[k] for b in bars for k in ("lo", "hi") if b.get(k) is not None]
    y0, y1 = min(0.0, min(extent)), max(0.0, max(extent))
    if y1 == y0:
        y1 = y0 + 1
    yticks = _ticks(y0, y1)
    y0, y1 = min(y0, yticks[0]), max(y1, yticks[-1])
    left, right, top, bottom = 56, 16, 34, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    sy = lambda v: top + (1 - (v - y0) / (y1 - y0)) * plot_h                  # noqa: E731
    slot = plot_w / len(bars)
    bar_w = min(48.0, slot * 0.6)
    body = []
    for t in yticks:
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        body.append(f'<text x="{left - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
    for i, b in enumerate(bars):
        cx = left + slot * (i + 0.5)
        label = f'{b["label"]} (n={b.get("n", "?")})'
        body.append(f'<text x="{cx:.1f}" y="{height - bottom + 16}" text-anchor="middle">{escape(label)}</text>')
        if b.get("value") is None:
            body.append(f'<text x="{cx:.1f}" y="{sy(0) - 6:.1f}" text-anchor="middle">no data</text>')
            continue
        top_y, base_y = sorted((sy(b["value"]), sy(0)))
        tip = f'{b["label"]}: {_fmt(b["value"])}'
        if b.get("lo") is not None and b.get("hi") is not None:
            tip += f' [{_fmt(b["lo"])}, {_fmt(b["hi"])}]'
        body.append(f'<rect x="{cx - bar_w / 2:.1f}" y="{top_y:.1f}" width="{bar_w:.1f}" '
                    f'height="{max(base_y - top_y, 1):.1f}" rx="4" fill="var(--s0)"><title>{escape(tip)}, '
                    f'n={b.get("n", "?")}</title></rect>')
        if b.get("lo") is not None and b.get("hi") is not None:
            lo, hi = sy(b["lo"]), sy(b["hi"])
            body.append(f'<path d="M{cx:.1f},{lo:.1f} L{cx:.1f},{hi:.1f} M{cx - 6:.1f},{lo:.1f} L{cx + 6:.1f},'
                        f'{lo:.1f} M{cx - 6:.1f},{hi:.1f} L{cx + 6:.1f},{hi:.1f}" stroke="var(--ink)" '
                        'stroke-width="1.5" fill="none"/>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    return _frame(width, height, title, xlabel, ylabel, body)
