"""LaserFX Effect Engine: coherent aerial choreography (Laser Sky, Tunnels, Cones, Crowns,
Lissajous, Helix Waves, Hot-Beam Matrices, Wide-Coverage Blackout Chasers, Single-Pass Beaded
Planes, Mirrored-Pair Beam Chases), transient-triggered white Breakdown bursts, color-theory
harmony palettes, Art-Net RGB override, and vectorized path sorting.
"""
from __future__ import annotations

import colorsys
import math
import time
from dataclasses import dataclass

import numpy as np

from .artnet import ArtNetState
from .audio import BREAKDOWN, BUILD, DROP, MAIN, SILENCE, Features

TAU = 2.0 * math.pi
WHITE = np.array([1.0, 1.0, 1.0], dtype=float)
LIM = 0.94


@dataclass
class Path:
    pts: np.ndarray            # (V, 2)
    rgb: np.ndarray            # (3,) or (V, 3)
    dwell: int = 0


def hsv(h: float, s: float = 1.0, v: float = 1.0) -> np.ndarray:
    return np.array(
        colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)),
        dtype=float,
    )


def rgb_to_hsv(rgb: tuple[float, float, float] | np.ndarray) -> tuple[float, float, float]:
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    return colorsys.rgb_to_hsv(
        min(max(r, 0.0), 1.0),
        min(max(g, 0.0), 1.0),
        min(max(b, 0.0), 1.0),
    )


# ---------------------------------------------------------------- COLOR THEORY HARMONY ENGINE

HARMONY_MODES: list[tuple[str, str]] = [
    ("auto", "Auto Harmony"),
    ("complementary", "Complementary (180°)"),
    ("split_complementary", "Split-Complementary"),
    ("rectangular", "Rectangular / Tetradic"),
    ("triadic", "Triadic (120°)"),
    ("analogous", "Analogous (±32°)"),
]

HARMONY_OFFSETS: dict[str, tuple[float, ...]] = {
    "complementary": (0.0, 0.50),
    "split_complementary": (0.0, 150.0 / 360.0, 210.0 / 360.0),
    "rectangular": (0.0, 60.0 / 360.0, 180.0 / 360.0, 240.0 / 360.0),
    "triadic": (0.0, 120.0 / 360.0, 240.0 / 360.0),
    "analogous": (0.0, 32.0 / 360.0, -32.0 / 360.0),
}

ANCHOR_HUES_DEG = (188.0, 0.0, 275.0, 112.0, 225.0, 32.0, 312.0, 165.0)


def build_harmony_palette(
    base_hue: float,
    mode: str = "complementary",
    artnet: ArtNetState | None = None,
) -> list[np.ndarray]:
    resolved_mode = mode if mode in HARMONY_OFFSETS else "complementary"

    if artnet is not None and artnet.connected:
        if artnet.harmony_override in HARMONY_OFFSETS:
            resolved_mode = artnet.harmony_override
        dim = min(max(artnet.dimmer, 0.0), 1.0)
        c1 = np.array(artnet.rgb1, dtype=float) * dim
        if float(c1.max()) < 0.02 and artnet.rgb2 is None:
            return [np.zeros(3, dtype=float), np.zeros(3, dtype=float)]

        if artnet.rgb2 is not None:
            c2 = np.array(artnet.rgb2, dtype=float) * dim
            h1, s1, v1 = rgb_to_hsv(c1)
            h2, s2, v2 = rgb_to_hsv(c2)
            mid_v = max(v1, v2, 0.25)
            return [
                c1,
                c2,
                hsv((h1 + 0.5) % 1.0, max(s1, 0.85), mid_v),
                hsv((h2 + 0.5) % 1.0, max(s2, 0.85), mid_v),
            ]

        h1, s1, v1 = rgb_to_hsv(c1)
        if s1 < 0.18:
            acc_h = artnet.accent_hue if artnet.accent_hue is not None else (base_hue % 1.0)
            return [c1] + [hsv(acc_h + off, 1.0, v1) for off in HARMONY_OFFSETS[resolved_mode]]

        if artnet.accent_hue is not None:
            h2 = artnet.accent_hue % 1.0
            return [
                c1,
                hsv(h2, max(s1, 0.92), v1),
                hsv((h1 + 0.5) % 1.0, max(s1, 0.90), v1),
                hsv((h2 + 0.5) % 1.0, max(s1, 0.90), v1),
            ]

        return [c1] + [hsv(h1 + off, max(s1, 0.94), v1) for off in HARMONY_OFFSETS[resolved_mode][1:]]

    anchor_idx = int(round(abs(base_hue * 10.0))) % len(ANCHOR_HUES_DEG)
    h0 = (ANCHOR_HUES_DEG[anchor_idx] / 360.0) + (base_hue - round(base_hue, 1)) * 0.15
    return [hsv(h0 + off, 1.0, 1.0) for off in HARMONY_OFFSETS[resolved_mode]]


def _apply_split_color_harmony(
    paths: list[Path],
    palette: list[np.ndarray],
    artnet: ArtNetState | None,
) -> list[Path]:
    if not paths or not palette:
        return paths

    artnet_on = artnet is not None and artnet.connected
    if artnet_on and artnet.dimmer <= 0.01:
        return [Path(p.pts, np.zeros(3, dtype=float), p.dwell) for p in paths]

    n_cols = len(palette)
    out: list[Path] = []
    for idx, p in enumerate(paths):
        arr = np.asarray(p.rgb, dtype=float)
        if arr.ndim == 1:
            lum = float(arr.max())
            if lum < 0.01:
                out.append(p)
                continue
            sat = float((arr.max() - arr.min()) / max(lum, 1e-6))
            if sat < 0.05 and not artnet_on:
                out.append(p)
                continue
            if (sat < 0.35 or p.dwell >= 4) and lum > 0.65 and not artnet_on:
                out.append(p)
                continue
            target = palette[idx % n_cols] * lum
            if sat < 0.55:
                white_mix = (0.55 - sat) / 0.55
                target = target * (1.0 - 0.65 * white_mix) + (WHITE * lum) * (0.65 * white_mix)
            out.append(Path(p.pts, np.clip(target, 0.0, 1.0), p.dwell))
        else:
            lum = np.max(arr, axis=1, keepdims=True)
            sat = (np.max(arr, axis=1, keepdims=True) - np.min(arr, axis=1, keepdims=True)) / np.maximum(lum, 1e-6)
            c_a = palette[idx % n_cols]
            c_b = palette[(idx + 1) % n_cols]
            u = np.linspace(0.0, 1.0, len(arr))[:, None]
            grad = (c_a * (1.0 - u) + c_b * u) * lum
            # Preserve embedded white-hot beads along beaded paths
            is_white_bead = (sat < 0.12) & (lum > 0.01)
            final_rgb = np.where(is_white_bead, WHITE * lum, grad)
            out.append(Path(p.pts, np.clip(final_rgb, 0.0, 1.0), p.dwell))
    return out


# ---------------------------------------------------------------- GEOMETRIC PRIMITIVES

def line(a, b, rgb, dwell: int = 0) -> Path:
    return Path(np.array([a, b], dtype=float), rgb, dwell)


def hot_dot(pt, rgb=WHITE, dwell: int = 5) -> Path:
    """Stationary white-core aerial hot point without stray spatial offset."""
    x = float(np.clip(pt[0], -LIM, LIM))
    y = float(np.clip(pt[1], -LIM, LIM))
    arr = np.asarray(rgb, dtype=float)
    lum = float(arr.max()) if arr.size else 1.0
    if lum > 0.01:
        sat = float((arr.max() - arr.min()) / max(lum, 1e-6))
        if sat < 0.05:
            core = np.clip(WHITE * lum, 0.0, 1.0)
        else:
            core = np.clip(arr * 0.25 + (WHITE * max(lum, 0.92)) * 0.75, 0.0, 1.0)
    else:
        core = arr
    return Path(np.array([[x, y], [x, y]], dtype=float), core, max(int(dwell), 4))


def beaded_line(
    a,
    b,
    line_rgb,
    bead_rgb=WHITE,
    count: int = 4,
    bead_dwell: int = 4,
    phase: float = 0.0,
    margin: float = 0.10,
) -> Path:
    """Single continuous beam path with evenly spaced, stationary white-hot bead dwell points
    baked directly into the polyline vertices (no second pass or separate dot paths)."""
    p0 = np.asarray(a, dtype=float)
    p1 = np.asarray(b, dtype=float)
    vec = p1 - p0
    seg_len = float(np.hypot(vec[0], vec[1]))
    c_line = np.asarray(line_rgb, dtype=float)
    c_bead = np.asarray(bead_rgb, dtype=float)

    n = max(int(count), 1)
    if seg_len < 0.18 or n <= 0:
        return Path(np.array([p0, p1], dtype=float), c_line, dwell=0)

    # Keep beads evenly spaced and inset from endpoints so corners/edges stay clean
    span = max(1.0 - 2.0 * margin, 0.25)
    u = ((np.arange(n, dtype=float) + 0.5 + (phase % 1.0)) / n) % 1.0
    ts = np.sort(margin + u * span)

    # Deduplicate any beads that wrap close together
    filtered_ts: list[float] = []
    min_gap = 0.06 / max(seg_len, 0.25)
    for t in ts:
        if not filtered_ts or (t - filtered_ts[-1]) >= min_gap:
            filtered_ts.append(float(t))

    pts_list: list[np.ndarray] = [p0]
    rgb_list: list[np.ndarray] = [c_line]
    dwell_reps = max(int(bead_dwell), 2)

    for t in filtered_ts:
        pt = p0 + vec * t
        # Snap color to white at the exact coordinate, dwell stationary, then restore beam color
        pts_list.append(pt)
        rgb_list.append(c_line)
        for _ in range(dwell_reps):
            pts_list.append(pt)
            rgb_list.append(c_bead)
        pts_list.append(pt)
        rgb_list.append(c_line)

    pts_list.append(p1)
    rgb_list.append(c_line)

    return Path(
        np.clip(np.asarray(pts_list, dtype=float), -LIM, LIM),
        np.clip(np.asarray(rgb_list, dtype=float), 0.0, 1.0),
        dwell=0,
    )


def dotted_line(a, b, rgb, dots: int = 5, dot_len: float = 0.036, phase: float = 0.0) -> list[Path]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vec = b - a
    total = float(np.hypot(vec[0], vec[1]))
    if total < 1e-4:
        return []
    frac_len = min(dot_len / total, 0.40 / max(dots, 1))
    out: list[Path] = []
    for i in range(dots):
        t0 = ((i + (phase % 1.0)) / dots) % 1.0
        t1 = min(t0 + frac_len, 1.0)
        if t1 <= t0 + 0.005:
            continue
        p0 = a + vec * t0
        p1 = a + vec * t1
        if i % 2 == 1:
            p0, p1 = p1, p0
        out.append(line(p0, p1, rgb, dwell=0))
    return out


def dashed_beam(a, b, rgb, segments: int = 3, duty: float = 0.65, phase: float = 0.0) -> list[Path]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out: list[Path] = []
    for i in range(segments):
        t0 = ((i + phase) % segments) / segments
        t1 = min(t0 + duty / segments, 1.0)
        if t1 <= t0 + 0.04:
            continue
        p0 = a + (b - a) * t0
        p1 = a + (b - a) * t1
        if i % 2 == 1:
            p0, p1 = p1, p0
        out.append(line(p0, p1, rgb, dwell=0))
    return out


def ray(ox: float, oy: float, ang: float, length: float = 2.6, bound: float = LIM):
    dx, dy, t = math.sin(ang), math.cos(ang), length
    if dx > 1e-9:
        t = min(t, (bound - ox) / dx)
    elif dx < -1e-9:
        t = min(t, (-bound - ox) / dx)
    if dy > 1e-9:
        t = min(t, (bound - oy) / dy)
    elif dy < -1e-9:
        t = min(t, (-bound - oy) / dy)
    t = max(t, 0.02)
    return (ox, oy), (ox + dx * t, oy + dy * t)


def clipped_plane(cx: float, cy: float, angle: float, span: float = LIM):
    nx, ny = math.cos(angle), math.sin(angle)
    tx, ty = -ny, nx
    t_min, t_max = -4.0, 4.0
    if abs(tx) > 1e-9:
        t1, t2 = (-span - cx) / tx, (span - cx) / tx
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
    if abs(ty) > 1e-9:
        t1, t2 = (-span - cy) / ty, (span - cy) / ty
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
    if t_max <= t_min + 0.08:
        return None
    return (cx + tx * t_min, cy + ty * t_min), (cx + tx * t_max, cy + ty * t_max)


def polygon_path(cx: float, cy: float, rx: float, ry: float, sides: int, rot: float, rgb) -> Path:
    angs = np.linspace(0.0, TAU, sides + 1) + rot
    xs = np.clip(cx + rx * np.cos(angs), -LIM, LIM)
    ys = np.clip(cy + ry * np.sin(angs), -LIM, LIM)
    return Path(np.stack([xs, ys], axis=1), rgb, dwell=0)


def star_polygon_path(cx: float, cy: float, r_outer: float, r_inner: float, points: int, rot: float, rgb) -> Path:
    """Closed star/crown polygon alternating between outer and inner radii."""
    n = points * 2 + 1
    angs = np.linspace(0.0, TAU, n) + rot
    radii = np.where(np.arange(n) % 2 == 0, r_outer, r_inner)
    xs = np.clip(cx + radii * np.cos(angs), -LIM, LIM)
    ys = np.clip(cy + radii * np.sin(angs), -LIM, LIM)
    return Path(np.stack([xs, ys], axis=1), rgb, dwell=0)


def _reverse_path(p: Path) -> Path:
    pts = p.pts[::-1].copy()
    rgb = p.rgb[::-1].copy() if np.asarray(p.rgb).ndim == 2 else p.rgb
    return Path(pts, rgb, p.dwell)


def _optimize_path_order(paths: list[Path], tol: float = 0.025) -> list[Path]:
    """Deduplicate overlapping straight segments and sort remaining paths via vectorized
    nearest-neighbor traversal to minimize blanking jumps."""
    if not paths:
        return []

    deduped: list[Path] = []
    sigs_a: list[np.ndarray] = []
    sigs_b: list[np.ndarray] = []
    for p in paths:
        if p.dwell < 3 and len(p.pts) == 2:
            a, b = p.pts[0], p.pts[1]
            if float(np.hypot(b[0] - a[0], b[1] - a[1])) >= 0.03:
                if sigs_a:
                    ka = np.asarray(sigs_a)
                    kb = np.asarray(sigs_b)
                    d_same = np.maximum(np.hypot(ka[:, 0] - a[0], ka[:, 1] - a[1]), np.hypot(kb[:, 0] - b[0], kb[:, 1] - b[1]))
                    d_flip = np.maximum(np.hypot(kb[:, 0] - a[0], kb[:, 1] - a[1]), np.hypot(ka[:, 0] - b[0], ka[:, 1] - b[1]))
                    if float(np.minimum(d_same, d_flip).min()) < tol:
                        continue
                sigs_a.append(a)
                sigs_b.append(b)
        deduped.append(p)

    n = len(deduped)
    if n <= 1:
        return deduped

    starts = np.array([p.pts[0] for p in deduped], dtype=float)
    ends = np.array([p.pts[-1] for p in deduped], dtype=float)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    ordered: list[Path] = [deduped[0]]
    cur_pos = ends[0]

    for _ in range(n - 1):
        d_fwd = np.hypot(starts[:, 0] - cur_pos[0], starts[:, 1] - cur_pos[1])
        d_rev = np.hypot(ends[:, 0] - cur_pos[0], ends[:, 1] - cur_pos[1])
        d_fwd[visited] = 1e9
        d_rev[visited] = 1e9
        best_fwd = int(np.argmin(d_fwd))
        best_rev = int(np.argmin(d_rev))
        if d_rev[best_rev] < d_fwd[best_fwd]:
            visited[best_rev] = True
            chosen = _reverse_path(deduped[best_rev])
            cur_pos = starts[best_rev]
        else:
            visited[best_fwd] = True
            chosen = deduped[best_fwd]
            cur_pos = ends[best_fwd]
        ordered.append(chosen)

    return ordered


def smooth_wave(bt: float, period_beats: float) -> float:
    return -math.cos(bt * TAU / period_beats)


def beat_sweep(bt: float, beats_per_pass: float = 1.0, alternate: bool = True) -> float:
    if alternate:
        return 0.5 - 0.5 * math.cos(math.pi * bt / beats_per_pass)
    idx = int(math.floor(bt / beats_per_pass))
    frac = (bt / beats_per_pass) - idx
    return 0.5 - 0.5 * math.cos(math.pi * frac)


def beat_pump(c: Ctx) -> float:
    return min(max(0.55 * c.pulse + 0.45 * c.kick, 0.0), 1.0)


def strobe_gate(c: Ctx, div: float = 4.0, duty: float = 0.48) -> bool:
    return (not c.strobe) or ((c.bt * div) % 1.0) < duty


def _ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lane_x(i: int, lanes: int) -> float:
    return -0.84 + 1.68 * i / max(lanes - 1, 1)


def _clip(v):
    return np.clip(v, -LIM, LIM)


@dataclass
class Ctx:
    f: Features
    bt: float
    hue: float
    rng: np.random.Generator
    rt: float = 0.0
    strobe: bool = True
    detail: float = 0.5
    harmony_colors: list[np.ndarray] | None = None

    @property
    def energy(self) -> float:
        return min(max(self.f.level, 0.0), 1.0)

    @property
    def pulse(self) -> float:
        return self.f.beat_pulse

    @property
    def kick(self) -> float:
        return self.f.kick

    @property
    def hi(self) -> float:
        return min(max(getattr(self.f, "hi_flux", 0.0), 0.0), 1.0)

    def count(self, lo: int, hi: int, x: float = 1.0) -> int:
        return int(round(lo + (hi - lo) * min(max(self.detail * x, 0.0), 1.0)))

    def palette(self, idx: int = 0, v: float = 1.0, white_boost: float = 0.0) -> np.ndarray:
        cols = self.harmony_colors if self.harmony_colors else [hsv(0.52, 1.0, 1.0), hsv(0.02, 1.0, 1.0)]
        base = cols[idx % len(cols)] * v
        if white_boost > 0.0:
            wb = min(max(white_boost, 0.0), 1.0)
            base = base * (1.0 - wb) + (WHITE * v) * wb
        return np.clip(base, 0.0, 1.0)


# ======================================================================== CALIBRATION TEST PATTERNS

TEST_PATTERNS = {
    "ilda_grid": "ILDA Tuning Grid",
    "corner_box": "Corner Sharpness Box",
    "rgb_align": "RGB Color Alignment",
}


def render_test_pattern(kind: str, bt: float) -> list[Path]:
    if kind == "corner_box":
        s = 0.84
        inner = 0.44
        outer_pts = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]], float)
        inner_pts = np.array([[-inner, -inner], [inner, -inner], [inner, inner], [-inner, inner], [-inner, -inner]], float)
        return [
            Path(outer_pts, WHITE, dwell=0),
            Path(inner_pts, hsv(0.54, 1.0, 1.0), dwell=0),
            line((-s, -s), (s, s), hsv(0.0, 1.0, 1.0), dwell=0),
            line((-s, s), (s, -s), hsv(0.33, 1.0, 1.0), dwell=0),
        ]

    if kind == "rgb_align":
        s = 0.82
        out = [Path(np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]], float), WHITE, dwell=0)]
        cols = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.3, 1.0]),
            WHITE,
        ]
        for idx, x in enumerate((-0.52, -0.17, 0.17, 0.52)):
            a, b = (x, -0.68), (x, 0.68)
            if idx % 2 == 1:
                a, b = b, a
            out.append(line(a, b, cols[idx], dwell=0))
        return out

    s = 0.88
    c_inner = 0.44
    border = np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]], float)
    box = np.array([[-c_inner, -c_inner], [c_inner, -c_inner], [c_inner, c_inner], [-c_inner, c_inner], [-c_inner, -c_inner]], float)
    circle = polygon_path(0.0, 0.0, s * 0.92, s * 0.92, 24, 0.0, hsv(0.33, 1.0, 1.0))
    return [
        Path(border, WHITE, dwell=0),
        circle,
        Path(box, hsv(0.54, 1.0, 1.0), dwell=0),
        line((-s, 0.0), (s, 0.0), hsv(0.0, 1.0, 1.0), dwell=0),
        line((0.0, -s), (0.0, s), hsv(0.14, 1.0, 1.0), dwell=0),
    ]


# ======================================================================== DROP OPENING IMPACTS

DROP_IMPACT_NAMES = [
    "drop_starburst",
    "drop_horizon_split",
    "drop_tunnel_shockwave",
    "drop_prism_cross",
]


def _drop_impact_effect(c: Ctx, progress: float, variant: int) -> list[Path]:
    fade = max(0.0, 1.0 - 0.25 * progress)
    col_main = WHITE * fade
    col_acc = c.palette(0, v=fade, white_boost=0.45)
    eased = math.sin(min(progress, 1.0) * math.pi * 0.5)
    out: list[Path] = []

    if variant == 0:
        n_rays = c.count(6, 8)
        spin = 0.28 * progress
        reach = 0.45 + 1.85 * eased
        for i in range(n_rays):
            ang = spin + (i / max(n_rays, 1)) * TAU
            p0, p1 = ray(0.0, 0.0, ang, reach)
            out.append(line(p0, p1, col_main if i == 0 else c.palette(i, v=fade), dwell=0))

    elif variant == 1:
        for idx, frac in enumerate((0.35, 0.85)):
            y = 0.14 + 0.74 * eased * frac
            out.append(line((-LIM, y), (LIM, y), c.palette(idx * 2, v=fade), dwell=0))
            out.append(line((LIM, -y), (-LIM, -y), c.palette(idx * 2 + 1, v=fade), dwell=0))

    elif variant == 2:
        for idx, r0 in enumerate((0.25, 0.55)):
            r = min(0.18 + 0.74 * eased + r0 * 0.35, LIM)
            out.append(polygon_path(0.0, 0.0, r, r * 0.88, 16, progress * 0.4, c.palette(idx, v=fade)))

    else:
        span = 0.15 + 0.76 * eased
        out.extend([
            line((-span, -LIM), (-span, LIM), c.palette(0, v=fade), dwell=0),
            line((span, LIM), (span, -LIM), c.palette(1, v=fade), dwell=0),
            line((-LIM, span), (LIM, span), c.palette(2, v=fade), dwell=0),
            line((LIM, -span), (-LIM, -span), col_acc, dwell=0),
        ])

    return out


# ======================================================================== LASER SKY SCENES (BREAKDOWN)

def sc_beaded_sky_canopy(c: Ctx) -> list[Path]:
    """Dual breathing Laser Sky sheets with single-pass embedded white-hot beam beads."""
    y_base = 0.24 + 0.14 * smooth_wave(c.bt, 28.0)
    tilt = 0.20 * smooth_wave(c.bt, 16.0)
    beads = c.count(4, 6)
    out: list[Path] = []
    for idx, gap in enumerate((-0.18, 0.18)):
        s = 1.0 if idx == 0 else -1.0
        a = (-LIM, float(np.clip(y_base + gap - s * tilt, -LIM, LIM)))
        b = (LIM, float(np.clip(y_base + gap + s * tilt, -LIM, LIM)))
        out.append(beaded_line(
            a, b,
            line_rgb=c.palette(idx, v=0.72),
            bead_rgb=WHITE,
            count=beads,
            bead_dwell=4,
            phase=c.bt * 0.12 * s,
        ))
    return out


def sc_liquid_sky_drift(c: Ctx) -> list[Path]:
    """Multi-tier Liquid Sky: 2–3 layered rolling atmospheric sheets with phase-offset pitch waves."""
    planes = c.count(2, 3)
    y_base = 0.22 + 0.14 * smooth_wave(c.bt, 32.0)
    out: list[Path] = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * 0.28
        tilt = 0.22 * math.sin(c.bt * TAU / 16.0 + p * 1.15)
        y0 = float(np.clip(y_base + gap - tilt, -LIM, LIM))
        y1 = float(np.clip(y_base + gap + tilt, -LIM, LIM))
        out.append(line((-LIM, y0), (LIM, y1), c.palette(p, v=0.90), dwell=0))
    return out


def sc_cathedral_sky_vault(c: Ctx) -> list[Path]:
    """Overarching V-ribbed canopy sheets creating a dramatic cathedral ceiling in haze."""
    ribs = c.count(2, 3)
    y_apex = 0.04 + 0.14 * smooth_wave(c.bt, 24.0)
    out: list[Path] = []
    for r in range(ribs):
        lift = 0.26 + r * 0.24 + 0.14 * math.sin(c.bt * TAU / 16.0 + r * 0.7)
        y_wing = float(np.clip(y_apex + lift, -LIM, LIM))
        y_tip = float(np.clip(y_apex + r * 0.14, -LIM, LIM))
        pts = np.array([[-LIM, y_wing], [0.0, y_tip], [LIM, y_wing]], float)
        out.append(Path(pts, c.palette(r, v=0.88), dwell=0))
    return out


def sc_ribbon_tide_sky(c: Ctx) -> list[Path]:
    """Dual counter-flowing sinusoidal liquid sky ribbons with traveling crest shimmer."""
    x = np.linspace(-LIM, LIM, 28)
    out: list[Path] = []
    for layer in range(2):
        dir_sign = 1.0 if layer == 0 else -1.0
        ph = dir_sign * (c.bt * TAU / 14.0) + layer * 1.6
        y_off = 0.30 - layer * 0.36 + 0.10 * smooth_wave(c.bt, 28.0)
        amp = 0.15 + 0.05 * math.sin(c.bt * TAU / 12.0 + layer)
        y = np.clip(y_off + amp * np.sin(1.6 * math.pi * x + ph), -LIM, LIM)
        out.append(Path(np.stack([x, y], axis=1), c.palette(layer, v=0.90), dwell=0))
    return out


def sc_prism_horizon_deck(c: Ctx) -> list[Path]:
    """Multi-plane horizon deck that breathes open and counter-tilts in harmonic layers."""
    decks = c.count(2, 3)
    spread = 0.18 + 0.24 * (0.5 + 0.5 * smooth_wave(c.bt, 16.0))
    y_mid = 0.18 + 0.12 * smooth_wave(c.bt, 32.0)
    out: list[Path] = []
    for i in range(decks):
        u = (i / max(decks - 1, 1)) - 0.5
        tilt = (0.18 if i % 2 == 0 else -0.18) * smooth_wave(c.bt, 14.0)
        y0 = float(np.clip(y_mid + u * 2.0 * spread - tilt, -LIM, LIM))
        y1 = float(np.clip(y_mid + u * 2.0 * spread + tilt, -LIM, LIM))
        out.append(line((-LIM, y0), (LIM, y1), c.palette(i, v=0.88), dwell=0))
    return out


def sc_morphing_plane_sky(c: Ctx) -> list[Path]:
    morph = smooth_wave(c.bt, 16.0)
    y_center = 0.22 + 0.14 * smooth_wave(c.bt, 24.0)
    tilt = 0.36 * morph
    return [
        line((-LIM, float(np.clip(y_center - tilt, -LIM, LIM))),
             (LIM, float(np.clip(y_center + tilt, -LIM, LIM))),
             c.palette(0, v=0.90), dwell=0),
        line((-LIM, float(np.clip(y_center + tilt * 0.85 - 0.32, -LIM, LIM))),
             (LIM, float(np.clip(y_center - tilt * 0.85 - 0.32, -LIM, LIM))),
             c.palette(1, v=0.86), dwell=0),
    ]


def sc_breathe_horizon_sky(c: Ctx) -> list[Path]:
    spread = 0.16 + 0.28 * (0.5 + 0.5 * smooth_wave(c.bt, 16.0))
    y_mid = 0.20 + 0.12 * smooth_wave(c.bt, 32.0)
    rock = 0.14 * smooth_wave(c.bt, 12.0)
    out: list[Path] = []
    for idx, sign in enumerate((-1.0, 1.0)):
        y0 = float(np.clip(y_mid + sign * spread - rock, -LIM, LIM))
        y1 = float(np.clip(y_mid + sign * spread + rock, -LIM, LIM))
        out.append(line((-LIM, y0), (LIM, y1), c.palette(idx, v=0.88), dwell=0))
    return out


def sc_velvet_sky_sweep(c: Ctx) -> list[Path]:
    u = beat_sweep(c.bt, beats_per_pass=8.0, alternate=True)
    x_center = -0.72 + 1.44 * u
    blades = c.count(2, 3)
    width = 0.34 + 0.10 * smooth_wave(c.bt, 16.0)
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * width
        x = float(np.clip(x_center + off, -LIM, LIM))
        out.append(line((x, -LIM), (x, LIM), c.palette(i, v=0.86), dwell=0))
    return out


def sc_silk_wave_sky(c: Ctx) -> list[Path]:
    x = np.linspace(-LIM, LIM, 28)
    ph = c.bt * TAU / 16.0
    y1 = np.clip(0.32 + 0.12 * smooth_wave(c.bt, 24.0) + 0.16 * np.sin(1.5 * math.pi * x + ph), -LIM, LIM)
    y2 = np.clip(-0.04 + 0.12 * smooth_wave(c.bt, 24.0) + 0.14 * np.cos(1.5 * math.pi * x - ph * 0.8), -LIM, LIM)
    return [
        Path(np.stack([x, y1], axis=1), c.palette(0, v=0.90), dwell=0),
        Path(np.stack([x, y2], axis=1), c.palette(1, v=0.82), dwell=0),
    ]


def sc_dual_horizon_mirage(c: Ctx) -> list[Path]:
    ang = 0.30 * smooth_wave(c.bt, 20.0)
    y_base = 0.24 + 0.12 * smooth_wave(c.bt, 32.0)
    return [
        line((-LIM, y_base - ang), (LIM, y_base + ang), c.palette(0, v=0.88), dwell=0),
        line((-LIM, y_base + ang * 0.65 - 0.34), (LIM, y_base - ang * 0.65 - 0.34), c.palette(1, v=0.84), dwell=0),
    ]


def sc_floating_dot_constellation(c: Ctx) -> list[Path]:
    y_sky = 0.12 + 0.12 * smooth_wave(c.bt, 24.0)
    tilt = 0.14 * smooth_wave(c.bt, 16.0)
    out: list[Path] = [line((-LIM, y_sky - tilt), (LIM, y_sky + tilt), c.palette(0, v=0.78), dwell=0)]
    n_stars = c.count(6, 8)
    for s in range(n_stars):
        u = (s + 0.5) / n_stars
        x = -0.84 + 1.68 * u + 0.05 * math.sin(c.bt * TAU / 16.0 + s)
        y = 0.46 + 0.24 * math.sin(c.bt * TAU / 20.0 + s * 1.7)
        out.append(hot_dot((x, y), WHITE, dwell=5))
    return out


def sc_slow_v_canopy_sky(c: Ctx) -> list[Path]:
    y_apex = 0.10 + 0.16 * smooth_wave(c.bt, 24.0)
    wing_lift = 0.30 + 0.24 * smooth_wave(c.bt, 16.0)
    y_wing = float(np.clip(y_apex + wing_lift, -LIM, LIM))
    return [
        line((-LIM, y_wing), (0.0, y_apex), c.palette(0, v=0.90), dwell=0),
        line((0.0, y_apex), (LIM, y_wing), c.palette(1, v=0.90), dwell=0),
    ]


def sc_aurora_sky_curtain(c: Ctx) -> list[Path]:
    x = np.linspace(-LIM, LIM, 28)
    out: list[Path] = []
    for layer in range(c.count(2, 3)):
        ph = c.bt * TAU / 18.0 + layer * 2.1
        y = 0.38 + 0.12 * smooth_wave(c.bt, 28.0) + 0.14 * np.sin(1.4 * math.pi * x + ph) - layer * 0.30
        pts = np.stack([x, np.clip(y, -LIM, LIM)], axis=1)
        out.append(Path(pts, c.palette(layer, v=0.88), dwell=0))
    return out


# ======================================================================== WHITE TRANSIENT BREAKDOWN BURSTS (MID-HIGH MELODIC / PERCUSSIVE)

BREAKDOWN_BURST_STYLES = (
    "starburst_scatter",
    "diagonal_slash",
    "hot_beam_cluster",
    "shock_ring",
    "crosshair_planes",
    "lightning_bolt",
    "triple_pillars",
)

BREAKDOWN_BURST_DURATION_BEATS = 0.32


def _apply_breakdown_transient_burst(
    paths: list[Path],
    progress: float,
    burst_params: dict,
    c: Ctx,
) -> list[Path]:
    """Pure white, quick-fade aerial burst triggered only on melodic or percussive mid-high sounds."""
    if progress < 0.0 or progress >= 1.0 or not burst_params:
        return paths

    if progress < 0.12:
        env = progress / 0.12
    else:
        rem = (1.0 - progress) / 0.88
        env = rem * rem
    if env <= 0.03:
        return paths

    style = burst_params.get("style", "starburst_scatter")
    cx = float(burst_params.get("cx", 0.0))
    cy = float(burst_params.get("cy", 0.0))
    ang = float(burst_params.get("ang", 0.4))
    scale = float(burst_params.get("scale", 1.0))
    variant = int(burst_params.get("variant", 0))
    seeds = burst_params.get("seeds", ())

    col_primary = WHITE * min(1.0, env)
    col_secondary = WHITE * min(1.0, 0.72 * env)
    extra: list[Path] = []

    if style == "starburst_scatter":
        n_pips = c.count(6, 10)
        grow = (0.12 + 0.58 * progress) * scale
        for i in range(n_pips):
            a = ang + (i / max(n_pips, 1)) * TAU + (seeds[i % len(seeds)] - 0.5) * 0.35
            r = grow * (0.65 + 0.55 * seeds[(i + 2) % len(seeds)])
            px = cx + r * math.cos(a)
            py = cy + r * 0.85 * math.sin(a)
            extra.append(hot_dot((px, py), col_primary if i % 2 == 0 else col_secondary, dwell=5))

    elif style == "diagonal_slash":
        seg = clipped_plane(cx, cy, ang, span=LIM)
        if seg is not None:
            extra.append(line(seg[0], seg[1], col_primary, dwell=0))
        seg2 = clipped_plane(cx + 0.18 * math.cos(ang), cy + 0.18 * math.sin(ang), ang + 0.12, span=LIM)
        if seg2 is not None and env > 0.25:
            extra.append(line(seg2[0], seg2[1], col_secondary, dwell=0))

    elif style == "hot_beam_cluster":
        n_dots = c.count(6, 9)
        for i in range(n_dots):
            sx = (seeds[(i * 2) % len(seeds)] * 2.0 - 1.0) * 0.86
            sy = (seeds[(i * 2 + 1) % len(seeds)] * 2.0 - 1.0) * 0.82
            extra.append(hot_dot((sx, sy), col_primary if i % 2 == 0 else col_secondary, dwell=5))

    elif style == "shock_ring":
        rad = min((0.10 + 0.74 * progress) * scale, LIM)
        sides = 6 if (variant % 2 == 0) else 12
        extra.append(polygon_path(cx * 0.75, cy * 0.75, rad, rad * 0.86, sides, ang + progress * 1.6, col_primary))

    elif style == "crosshair_planes":
        hx = float(np.clip(cx, -0.82, 0.82))
        hy = float(np.clip(cy, -0.82, 0.82))
        extra.append(line((-LIM, hy), (LIM, hy), col_primary, dwell=0))
        extra.append(line((hx, -LIM), (hx, LIM), col_secondary, dwell=0))

    elif style == "lightning_bolt":
        xs = np.linspace(-0.88, 0.88, 6)
        ys = [
            float(np.clip(cy + (seeds[idx_x % len(seeds)] - 0.5) * 0.68 * scale, -LIM, LIM))
            for idx_x in range(6)
        ]
        extra.append(Path(np.stack([xs, np.array(ys, dtype=float)], axis=1), col_primary, dwell=0))

    else:  # "triple_pillars"
        for i in range(3):
            px = float(np.clip(cx + (i - 1) * (0.28 + 0.26 * seeds[i % len(seeds)]), -LIM, LIM))
            tilt = (seeds[(i + 3) % len(seeds)] - 0.5) * 0.24
            extra.append(line((px - tilt, -LIM), (px + tilt, LIM), col_primary if i == 1 else col_secondary, dwell=0))

    return paths + extra


# ======================================================================== RAPID WIDE-COVERAGE BLACKOUT CHASING EFFECTS

def sc_blackout_zone_chase(c: Ctx) -> list[Path]:
    """Rapid 16th-note blackout chase jumping across 6 wide full-span spatial zones."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    if (step_f - step) > 0.62:
        return []
    pump = beat_pump(c)
    zone = step % 6
    col_a = c.palette(zone, v=1.0, white_boost=0.65 * pump)
    col_b = c.palette(zone + 1, v=1.0, white_boost=0.45 * pump)

    if zone == 0:
        return [line((-LIM, 0.78), (LIM, 0.62), col_a), line((LIM, 0.36), (-LIM, 0.48), col_b)]
    if zone == 1:
        return [line((-LIM, -0.78), (LIM, -0.62), col_a), line((LIM, -0.36), (-LIM, -0.48), col_b)]
    if zone == 2:
        return [line((-0.82, -LIM), (-0.66, LIM), col_a), line((-0.38, LIM), (-0.50, -LIM), col_b)]
    if zone == 3:
        return [line((0.82, -LIM), (0.66, LIM), col_a), line((0.38, LIM), (0.50, -LIM), col_b)]
    if zone == 4:
        return [line((-LIM, -0.84), (LIM, 0.84), col_a), line((-0.72, -LIM), (LIM, 0.56), col_b)]
    return [line((-LIM, 0.84), (LIM, -0.84), col_a), line((-0.72, LIM), (LIM, -0.56), col_b)]


def sc_quadrant_blackout_relay(c: Ctx) -> list[Path]:
    """High-speed 4-quadrant wide blackout relay (TL -> BR -> TR -> BL) with hard off-time."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    if (step_f - step) > 0.58:
        return []
    pump = beat_pump(c)
    q = (0, 3, 1, 2)[step % 4]
    sx = -1.0 if (q in (0, 2)) else 1.0
    sy = 1.0 if (q in (0, 1)) else -1.0
    col_a = c.palette(step, v=1.0, white_boost=0.70 * pump)
    col_b = c.palette(step + 2, v=1.0, white_boost=0.45 * pump)
    return [
        line((0.0, sy * 0.48), (sx * LIM, sy * 0.48), col_a),
        line((sx * 0.48, 0.0), (sx * 0.48, sy * LIM), col_b),
        line((0.0, 0.0), (sx * LIM, sy * LIM), col_a),
    ]


def sc_shutter_grid_chase(c: Ctx) -> list[Path]:
    """Rapid wide-coverage shutter chase stepping across 5 horizontal & 5 vertical lanes."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    if (step_f - step) > 0.60:
        return []
    pump = beat_pump(c)
    lane = step % 5
    pos = -0.82 + (lane / 4.0) * 1.64
    col_a = c.palette(lane, v=1.0, white_boost=0.65 * pump)
    col_b = c.palette(lane + 1, v=1.0, white_boost=0.40 * pump)
    off = float(np.clip(pos + (0.24 if pos < 0 else -0.24), -LIM, LIM))
    if (step // 5) % 2 == 0:
        return [line((-LIM, pos), (LIM, pos), col_a), line((LIM, off), (-LIM, off), col_b)]
    return [line((pos, -LIM), (pos, LIM), col_a), line((off, LIM), (off, -LIM), col_b)]


def sc_pingpong_blackout_whip(c: Ctx) -> list[Path]:
    """Wide full-span blackout whip snapping between extreme edges at 16th-note rate."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    if (step_f - step) > 0.56:
        return []
    pump = beat_pump(c)
    sign = 1.0 if (step % 2 == 0) else -1.0
    col_a = c.palette(step % 3, v=1.0, white_boost=0.72 * pump)
    col_b = c.palette((step + 1) % 3, v=1.0, white_boost=0.45 * pump)
    outer, inner = sign * 0.84, sign * 0.46
    if (step // 4) % 2 == 0:
        return [
            line((-LIM, outer), (LIM, outer * 0.92), col_a),
            line((LIM, inner), (-LIM, inner * 1.06), col_b),
        ]
    return [
        line((outer, -LIM), (outer * 0.92, LIM), col_a),
        line((inner, LIM), (inner * 1.06, -LIM), col_b),
    ]


# ======================================================================== MIRRORED-PAIR BEAM CHASES & SWEEPING BARS

def sc_beat_bar_sweep(c: Ctx) -> list[Path]:
    """On-beat sweeping bars: one eased pass of the field per beat with fading echo bars behind."""
    pump = beat_pump(c)
    bars = c.count(1, 2)
    lean = 0.14 * smooth_wave(c.bt, 8.0)
    gap = 0.30
    out: list[Path] = []
    for k in range(3 if c.detail >= 0.4 else 2):
        u = beat_sweep(c.bt - 0.07 * k, 1.0, True)
        xc = -0.80 + 1.60 * u
        v = (1.0, 0.45, 0.22)[k]
        for i in range(bars):
            x = xc + (i - (bars - 1) / 2.0) * gap
            col = c.palette(i, v=v, white_boost=0.55 * pump if k == 0 else 0.0)
            out.append(line((float(_clip(x - lean)), -LIM), (float(_clip(x + lean)), LIM), col))
    return out


def _lane_chase(
    c: Ctx,
    step_f: float,
    duty: float | None = None,
    lanes: int = 8,
    tail: int = 2,
    boost: float = 0.0,
) -> list[Path]:
    """A lit bar hopping lane to lane (bouncing at the edges), leaving a short fading tail."""
    step = int(math.floor(step_f))
    if duty is not None and c.strobe and (step_f - step) > duty:
        return []
    pump = beat_pump(c)
    span = 2 * lanes - 2
    k = step % span
    head = k if k < lanes else span - k
    dirn = 1 if k < lanes - 1 else -1
    pal = (step // lanes) % 4
    out: list[Path] = []
    for j in range(tail + 1):
        i = head - dirn * j
        if not 0 <= i < lanes:
            continue
        v = 1.0 if j == 0 else (0.5, 0.25, 0.12)[min(j - 1, 2)]
        col = c.palette(pal, v=v, white_boost=boost * pump if j == 0 else 0.0)
        x = _lane_x(i, lanes)
        a, b = (x, -LIM), (x, LIM)
        out.append(line(*((a, b) if j % 2 == 0 else (b, a)), col))
    return out


def sc_bar_chase_lanes(c: Ctx) -> list[Path]:
    """Running-light chase across 8 vertical lanes at 8th notes."""
    return _lane_chase(c, c.bt * 2.0, tail=c.count(1, 3), boost=0.4)


def sc_drop_lane_chase(c: Ctx) -> list[Path]:
    """Running-light chase at 16th notes with hard blackout gaps and white-hot heads."""
    return _lane_chase(c, c.bt * 4.0, duty=0.58, tail=1, boost=0.75)


def sc_beam_chase_diagonals(c: Ctx) -> list[Path]:
    """Two diagonal beams chasing across 7 positions at 8th notes with single-pass embedded beads."""
    n = 7
    head = int(math.floor(c.bt * 2.0)) % n
    phi = -0.85 if (int(c.bt // 8) % 2) else 0.85
    out: list[Path] = []
    for i, v, pal in (
        (head, 1.0, 0),
        ((head - 1) % n, 0.45, 0),
        ((head + 3) % n, 1.0, 1),
        ((head + 2) % n, 0.45, 1),
    ):
        x0 = -0.90 + 1.80 * i / (n - 1)
        seg = clipped_plane(x0, 0.0, phi + math.pi / 2.0)
        if seg is not None:
            col = c.palette(pal, v=v * 0.82 if v == 1.0 else v)
            if v == 1.0 and c.detail >= 0.35:
                out.append(beaded_line(
                    seg[0], seg[1],
                    line_rgb=col,
                    bead_rgb=WHITE,
                    count=c.count(3, 5),
                    bead_dwell=4,
                    phase=c.bt * 0.25,
                ))
            else:
                out.append(line(seg[0], seg[1], col))
    return out


_SCISSOR = (0.35, 0.95, 0.60, 1.30)


def sc_scissor_snap(c: Ctx) -> list[Path]:
    """Two long beams crossing at a drifting pivot with single-pass embedded white-hot beads."""
    sf = c.bt * 2.0
    step = int(math.floor(sf))
    a = _SCISSOR[(step - 1) % 4] + (_SCISSOR[step % 4] - _SCISSOR[(step - 1) % 4]) * _ease((sf - step) / 0.3)
    px = 0.30 * smooth_wave(c.bt, 8.0)
    out: list[Path] = []
    for i, sign in enumerate((1.0, -1.0)):
        seg = clipped_plane(px, 0.0, -sign * a)
        if seg is not None:
            col = c.palette(i, v=0.82)
            if c.detail >= 0.35:
                out.append(beaded_line(
                    seg[0], seg[1],
                    line_rgb=col,
                    bead_rgb=WHITE,
                    count=c.count(3, 5),
                    bead_dwell=4,
                    phase=c.bt * 0.22 * sign,
                ))
            else:
                out.append(line(seg[0], seg[1], col))
    if c.detail >= 0.6:
        out.append(line((-LIM, 0.0), (LIM, 0.0), c.palette(2, v=0.45)))
    return out


_S = 0.84
_CORNERS = [(-_S, -_S), (_S, -_S), (_S, _S), (-_S, _S)]


def _perim(s: float, scale: float = 1.0):
    s %= 4.0
    i = int(s) % 4
    t = min(max(s - i, 0.0), 1.0) if int(s) < 4 else 0.0
    a, b = _CORNERS[i], _CORNERS[(i + 1) % 4]
    return ((a[0] + (b[0] - a[0]) * t) * scale, (a[1] + (b[1] - a[1]) * t) * scale)


def _comet(head: float, length: float, scale: float, col: np.ndarray, flip: bool) -> Path:
    s0 = head - length
    ss = set(np.linspace(s0, head, 12).tolist())
    k = math.floor(s0) + 1
    while k < head:
        ss.add(float(k))
        k += 1
    pts = np.array([_perim(s, scale) for s in sorted(ss)])
    if flip:
        pts[:, 0] *= -1.0
    fade = np.linspace(0.10, 1.0, len(pts))[:, None]
    return Path(pts, np.clip(col[None, :] * fade, 0.0, 1.0), dwell=2)


def sc_perimeter_comet(c: Ctx) -> list[Path]:
    """Comets racing around the frame (one lap per 2 beats), plus a counter-rotating inner pair."""
    pump = beat_pump(c)
    h = ((c.bt / 2.0) % 1.0) * 4.0
    out = [
        _comet(h, 1.1, 1.0, c.palette(0, white_boost=0.5 * pump), False),
        _comet(h + 2.0, 1.1, 1.0, c.palette(1, white_boost=0.5 * pump), False),
    ]
    if c.detail >= 0.5:
        out.append(_comet(h + 1.0, 0.9, 0.5, c.palette(2, v=0.9), True))
        out.append(_comet(h + 3.0, 0.9, 0.5, c.palette(3, v=0.9), True))
    return out


def _chevrons(c: Ctx, speed: float) -> list[Path]:
    pump = beat_pump(c)
    n = c.count(3, 5)
    spacing = 1.8 / n
    ph = (c.bt * speed) % 1.0
    out: list[Path] = []
    for k in range(n + 1):
        x = -0.9 + (k + ph) * spacing
        if x > LIM:
            continue
        v = (0.35 + 0.65 * (x + 0.9) / 1.8) * min(1.0, (x + 0.9) / 0.3)
        h, d = 0.55, 0.18
        pts = np.array([[x - d, h], [x, 0.0], [x - d, -h]])
        pts[:, 0] = _clip(pts[:, 0])
        out.append(Path(pts, c.palette(k, v=max(v, 0.0), white_boost=0.5 * pump if k == n else 0.0), dwell=2))
    return out


def sc_chevron_run(c: Ctx) -> list[Path]:
    """Chevrons >>> marching across on the beat."""
    return _chevrons(c, 0.5)


def sc_drop_chevron_run(c: Ctx) -> list[Path]:
    """Double-speed chevrons, gated into 8th-note flashes."""
    return _chevrons(c, 1.0) if strobe_gate(c, div=2.0, duty=0.62) else []


def sc_eq_bars(c: Ctx) -> list[Path]:
    """Seven vertical bars reacting to bass, mid, and high frequency energy."""
    pump = beat_pump(c)
    bass, mid, high = c.f.bass, c.f.mid, c.f.high
    out: list[Path] = []
    for i in range(7):
        d = abs(i - 3) / 3.0
        band = min(max((1.0 - d) * bass + d * (0.6 * mid + 0.4 * high), 0.0), 1.0)
        h = 0.14 + 0.74 * min(1.0, band * (0.8 + 0.4 * pump))
        x = -0.78 + 1.56 * i / 6.0
        out.append(line((x, -h), (x, h), c.palette(i, v=0.7 + 0.3 * band, white_boost=0.5 * pump if i == 3 else 0.0)))
    return out


# ======================================================================== MAIN, DROP & BUILD SCENES

def sc_beaded_horizon_planes(c: Ctx) -> list[Path]:
    """Sweeping horizontal planes with 4–6 single-pass embedded white-hot beam beads."""
    u = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)
    y_center = -0.64 + 1.28 * u
    slant = 0.18 * smooth_wave(c.bt, 6.0)
    beads = c.count(4, 6)
    out: list[Path] = []
    for idx, off in enumerate((-0.24, 0.24)):
        y0 = float(np.clip(y_center + off - slant, -LIM, LIM))
        y1 = float(np.clip(y_center + off + slant, -LIM, LIM))
        out.append(beaded_line(
            (-LIM, y0), (LIM, y1),
            line_rgb=c.palette(idx, v=0.76),
            bead_rgb=WHITE,
            count=beads,
            bead_dwell=4,
            phase=c.bt * (0.28 if idx == 0 else -0.28),
        ))
    return out


def sc_laser_crown_vortex(c: Ctx) -> list[Path]:
    """Counter-rotating nested laser star/crown polygons pulsing on the beat."""
    pump = beat_pump(c)
    r_out = 0.62 + 0.24 * pump
    r_in = 0.28 + 0.14 * pump
    cx = 0.14 * smooth_wave(c.bt, 8.0)
    cy = 0.10 * smooth_wave(c.bt, 12.0)
    pts_n = c.count(5, 7)
    out = [
        star_polygon_path(cx, cy, r_out, r_in, pts_n, c.bt * 0.55, c.palette(0, v=1.0, white_boost=0.50 * pump)),
    ]
    if c.detail >= 0.35:
        out.append(star_polygon_path(cx, cy, r_out * 0.56, r_in * 0.52, pts_n, -c.bt * 0.82, c.palette(1, v=0.95)))
    return out


def sc_infinity_lissajous(c: Ctx) -> list[Path]:
    """Smooth 3D figure-8 / Lissajous aerial ribbon loop with harmonic color gradient."""
    pump = beat_pump(c)
    t = np.linspace(0.0, TAU, 36)
    ph = c.bt * TAU / 6.0
    rx = 0.76 + 0.14 * pump
    ry = 0.58 + 0.18 * pump
    xs = np.clip(rx * np.sin(t + ph), -LIM, LIM)
    ys = np.clip(ry * np.sin(2.0 * t + ph * 0.5), -LIM, LIM)
    rgb_grad = np.ones((len(t), 3), dtype=float) * (0.85 + 0.15 * pump)
    return [Path(np.stack([xs, ys], axis=1), rgb_grad, dwell=0)]


def sc_diamond_tunnel(c: Ctx) -> list[Path]:
    """Concentric rotating 4-point diamond/star tunnel rings zooming toward the viewer."""
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    rings = c.count(2, 3)
    cx = 0.18 * smooth_wave(c.bt, 8.0)
    cy = 0.14 * smooth_wave(c.bt, 12.0)
    out: list[Path] = []
    for r_idx in range(rings):
        phase = (frac + r_idx / rings) % 1.0
        rad = 0.20 + 0.72 * phase
        rot = (math.pi / 4.0) + 0.35 * smooth_wave(c.bt, 8.0) + r_idx * 0.38
        col = c.palette(r_idx, v=1.0, white_boost=0.55 * pump if r_idx == 0 else 0.0)
        out.append(polygon_path(cx, cy, rad, rad * 0.90, 4, rot, col))
    return out


def sc_prism_matrix_grid(c: Ctx) -> list[Path]:
    """3x3 or 4x3 breathing aerial hot-beam dot grid with synchronized wave tilt."""
    pump = beat_pump(c)
    cols = c.count(3, 4)
    rows = 3
    spread_x = 1.38 + 0.24 * pump
    spread_y = 1.18 + 0.22 * pump
    tilt = 0.16 * smooth_wave(c.bt, 6.0)
    out: list[Path] = []
    for r in range(rows):
        v_u = (r / max(rows - 1, 1)) - 0.5
        for col_i in range(cols):
            h_u = (col_i / max(cols - 1, 1)) - 0.5
            x = h_u * spread_x + v_u * tilt
            y = v_u * spread_y - h_u * tilt
            col = WHITE if (r + col_i) % 2 == 0 else c.palette(r * cols + col_i, white_boost=0.65)
            out.append(hot_dot((x, y), col, dwell=5))
    return out


def sc_kinetic_helix_wave(c: Ctx) -> list[Path]:
    """Dual intertwined DNA-style sinusoidal laser ribbons rotating in phase."""
    pump = beat_pump(c)
    x = np.linspace(-LIM, LIM, 28)
    ph = c.bt * TAU / 4.0
    amp = 0.38 + 0.22 * pump
    y1 = np.clip(amp * np.sin(2.0 * math.pi * x + ph), -LIM, LIM)
    y2 = np.clip(amp * np.sin(2.0 * math.pi * x + ph + math.pi), -LIM, LIM)
    return [
        Path(np.stack([x, y1], axis=1), c.palette(0, v=1.0, white_boost=0.45 * pump), dwell=0),
        Path(np.stack([x, y2], axis=1), c.palette(1, v=1.0, white_boost=0.45 * pump), dwell=0),
    ]


def sc_vortex_star_polygon(c: Ctx) -> list[Path]:
    """Rotating hexagram (6-point star made of two counter-spinning triangles)."""
    pump = beat_pump(c)
    rad = 0.62 + 0.24 * pump
    rot = c.bt * 0.48
    cx = 0.14 * math.sin(c.bt * TAU / 12.0)
    cy = 0.12 * math.cos(c.bt * TAU / 12.0)
    return [
        polygon_path(cx, cy, rad, rad * 0.92, 3, rot, c.palette(0, v=1.0, white_boost=0.50 * pump)),
        polygon_path(cx, cy, rad, rad * 0.92, 3, -rot + math.pi / 3.0, c.palette(1, v=1.0, white_boost=0.50 * pump)),
    ]


def sc_laser_cage_box(c: Ctx) -> list[Path]:
    """3D perspective wireframe tunnel box with shifting vanishing point."""
    pump = beat_pump(c)
    outer_s = 0.82
    inner_s = 0.34 + 0.12 * pump
    vx = 0.26 * smooth_wave(c.bt, 8.0)
    vy = 0.20 * smooth_wave(c.bt, 12.0)
    outer = np.array([[-outer_s, -outer_s], [outer_s, -outer_s], [outer_s, outer_s], [-outer_s, outer_s], [-outer_s, -outer_s]], float)
    inner = np.array([[vx - inner_s, vy - inner_s], [vx + inner_s, vy - inner_s], [vx + inner_s, vy + inner_s], [vx - inner_s, vy + inner_s], [vx - inner_s, vy - inner_s]], float)
    out: list[Path] = [
        Path(outer, c.palette(0, v=1.0, white_boost=0.45 * pump), dwell=0),
        Path(inner, c.palette(1, v=1.0, white_boost=0.45 * pump), dwell=0),
    ]
    if c.detail >= 0.45:
        for corner_idx in range(4):
            out.append(line(outer[corner_idx], inner[corner_idx], c.palette(2 + (corner_idx % 2), v=0.90), dwell=0))
    return out


def sc_shockwave_octagon(c: Ctx) -> list[Path]:
    """Beat-triggered expanding octagonal rings with counter-rotating geometry."""
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    rings = c.count(2, 3)
    out: list[Path] = []
    for i in range(rings):
        ph = (frac + i / rings) % 1.0
        r = 0.22 + 0.70 * math.sin(ph * math.pi * 0.5)
        rot = (math.pi / 8.0) + (0.35 if i % 2 == 0 else -0.35) * c.bt
        out.append(polygon_path(0.0, 0.0, r, r * 0.90, 8, rot, c.palette(i, v=1.0, white_boost=0.55 * pump if i == 0 else 0.0)))
    return out


def sc_concentric_tunnel(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    rings = c.count(2, 3)
    cx = 0.18 * smooth_wave(c.bt, 8.0)
    cy = 0.12 * smooth_wave(c.bt, 12.0)
    rot = c.bt * 0.25
    out: list[Path] = []
    for r_idx in range(rings):
        phase = (frac + r_idx / rings) % 1.0
        rad = 0.22 + 0.68 * phase
        col = c.palette(r_idx, v=1.0, white_boost=0.55 * pump if r_idx == 0 else 0.0)
        out.append(polygon_path(cx, cy, rad, rad * 0.88, 16, rot + r_idx * 0.2, col))
    return out


def sc_rotating_laser_cone(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    t = np.linspace(0.0, TAU, 32)
    tilt_x = 0.32 * smooth_wave(c.bt, 8.0)
    tilt_y = 0.26 * smooth_wave(c.bt, 12.0)
    rx = 0.56 + 0.24 * pump
    ry = 0.48 + 0.20 * pump
    cx = 0.16 * math.sin(c.bt * TAU / 16.0)
    cy = 0.12 * math.cos(c.bt * TAU / 16.0)
    xs = np.clip(cx + rx * np.cos(t) + tilt_x * np.sin(2.0 * t + c.bt), -LIM, LIM)
    ys = np.clip(cy + ry * np.sin(t) + tilt_y * np.cos(2.0 * t - c.bt), -LIM, LIM)
    col = c.palette(0, v=1.0, white_boost=0.45 * pump)
    out = [Path(np.stack([xs, ys], axis=1), col, dwell=0)]
    if c.detail >= 0.45:
        xs2 = np.clip(cx + (rx * 0.58) * np.cos(t + 0.5), -LIM, LIM)
        ys2 = np.clip(cy + (ry * 0.58) * np.sin(t + 0.5), -LIM, LIM)
        out.append(Path(np.stack([xs2, ys2], axis=1), c.palette(1, v=0.95), dwell=0))
    return out


def sc_split_direction_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u_top = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    x_top = -0.82 + 1.64 * u_top
    x_bot = 0.82 - 1.64 * u_top
    blades = c.count(2, 2)
    gap = 0.18 + 0.05 * pump
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        xt = float(np.clip(x_top + off, -LIM, LIM))
        xb = float(np.clip(x_bot + off, -LIM, LIM))
        out.append(line((xt, 0.10), (xt, LIM), c.palette(i, white_boost=0.55 * pump if i == 0 else 0.0), dwell=0))
        out.append(line((xb, -LIM), (xb, -0.10), c.palette(i + 2, white_boost=0.55 * pump if i == 0 else 0.0), dwell=0))
    return out


def sc_half_plane_stagger(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u1 = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    u2 = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)
    y_left = -0.82 + 1.64 * u1
    y_right = 0.82 - 1.64 * u2
    x_top = -0.84 + 1.68 * u2
    x_bot = 0.84 - 1.68 * u1
    return [
        line((-LIM, y_left), (-0.10, y_left + 0.08), c.palette(0, white_boost=0.5 * pump), dwell=0),
        line((LIM, y_right), (0.10, y_right - 0.08), c.palette(1, white_boost=0.5 * pump), dwell=0),
        line((x_top, 0.10), (x_top, LIM), c.palette(2, white_boost=0.4 * pump), dwell=0),
        line((x_bot, -LIM), (x_bot, -0.10), c.palette(3, white_boost=0.4 * pump), dwell=0),
    ]


def sc_double_time_whip(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    x_center = -0.80 + 1.60 * u
    slant = 0.20 * math.sin(c.bt * TAU)
    blades = c.count(2, 2)
    gap = 0.20 + 0.05 * pump
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        x0 = float(np.clip(x_center + off - slant, -LIM, LIM))
        x1 = float(np.clip(x_center + off + slant, -LIM, LIM))
        col = c.palette(i, v=1.0, white_boost=0.65 * pump if i == 0 else 0.0)
        out.append(line((x0, -LIM), (x1, LIM), col, dwell=0))
    return out


def sc_half_time_glide(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)
    y_center = -0.74 + 1.48 * u
    slant = 0.18 * smooth_wave(c.bt, 8.0)
    blades = c.count(2, 3)
    gap = 0.20 + 0.06 * pump
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        y0 = float(np.clip(y_center + off - slant, -LIM, LIM))
        y1 = float(np.clip(y_center + off + slant, -LIM, LIM))
        col = c.palette(i, v=1.0, white_boost=0.55 * pump if i == 1 else 0.0)
        out.append(line((-LIM, y0), (LIM, y1), col, dwell=0))
    return out


def sc_dot_matrix_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_center = -0.72 + 1.44 * u
    rows = c.count(2, 3)
    dots_per_row = c.count(5, 7)
    slant = 0.14 * smooth_wave(c.bt, 4.0)
    out: list[Path] = []
    for r in range(rows):
        off = (r - (rows - 1) / 2.0) * (0.24 + 0.08 * pump)
        y0 = float(np.clip(y_center + off - slant, -LIM, LIM))
        y1 = float(np.clip(y_center + off + slant, -LIM, LIM))
        col = c.palette(r, v=1.0, white_boost=0.5 * pump if r == 0 else 0.0)
        out.extend(dotted_line((-LIM, y0), (LIM, y1), col, dots=dots_per_row, dot_len=0.038, phase=c.bt * 1.5 + r * 0.35))
    return out


def sc_drop_hyper_strobe(c: Ctx) -> list[Path]:
    div = 4.0 if (int(c.bt / 2) % 2 == 1) else 2.0
    if not strobe_gate(c, div=div, duty=0.48):
        return []
    pump = beat_pump(c)
    step = int(c.bt * div)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    if step % 3 == 0:
        x = -0.80 + 1.60 * u
        x2 = float(np.clip(x - 0.24 if x > -0.56 else x + 0.24, -LIM, LIM))
        return [
            line((x, -LIM), (x, LIM), c.palette(0, white_boost=0.75), dwell=0),
            line((x2, LIM), (x2, -LIM), c.palette(1), dwell=0),
        ]
    if step % 3 == 1:
        y_top = 0.18 + 0.72 * u
        y_bot = -0.18 - 0.72 * u
        return [
            line((-LIM, y_top), (LIM, y_top), c.palette(1, white_boost=0.65 * pump), dwell=0),
            line((LIM, y_bot), (-LIM, y_bot), c.palette(2, white_boost=0.65), dwell=0),
        ]
    r = 0.34 + 0.54 * u
    return [star_polygon_path(0.0, 0.0, r, r * 0.45, 6, c.bt * 1.2, c.palette(0, white_boost=0.55))]


def sc_full_width_wall(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    x_center = -0.76 + 1.52 * u
    blades = c.count(2, 2)
    thickness = 0.18 + 0.05 * pump
    tilt = 0.06 * math.sin(c.bt * math.pi)
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * thickness
        x0 = float(np.clip(x_center + off - tilt, -LIM, LIM))
        x1 = float(np.clip(x_center + off + tilt, -LIM, LIM))
        col = c.palette(i, v=1.0, white_boost=0.60 * pump if i == 0 else 0.0)
        out.append(line((x0, -LIM), (x1, LIM), col, dwell=0))
    return out


def sc_full_height_sheet(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_center = -0.76 + 1.52 * u
    slant = 0.14 * math.sin(c.bt * math.pi)
    blades = c.count(2, 2)
    gap = 0.20 + 0.05 * pump
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        y0 = float(np.clip(y_center + off - slant, -LIM, LIM))
        y1 = float(np.clip(y_center + off + slant, -LIM, LIM))
        col = c.palette(i, v=1.0, white_boost=0.60 * pump if i == 0 else 0.0)
        out.append(line((-LIM, y0), (LIM, y1), col, dwell=0))
    return out


def sc_orbital_plane_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    angle = c.bt * (TAU / 8.0)
    dist = 0.24 * smooth_wave(c.bt, 4.0)
    cx, cy = dist * math.cos(angle), dist * math.sin(angle)
    blades = c.count(2, 2)
    spacing = 0.22 + 0.06 * pump
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * spacing
        px = cx + off * math.cos(angle)
        py = cy + off * math.sin(angle)
        seg = clipped_plane(px, py, angle, span=LIM)
        if seg is not None:
            col = c.palette(i, v=1.0, white_boost=0.65 * pump if i == 0 else 0.0)
            out.append(line(seg[0], seg[1], col, dwell=0))
    return out


def sc_multi_plane_cross_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    n_planes = c.count(2, 2)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    v = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)
    out: list[Path] = []
    for i in range(n_planes):
        x = -0.86 + 1.72 * ((u + i / max(n_planes, 1)) % 1.0)
        out.append(line((x, -LIM), (x, LIM), c.palette(i, v=1.0, white_boost=0.5 * pump if i == 0 else 0.0), dwell=0))
    for j in range(n_planes):
        y = -0.86 + 1.72 * ((1.0 - v + j / max(n_planes, 1)) % 1.0)
        out.append(line((-LIM, y), (LIM, y), c.palette(j + 2, v=1.0, white_boost=0.5 * pump if j == 0 else 0.0), dwell=0))
    return out


def sc_drop_x_blades(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    planes = c.count(2, 3)
    tilt = 0.50 * smooth_wave(c.bt, 4.0)
    y_sweep = 0.24 * smooth_wave(c.bt, 8.0)
    out: list[Path] = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.24 + 0.12 * pump)
        y0 = float(np.clip(y_sweep + gap - tilt, -LIM, LIM))
        y1 = float(np.clip(y_sweep + gap + tilt, -LIM, LIM))
        col = c.palette(p, v=1.0, white_boost=0.65 * pump if p == 0 else 0.0)
        out.append(line((-LIM, y0), (LIM, y1), col, dwell=0))
    return out


def sc_quadrant_mirror_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    sweep_ang = 0.22 + 0.96 * u
    p0, p1 = ray(-LIM, -LIM, sweep_ang, 1.9 + 0.3 * pump)
    p2, p3 = ray(-LIM, LIM, math.pi - sweep_ang, 1.9 + 0.3 * pump)
    return [
        line(p0, p1, c.palette(0, white_boost=0.55 * pump), dwell=0),
        line(p3, p2, c.palette(1, white_boost=0.55 * pump), dwell=0),
    ]


def sc_pendulum_beam_bar(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    ang = 0.80 * smooth_wave(c.bt, 4.0)
    bars = c.count(2, 2)
    spacing = 0.24 + 0.06 * pump
    out: list[Path] = []
    for i in range(bars):
        off = (i - (bars - 1) / 2.0) * spacing
        seg = clipped_plane(-0.20 + off * math.cos(ang), off * math.sin(ang), ang, span=LIM)
        if seg is not None:
            col = c.palette(i, white_boost=0.60 * pump if i == 0 else 0.0)
            out.append(line(seg[0], seg[1], col, dwell=0))
    return out


def sc_crisscross_ribbon_planes(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_a = -0.72 + 1.44 * u
    y_b = 0.72 - 1.44 * u
    return [
        line((-LIM, y_a), (LIM, float(np.clip(y_a + 0.38, -LIM, LIM))), c.palette(0, white_boost=0.5 * pump), dwell=0),
        line((LIM, y_b), (-LIM, float(np.clip(y_b + 0.38, -LIM, LIM))), c.palette(1, white_boost=0.5 * pump), dwell=0),
    ]


def sc_alternating_horizon_gates(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    gap = 0.20 + 0.68 * (0.5 - 0.5 * math.cos(math.pi * c.bt))
    out = [
        line((-LIM, gap), (LIM, gap), c.palette(0, white_boost=0.60 * pump), dwell=0),
        line((LIM, -gap), (-LIM, -gap), c.palette(1, white_boost=0.60 * pump), dwell=0),
    ]
    if c.detail >= 0.45:
        out.extend(dotted_line((-0.88, 0.0), (0.88, 0.0), c.palette(2, v=0.9, white_boost=0.35 * pump), dots=4, dot_len=0.035, phase=c.bt * 2.0))
    return out


def sc_hyper_bar_ricochet(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u1 = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    u2 = beat_sweep(c.bt + 0.25, beats_per_pass=0.5, alternate=True)
    x1 = -0.86 + 1.72 * u1
    x2 = 0.86 - 1.72 * u2
    if abs(x1 - x2) < 0.22:
        x2 = float(np.clip(x1 + (0.24 if x1 < 0 else -0.24), -LIM, LIM))
    y1 = -0.86 + 1.72 * u2
    return [
        line((x1, -LIM), (x1, LIM), c.palette(0, white_boost=0.65 * pump), dwell=0),
        line((x2, LIM), (x2, -LIM), c.palette(1, white_boost=0.65 * pump), dwell=0),
        line((-LIM, y1), (LIM, y1), c.palette(2, white_boost=0.45 * pump), dwell=0),
    ]


def sc_horizontal_plane_blast(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    eased = math.sin(frac * math.pi * 0.5)
    pairs = c.count(2, 2)
    out: list[Path] = []
    for p in range(pairs):
        y = 0.18 + 0.70 * eased * ((p + 1) / pairs)
        out.append(line((-LIM, y), (LIM, y), c.palette(p * 2, white_boost=0.65 * pump if p == 0 else 0.0), dwell=0))
        out.append(line((LIM, -y), (-LIM, -y), c.palette(p * 2 + 1, white_boost=0.65 * pump if p == 0 else 0.0), dwell=0))
    return out


def sc_hot_beam_matrix(c: Ctx) -> list[Path]:
    cols = c.count(4, 5)
    rows = 2
    sway_x = 0.18 * smooth_wave(c.bt, 4.0)
    sway_y = 0.16 * smooth_wave(c.bt, 8.0)
    out: list[Path] = []
    for r in range(rows):
        y = (-0.34 if r == 0 else 0.34) + sway_y * (1.0 if r == 0 else -1.0)
        for col_i in range(cols):
            u = (col_i / max(cols - 1, 1)) - 0.5
            x = u * 1.48 + sway_x
            out.append(hot_dot((x, y), WHITE if col_i % 2 == 0 else c.palette(r * 2 + col_i, white_boost=0.75), dwell=5))
    return out


def sc_quad_corner_lasers(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    blades = c.count(2, 3)
    out: list[Path] = []
    for i in range(blades):
        ang = 0.24 + 0.80 * u + (i - 1) * 0.20
        p0, p1 = ray(-0.90, -0.90, ang, 2.5)
        out.append(line(p0, p1, c.palette(i, white_boost=0.65 * pump), dwell=0))
    return out


def sc_strobe_tunnel_ribs(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    layers = c.count(2, 3)
    cx = -0.18 * smooth_wave(c.bt, 8.0)
    out: list[Path] = []
    for l in range(layers):
        s = 0.26 + 0.64 * ((frac + l / layers) % 1.0)
        x0 = max(cx - s, -LIM)
        x1 = min(cx + s, LIM)
        y0, y1 = -s * 0.85, s * 0.85
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)
        out.append(Path(pts, c.palette(l, white_boost=0.55 * pump), dwell=0))
    return out


# ======================================================================== BUILD SECTION SCENES

def sc_build_lane_accel(c: Ctx) -> list[Path]:
    """Lane chase that accelerates through the build (driven by rise-time), tail growing and heads whitening."""
    b = min(max(c.f.build, 0.0), 1.0)
    return _lane_chase(c, c.rt * 2.0, tail=1 + int(2 * b), boost=0.85 * b * b)


def sc_build_blackout_chase(c: Ctx) -> list[Path]:
    """Escalating wide-coverage blackout chase accelerating from 8ths to 16ths during Builds."""
    b = c.f.build
    rate = 2.0 if b < 0.5 else 4.0
    step_f = c.rt * rate
    step = int(math.floor(step_f))
    if (step_f - step) > 0.60:
        return []
    zone = step % 4
    col = c.palette(zone, v=0.85 + 0.15 * b, white_boost=0.75 * (b ** 2))
    if zone == 0:
        return [line((-LIM, 0.78), (LIM, 0.78), col)]
    if zone == 1:
        return [line((-LIM, -0.78), (LIM, -0.78), col)]
    if zone == 2:
        return [line((-0.80, -LIM), (-0.80, LIM), col)]
    return [line((0.80, -LIM), (0.80, LIM), col)]


def sc_build_helix_tighten(c: Ctx) -> list[Path]:
    """Dual accelerating helix ribbons that compress toward center as build approaches 1.0."""
    b = c.f.build
    x = np.linspace(-LIM, LIM, 28)
    ph = c.rt * 1.9
    amp = 0.56 * (1.0 - 0.78 * b) + 0.06
    y1 = np.clip(amp * np.sin(2.2 * math.pi * x + ph), -LIM, LIM)
    y2 = np.clip(amp * np.sin(2.2 * math.pi * x + ph + math.pi), -LIM, LIM)
    return [
        Path(np.stack([x, y1], axis=1), c.palette(0, v=0.82 + 0.18 * b, white_boost=0.70 * (b ** 2)), dwell=0),
        Path(np.stack([x, y2], axis=1), c.palette(1, v=0.82 + 0.18 * b, white_boost=0.70 * (b ** 2)), dwell=0),
    ]


def sc_build_tunnel_collapse(c: Ctx) -> list[Path]:
    """Spinning concentric polygon tunnel tightening inward into a bright focal point."""
    b = c.f.build
    rings = c.count(2, 3)
    out: list[Path] = []
    for i in range(rings):
        r = max(0.08, (0.86 - i * 0.22) * (1.0 - 0.78 * b) + 0.08)
        rot = c.rt * (0.9 + 1.6 * b) * (1.0 if i % 2 == 0 else -1.0)
        out.append(polygon_path(0.0, 0.0, r, r * 0.88, 6, rot, c.palette(i, v=0.82 + 0.18 * b, white_boost=0.75 * (b ** 2))))
    return out


def sc_build_rising_sheet(c: Ctx) -> list[Path]:
    b = c.f.build
    planes = c.count(2, 2)
    y_rise = -0.55 + 1.35 * b
    tilt = (0.28 * (1.0 - 0.5 * b)) * math.sin(c.rt * TAU / 4.0)
    out: list[Path] = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.28 * (1.0 - 0.4 * b) + 0.14)
        y0 = float(np.clip(y_rise + gap - tilt, -LIM, LIM))
        y1 = float(np.clip(y_rise + gap + tilt, -LIM, LIM))
        col = c.palette(p, v=0.82 + 0.18 * b, white_boost=0.65 * (b ** 2))
        out.append(line((-LIM, y0), (LIM, y1), col, dwell=0))
    return out


def sc_build_prism_tighten(c: Ctx) -> list[Path]:
    b = c.f.build
    u = beat_sweep(c.rt, beats_per_pass=1.0, alternate=True)
    x_center = -0.82 + 1.64 * u
    blades = c.count(2, 3)
    width = 0.46 * (1.0 - 0.55 * b) + 0.16
    out: list[Path] = []
    for i in range(blades):
        off = (i / max(blades - 1, 1) - 0.5) * width
        x = float(np.clip(x_center + off, -LIM, LIM))
        col = c.palette(i, v=0.80 + 0.20 * b, white_boost=0.70 * (b ** 2))
        out.append(line((x, -LIM), (x, LIM), col, dwell=0))
    return out


def sc_build_accelerating_bars(c: Ctx) -> list[Path]:
    b = c.f.build
    u = beat_sweep(c.rt, beats_per_pass=0.5, alternate=True)
    span = 0.86 * (1.0 - 0.65 * b) + 0.14
    x = -span + 2.0 * span * u
    return [
        line((x, -LIM), (x, LIM), c.palette(0, v=0.82 + 0.18 * b, white_boost=0.7 * (b ** 2)), dwell=0),
        line((-x, LIM), (-x, -LIM), c.palette(1, v=0.82 + 0.18 * b, white_boost=0.7 * (b ** 2)), dwell=0),
    ]


def sc_build_dot_convergence(c: Ctx) -> list[Path]:
    b = c.f.build
    n_dots = c.count(8, 12)
    r = 0.86 * (1.0 - 0.74 * b) + 0.12
    ox = -0.28 * (1.0 - b)
    out: list[Path] = []
    for i in range(n_dots):
        ang = c.rt * 2.2 + (i / n_dots) * TAU
        out.append(hot_dot((ox + r * math.cos(ang), r * math.sin(ang)), WHITE, dwell=5))
    return out


def sc_build_split_pinch(c: Ctx) -> list[Path]:
    b = c.f.build
    y_gap = 0.84 * (1.0 - 0.80 * b) + 0.10
    wobble = 0.14 * (1.0 - b) * math.sin(c.rt * TAU / 2.0)
    return [
        line((-LIM, y_gap - wobble), (LIM, y_gap + wobble), c.palette(0, v=0.82 + 0.18 * b, white_boost=0.7 * (b ** 2)), dwell=0),
        line((LIM, -y_gap + wobble), (-LIM, -y_gap - wobble), c.palette(1, v=0.82 + 0.18 * b, white_boost=0.7 * (b ** 2)), dwell=0),
    ]


# ======================================================================== SECONDARY MODIFIERS

def _apply_modifier(paths: list[Path], mod: str, c: Ctx) -> list[Path]:
    if not mod or not paths:
        return paths

    if mod == "plane_dwell_beads":
        # Upgrade existing straight beam lines in-place into single-pass beaded paths
        n_beads = c.count(3, 5)
        out: list[Path] = []
        beaded_lines = 0
        for idx, p in enumerate(paths):
            if (
                beaded_lines < 4
                and p.dwell < 3
                and len(p.pts) == 2
                and np.asarray(p.rgb).ndim == 1
            ):
                seg_len = float(np.hypot(*(p.pts[1] - p.pts[0])))
                if seg_len >= 0.45:
                    out.append(beaded_line(
                        p.pts[0],
                        p.pts[1],
                        line_rgb=p.rgb,
                        bead_rgb=WHITE,
                        count=n_beads,
                        bead_dwell=4,
                        phase=c.bt * 0.22 * (1.0 if idx % 2 == 0 else -1.0),
                    ))
                    beaded_lines += 1
                    continue
            out.append(p)
        return out

    if mod == "blackout_zone_chopper":
        step_f = c.bt * 4.0
        step = int(math.floor(step_f))
        if (step_f - step) > 0.58:
            return []
        zone = step % 4
        out = []
        for p in paths:
            pts = p.pts.copy()
            if zone == 0:
                pts[:, 1] = np.clip(np.abs(pts[:, 1]) * 0.95 + 0.06, -LIM, LIM)
            elif zone == 1:
                pts[:, 1] = np.clip(-np.abs(pts[:, 1]) * 0.95 - 0.06, -LIM, LIM)
            elif zone == 2:
                pts[:, 0] = np.clip(-np.abs(pts[:, 0]) * 0.95 - 0.06, -LIM, LIM)
            else:
                pts[:, 0] = np.clip(np.abs(pts[:, 0]) * 0.95 + 0.06, -LIM, LIM)
            out.append(Path(pts, np.clip(p.rgb * 0.72 + WHITE * 0.28, 0.0, 1.0), p.dwell))
        return out

    if mod == "fast_strobe_16th":
        div = 4.0 if (int(c.bt) % 2 == 1) else 2.0
        if not strobe_gate(c, div=div, duty=0.5):
            return []
        return [Path(p.pts, np.clip(p.rgb * 0.65 + WHITE * 0.35, 0.0, 1.0), p.dwell) for p in paths]

    if mod == "beat_rotate":
        ang = 0.30 * smooth_wave(c.bt, 4.0)
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for p in paths:
            x, y = p.pts[:, 0], p.pts[:, 1]
            nx = np.clip(x * ca - y * sa, -LIM, LIM)
            ny = np.clip(x * sa + y * ca, -LIM, LIM)
            out.append(Path(np.stack([nx, ny], axis=1), p.rgb, p.dwell))
        return out

    if mod == "stereo_pan_shift":
        dx = 0.18 * smooth_wave(c.bt, 8.0)
        out = []
        for p in paths:
            pts = p.pts.copy()
            pts[:, 0] = np.clip(pts[:, 0] + dx, -LIM, LIM)
            out.append(Path(pts, p.rgb, p.dwell))
        return out

    if mod == "kick_zoom_pump":
        scale = 0.86 + 0.14 * beat_pump(c)
        return [Path(np.clip(p.pts * scale, -LIM, LIM), p.rgb, p.dwell) for p in paths]

    if mod == "counter_sweep_plane":
        u = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)
        y = -0.84 + 1.68 * u
        return paths + [line((-LIM, y), (LIM, y), c.palette(2, v=0.95, white_boost=0.4 * beat_pump(c)), dwell=0)]

    if mod == "hat_packet_chop":
        if c.hi < 0.38:
            return paths
        out = []
        for idx, p in enumerate(paths):
            if p.dwell < 3 and len(p.pts) == 2 and float(np.hypot(*(p.pts[1] - p.pts[0]))) > 0.05:
                out.extend(dashed_beam(p.pts[0], p.pts[1], p.rgb, segments=3, duty=0.65, phase=c.bt * 2.0 + idx * 0.25))
            else:
                out.append(p)
        return out

    return paths


# ======================================================================== SMOOTH RGB-SAFE PRESET MORPHING

def _resample_poly(pts: np.ndarray, k: int) -> np.ndarray:
    if len(pts) == k:
        return pts.copy()
    if len(pts) < 2:
        return np.repeat(pts[:1], k, axis=0)
    d = np.hypot(*np.diff(pts, axis=0).T)
    total = float(d.sum())
    if total < 1e-6:
        return np.repeat(pts[:1], k, axis=0)
    s = np.concatenate([[0.0], np.cumsum(d) / total])
    u = np.linspace(0.0, 1.0, k)
    return np.stack([np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], axis=1)


def _resample_rgb(rgb: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(rgb, dtype=float)
    if arr.ndim == 1:
        return np.broadcast_to(arr, (k, 3)).copy()
    if len(arr) == k:
        return arr.copy()
    if len(arr) < 2:
        return np.repeat(arr[:1], k, axis=0)
    u_src = np.linspace(0.0, 1.0, len(arr))
    u_dst = np.linspace(0.0, 1.0, k)
    return np.stack([np.interp(u_dst, u_src, arr[:, ch]) for ch in range(3)], axis=1)


def _morph_paths(prev_paths: list[Path], next_paths: list[Path], alpha: float) -> list[Path]:
    if not prev_paths:
        return next_paths
    if not next_paths:
        return prev_paths
    t = alpha * alpha * (3.0 - 2.0 * alpha)
    n_out = max(len(prev_paths), len(next_paths))
    morphed: list[Path] = []
    for i in range(n_out):
        p0 = prev_paths[i % len(prev_paths)]
        p1 = next_paths[i % len(next_paths)]
        v_count = max(len(p0.pts), len(p1.pts))
        pts = _resample_poly(p0.pts, v_count) * (1.0 - t) + _resample_poly(p1.pts, v_count) * t
        rgb = _resample_rgb(p0.rgb, v_count) * (1.0 - t) + _resample_rgb(p1.rgb, v_count) * t
        dwell = int(round(p0.dwell * (1.0 - t) + p1.dwell * t))
        morphed.append(Path(pts, rgb, dwell))
    return morphed


SCENES = {
    # Mirrored-Pair Beam Chases & Sweeping Bars
    "beat_bar_sweep": sc_beat_bar_sweep,
    "bar_chase_lanes": sc_bar_chase_lanes,
    "drop_lane_chase": sc_drop_lane_chase,
    "beam_chase_diagonals": sc_beam_chase_diagonals,
    "scissor_snap": sc_scissor_snap,
    "perimeter_comet": sc_perimeter_comet,
    "chevron_run": sc_chevron_run,
    "drop_chevron_run": sc_drop_chevron_run,
    "eq_bars": sc_eq_bars,
    # Single-Pass Beaded Plane Scenes
    "beaded_horizon_planes": sc_beaded_horizon_planes,
    "beaded_sky_canopy": sc_beaded_sky_canopy,
    # Rapid Wide-Coverage Blackout Chasers
    "blackout_zone_chase": sc_blackout_zone_chase,
    "quadrant_blackout_relay": sc_quadrant_blackout_relay,
    "shutter_grid_chase": sc_shutter_grid_chase,
    "pingpong_blackout_whip": sc_pingpong_blackout_whip,
    # Aerial Geometry
    "laser_crown_vortex": sc_laser_crown_vortex,
    "infinity_lissajous": sc_infinity_lissajous,
    "diamond_tunnel": sc_diamond_tunnel,
    "prism_matrix_grid": sc_prism_matrix_grid,
    "kinetic_helix_wave": sc_kinetic_helix_wave,
    "vortex_star_polygon": sc_vortex_star_polygon,
    "laser_cage_box": sc_laser_cage_box,
    "shockwave_octagon": sc_shockwave_octagon,
    "concentric_tunnel": sc_concentric_tunnel,
    "rotating_laser_cone": sc_rotating_laser_cone,
    "hot_beam_matrix": sc_hot_beam_matrix,
    "orbital_plane_sweep": sc_orbital_plane_sweep,
    # Laser Sky Suite
    "liquid_sky_drift": sc_liquid_sky_drift,
    "cathedral_sky_vault": sc_cathedral_sky_vault,
    "ribbon_tide_sky": sc_ribbon_tide_sky,
    "prism_horizon_deck": sc_prism_horizon_deck,
    "morphing_plane_sky": sc_morphing_plane_sky,
    "breathe_horizon_sky": sc_breathe_horizon_sky,
    "velvet_sky_sweep": sc_velvet_sky_sweep,
    "silk_wave_sky": sc_silk_wave_sky,
    "dual_horizon_mirage": sc_dual_horizon_mirage,
    "floating_dot_constellation": sc_floating_dot_constellation,
    "slow_v_canopy_sky": sc_slow_v_canopy_sky,
    "aurora_sky_curtain": sc_aurora_sky_curtain,
    # Kinetic Planes, Sweeps & Strobes
    "split_direction_sweep": sc_split_direction_sweep,
    "half_plane_stagger": sc_half_plane_stagger,
    "double_time_whip": sc_double_time_whip,
    "half_time_glide": sc_half_time_glide,
    "dot_matrix_sweep": sc_dot_matrix_sweep,
    "drop_hyper_strobe": sc_drop_hyper_strobe,
    "full_width_wall": sc_full_width_wall,
    "full_height_sheet": sc_full_height_sheet,
    "multi_plane_cross_sweep": sc_multi_plane_cross_sweep,
    "drop_x_blades": sc_drop_x_blades,
    "quadrant_mirror_sweep": sc_quadrant_mirror_sweep,
    "pendulum_beam_bar": sc_pendulum_beam_bar,
    "crisscross_ribbon_planes": sc_crisscross_ribbon_planes,
    "alternating_horizon_gates": sc_alternating_horizon_gates,
    "hyper_bar_ricochet": sc_hyper_bar_ricochet,
    "horizontal_plane_blast": sc_horizontal_plane_blast,
    "quad_corner_lasers": sc_quad_corner_lasers,
    "strobe_tunnel_ribs": sc_strobe_tunnel_ribs,
    # Build Scenes
    "build_lane_accel": sc_build_lane_accel,
    "build_blackout_chase": sc_build_blackout_chase,
    "build_helix_tighten": sc_build_helix_tighten,
    "build_tunnel_collapse": sc_build_tunnel_collapse,
    "build_rising_sheet": sc_build_rising_sheet,
    "build_prism_tighten": sc_build_prism_tighten,
    "build_accelerating_bars": sc_build_accelerating_bars,
    "build_dot_convergence": sc_build_dot_convergence,
    "build_split_pinch": sc_build_split_pinch,
    # Legacy aliases
    "tunnel": sc_concentric_tunnel,
    "tunnel_fast": sc_strobe_tunnel_ribs,
    "starburst": sc_laser_crown_vortex,
    "starburst_fast": sc_hot_beam_matrix,
    "lissajous": sc_infinity_lissajous,
    "scope": sc_kinetic_helix_wave,
    "spiral": sc_rotating_laser_cone,
    "spiral_slow": sc_velvet_sky_sweep,
    "grid": sc_prism_matrix_grid,
    "drop_blast": sc_horizontal_plane_blast,
    "breath": sc_breathe_horizon_sky,
    "riser": sc_build_tunnel_collapse,
    "ring_pulse": sc_shockwave_octagon,
    "sky_cross": sc_dual_horizon_mirage,
    "scan_sweep": sc_ribbon_tide_sky,
    "build_scan": sc_build_rising_sheet,
    "strobe_poly": sc_drop_hyper_strobe,
    "strobe_beams": sc_hyper_bar_ricochet,
    "strobe_burst": sc_hot_beam_matrix,
}

QUICK_CUES: list[tuple[str, str]] = [
    ("beaded_sky_canopy", "Beaded Sky"),
    ("beaded_horizon_planes", "Beaded Planes"),
    ("scissor_snap", "Scissor Snap"),
    ("blackout_zone_chase", "Blackout Chase"),
    ("beam_chase_diagonals", "Diag Chase"),
    ("perimeter_comet", "Comet Frame"),
    ("chevron_run", "Chevrons"),
    ("eq_bars", "EQ Bars"),
]

POOLS = {
    MAIN: [
        "beat_bar_sweep",
        "bar_chase_lanes",
        "beam_chase_diagonals",
        "scissor_snap",
        "perimeter_comet",
        "chevron_run",
        "eq_bars",
        "beaded_horizon_planes",
        "blackout_zone_chase",
        "quadrant_blackout_relay",
        "shutter_grid_chase",
        "pingpong_blackout_whip",
        "laser_crown_vortex",
        "infinity_lissajous",
        "diamond_tunnel",
        "prism_matrix_grid",
        "kinetic_helix_wave",
        "vortex_star_polygon",
        "laser_cage_box",
        "shockwave_octagon",
        "concentric_tunnel",
        "rotating_laser_cone",
        "split_direction_sweep",
        "half_plane_stagger",
        "dot_matrix_sweep",
        "half_time_glide",
        "full_width_wall",
        "full_height_sheet",
        "orbital_plane_sweep",
        "multi_plane_cross_sweep",
        "quadrant_mirror_sweep",
        "pendulum_beam_bar",
        "crisscross_ribbon_planes",
        "alternating_horizon_gates",
    ],
    DROP: [
        "drop_lane_chase",
        "drop_chevron_run",
        "scissor_snap",
        "beam_chase_diagonals",
        "beat_bar_sweep",
        "beaded_horizon_planes",
        "blackout_zone_chase",
        "quadrant_blackout_relay",
        "shutter_grid_chase",
        "pingpong_blackout_whip",
        "drop_hyper_strobe",
        "laser_crown_vortex",
        "diamond_tunnel",
        "shockwave_octagon",
        "vortex_star_polygon",
        "laser_cage_box",
        "concentric_tunnel",
        "hot_beam_matrix",
        "prism_matrix_grid",
        "double_time_whip",
        "hyper_bar_ricochet",
        "horizontal_plane_blast",
        "quad_corner_lasers",
        "strobe_tunnel_ribs",
        "split_direction_sweep",
        "orbital_plane_sweep",
        "half_plane_stagger",
        "multi_plane_cross_sweep",
        "drop_x_blades",
    ],
    BREAKDOWN: [
        "beaded_sky_canopy",
        "liquid_sky_drift",
        "cathedral_sky_vault",
        "ribbon_tide_sky",
        "prism_horizon_deck",
        "aurora_sky_curtain",
        "silk_wave_sky",
        "morphing_plane_sky",
        "breathe_horizon_sky",
        "velvet_sky_sweep",
        "dual_horizon_mirage",
        "floating_dot_constellation",
        "slow_v_canopy_sky",
    ],
    BUILD: [
        "build_lane_accel",
        "build_blackout_chase",
        "build_helix_tighten",
        "build_tunnel_collapse",
        "build_prism_tighten",
        "build_rising_sheet",
        "build_accelerating_bars",
        "build_dot_convergence",
        "build_split_pinch",
    ],
    SILENCE: [],
}

MODIFIER_POOLS = {
    MAIN: ["plane_dwell_beads", "blackout_zone_chopper", "beat_rotate", "stereo_pan_shift", "kick_zoom_pump", "counter_sweep_plane", "hat_packet_chop"],
    DROP: ["plane_dwell_beads", "blackout_zone_chopper", "fast_strobe_16th", "beat_rotate", "counter_sweep_plane", "hat_packet_chop"],
    BREAKDOWN: ["plane_dwell_beads", "stereo_pan_shift"],
    BUILD: ["kick_zoom_pump"],
    SILENCE: [],
}

SWAP_SEQUENCES: dict[str, tuple[int, ...]] = {
    "1-1-1-2": (1, 1, 1, 2),
    "1-2-1-2": (1, 2, 1, 2),
    "1-1-2-2": (1, 1, 2, 2),
    "1-2-2-1": (1, 2, 2, 1),
    "1-1-1-1-2-2-2-2": (1, 1, 1, 1, 2, 2, 2, 2),
    "1-2-1-1-1-2-1-2": (1, 2, 1, 1, 1, 2, 1, 2),
}
SWAP_SEQ_NAMES = list(SWAP_SEQUENCES.keys())

BARS_PER_SCENE = {MAIN: 4, DROP: 2, BREAKDOWN: 4, BUILD: 2, SILENCE: 99}
MORPH_BEATS = 1.5
SWAP_MORPH_BEATS = 0.25
DROP_IMPACT_BEATS = 1.5


class FXEngine:
    """High-FPS FX engine with Laser Sky, single-pass embedded beaded planes, mirrored-pair beam
    chases, transient-triggered white Breakdown bursts, rapid wide-coverage blackout chasers,
    Color-Theory Harmony Palettes & Art-Net RGB override."""

    def __init__(self, seed: int = 1, force_scene: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene, self.prev_scene = None, None, None
        self.harmony_mode: str = "auto"
        self._auto_harmony: str = "complementary"
        self.swap_mode: str | None = None
        self.swap_instant: bool = False
        self.swap_scene: str | None = None
        self.swap_hue_offset: float = 0.2
        self.swap_seq_name: str | None = None
        self.swap_beats_total: int = 0
        self._swap_start_beat: int = 0
        self._active_slot: int = 1
        self._slot_morph_left: float = 0.0

        self._drop_impact_left: float = 0.0
        self._drop_impact_variant: int = 0
        self._drop_impact_bag: list[int] = []

        self._burst_bag: list[str] = []
        self._burst_active_left: float = 0.0
        self._burst_cooldown: float = 0.0
        self._burst_params: dict = {}
        self._prev_mid_hi: float = 0.0
        self._prev_hi_flux: float = 0.0
        self._prev_snare: float = 0.0
        self._mid_hi_ema: float = 0.0

        self.modifier: str | None = None
        self._morph_left = 0.0
        self.hue, self.bars, self._last_beat = 0.0, 0, 0
        self.rt, self._last_bt, self.strobe_enabled, self.detail = 0.0, 0.0, True, 0.5

        self._raw_bt_anchor: float = 0.0
        self._wall_anchor: float = time.perf_counter()
        self._est_bps: float = 2.1

        self._bags: dict[str, list[str]] = {}
        self._mod_bags: dict[str, list[str]] = {}
        self._seq_bag: list[str] = []
        self._harmony_bag: list[str] = []

    def _trigger_breakdown_burst(self):
        if not self._burst_bag:
            self._burst_bag = list(BREAKDOWN_BURST_STYLES)
            self.rng.shuffle(self._burst_bag)
        style = self._burst_bag.pop(0)
        self._burst_params = {
            "style": style,
            "cx": float(self.rng.uniform(-0.78, 0.78)),
            "cy": float(self.rng.uniform(-0.68, 0.78)),
            "ang": float(self.rng.uniform(-math.pi * 0.48, math.pi * 0.48)),
            "scale": float(self.rng.uniform(0.80, 1.30)),
            "variant": int(self.rng.integers(0, 4)),
            "seeds": tuple(float(x) for x in self.rng.random(12)),
        }
        self._burst_active_left = BREAKDOWN_BURST_DURATION_BEATS
        self._burst_cooldown = 0.18

    def _update_breakdown_burst(self, f: Features, dbt: float):
        if dbt > 0.0:
            self._burst_active_left = max(0.0, self._burst_active_left - dbt)
            self._burst_cooldown = max(0.0, self._burst_cooldown - dbt)

        mid = min(max(float(getattr(f, "mid", 0.0)), 0.0), 1.0)
        high = min(max(float(getattr(f, "high", 0.0)), 0.0), 1.0)
        hi_flux = min(max(float(getattr(f, "hi_flux", 0.0)), 0.0), 1.0)
        snare = min(max(float(getattr(f, "snare", 0.0)), 0.0), 1.0)

        mid_hi = 0.60 * mid + 0.40 * high
        if self._mid_hi_ema <= 1e-6:
            self._mid_hi_ema = mid_hi
        else:
            self._mid_hi_ema = 0.88 * self._mid_hi_ema + 0.12 * mid_hi

        melodic_onset = (
            mid_hi > max(0.14, self._mid_hi_ema * 1.18)
            and (mid_hi - self._prev_mid_hi) > 0.032
        )
        percussive_onset = (
            (hi_flux > 0.24 and (hi_flux - self._prev_hi_flux) > 0.05)
            or (snare > 0.26 and (snare - self._prev_snare) > 0.06)
        )

        self._prev_mid_hi = mid_hi
        self._prev_hi_flux = hi_flux
        self._prev_snare = snare

        if f.section == BREAKDOWN and not self.force and self._burst_cooldown <= 0.0:
            if melodic_onset or percussive_onset:
                self._trigger_breakdown_burst()

    def _pick_auto_harmony(self, section: str):
        if section == BREAKDOWN:
            choices = ["analogous", "split_complementary", "triadic"]
        elif section == BUILD:
            choices = ["split_complementary", "triadic"]
        elif section == DROP:
            choices = ["complementary", "rectangular", "triadic"]
        else:
            choices = ["complementary", "split_complementary", "rectangular", "triadic"]
        if not self._harmony_bag:
            self._harmony_bag = list(choices)
            self.rng.shuffle(self._harmony_bag)
        cand = self._harmony_bag.pop(0)
        if cand not in choices:
            cand = choices[0]
        self._auto_harmony = cand

    def _draw_from_bag(self, section: str, exclude: str | None = None) -> str | None:
        pool = POOLS.get(section, [])
        if not pool:
            return None
        bag = self._bags.get(section, [])
        if not bag:
            bag = list(pool)
            self.rng.shuffle(bag)
            if len(bag) > 1 and bag[0] == exclude:
                bag[0], bag[-1] = bag[-1], bag[0]
        choice = bag.pop(0)
        if choice == exclude and len(pool) > 1:
            bag.append(choice)
            choice = bag.pop(0)
        self._bags[section] = bag
        return choice

    def _pick_drop_impact_variant(self) -> int:
        if not self._drop_impact_bag:
            self._drop_impact_bag = list(range(len(DROP_IMPACT_NAMES)))
            self.rng.shuffle(self._drop_impact_bag)
        return self._drop_impact_bag.pop(0)

    def _pick_modifier(self):
        if self.force:
            self.modifier = None
            return
        pool = MODIFIER_POOLS.get(self.section, [])
        chance = 0.62 if self.section == DROP else (0.52 if self.section == MAIN else 0.45)
        if not pool or float(self.rng.random()) > chance:
            self.modifier = None
            return
        bag = self._mod_bags.get(self.section, [])
        if not bag:
            bag = list(pool)
            self.rng.shuffle(bag)
        self.modifier = bag.pop(0)
        self._mod_bags[self.section] = bag

    def _pick_swap_phrase(self, current_beat: int):
        self.swap_mode = None
        self.swap_instant = False
        self.swap_scene = None
        self.swap_seq_name = None
        self.swap_beats_total = 0
        self._active_slot = 1
        self._slot_morph_left = 0.0

        if self.force or self.section not in (MAIN, DROP):
            return

        swap_chance = 0.65 if self.section == DROP else 0.50
        if float(self.rng.random()) > swap_chance:
            return

        if not self._seq_bag:
            self._seq_bag = list(SWAP_SEQ_NAMES)
            self.rng.shuffle(self._seq_bag)
        seq_name = self._seq_bag.pop(0)

        self.swap_instant = bool(float(self.rng.random()) < 0.50)
        if float(self.rng.random()) < 0.50:
            self.swap_mode = "color"
            step = int(self.rng.integers(1, len(ANCHOR_HUES_DEG)))
            self.swap_hue_offset = step * 0.1
            self.swap_seq_name = seq_name
            self.swap_beats_total = BARS_PER_SCENE[self.section] * 4
            self._swap_start_beat = current_beat
        else:
            pool = POOLS.get(self.section, [])
            if len(pool) < 2:
                return
            partner = self._draw_from_bag(self.section, exclude=self.scene)
            if not partner or partner == self.scene:
                return
            self.swap_mode = "effect"
            self.swap_scene = partner
            self.swap_seq_name = seq_name
            self.swap_beats_total = BARS_PER_SCENE[self.section] * 4
            self._swap_start_beat = current_beat

    def _pick(self, current_beat: int = 0):
        self.prev_scene = self.scene
        self._pick_auto_harmony(self.section or MAIN)
        if self.force:
            self.scene = self.force
            self._pick_modifier()
            self._pick_swap_phrase(current_beat)
            if self.prev_scene and self.prev_scene != self.scene and self.prev_scene in SCENES:
                self._morph_left = 0.65
            return

        self.scene = self._draw_from_bag(self.section, exclude=self.prev_scene)
        if not self.scene:
            self.modifier = None
            self.swap_mode = None
            self.swap_scene = None
            self.swap_seq_name = None
            return

        self._pick_modifier()
        self._pick_swap_phrase(current_beat)
        if self.prev_scene and self.prev_scene != self.scene:
            self._morph_left = MORPH_BEATS

    def update(self, f: Features, artnet: ArtNetState | None = None):
        now = time.perf_counter()
        raw_bt = f.beat_count + f.beat_phase
        bpm = getattr(f, "bpm", 0.0)
        if bpm and bpm > 40.0:
            self._est_bps = bpm / 60.0

        if abs(raw_bt - self._raw_bt_anchor) > 1e-6:
            dt_wall = now - self._wall_anchor
            if 0.002 < dt_wall < 0.5 and 0.0 < (raw_bt - self._raw_bt_anchor) < 1.0:
                measured_bps = (raw_bt - self._raw_bt_anchor) / dt_wall
                self._est_bps = 0.85 * self._est_bps + 0.15 * measured_bps
            self._raw_bt_anchor = raw_bt
            self._wall_anchor = now
            bt = raw_bt
        else:
            extra_beats = min((now - self._wall_anchor) * self._est_bps, 0.25)
            bt = raw_bt + extra_beats

        if f.section != self.section:
            self.section, self.bars = f.section, 0
            self._pick(f.beat_count)
            self._burst_active_left = 0.0
            self._burst_cooldown = 0.0
            if f.section == DROP and not self.force:
                self._drop_impact_left = DROP_IMPACT_BEATS
                self._drop_impact_variant = self._pick_drop_impact_variant()
                self._morph_left = 0.0

        if f.beat_count != self._last_beat:
            if f.bar_pos == 0:
                self.bars += 1
                if self.bars % 4 == 0:
                    self.hue += 0.1
                if self.bars % BARS_PER_SCENE[self.section] == 0:
                    self._pick(f.beat_count)
            self._last_beat = f.beat_count

        base_hue = self.hue
        if f.section == DROP:
            base_hue = [0.0, 0.1, 0.2, 0.4][(f.beat_count // 16) % 4]
        elif f.section == BREAKDOWN:
            base_hue = [0.0, 0.2, 0.5, 0.7][(f.beat_count // 8) % 4]

        active_harmony = self._auto_harmony if self.harmony_mode == "auto" else self.harmony_mode
        if artnet is not None and artnet.connected and artnet.harmony_override:
            active_harmony = artnet.harmony_override

        dbt = bt - self._last_bt
        self._last_bt = bt
        valid_dbt = dbt if (0 < dbt < 1) else 0.0
        if valid_dbt > 0.0:
            self.rt += valid_dbt * (1 + 4 * f.build if f.section == BUILD else 1)
            if self._morph_left > 0.0:
                self._morph_left = max(0.0, self._morph_left - valid_dbt)
            if self._slot_morph_left > 0.0:
                self._slot_morph_left = max(0.0, self._slot_morph_left - valid_dbt)
            if self._drop_impact_left > 0.0:
                self._drop_impact_left = max(0.0, self._drop_impact_left - valid_dbt)

        self._update_breakdown_burst(f, valid_dbt)

        if not self.scene or (f.section == SILENCE and not self.force):
            return [], None

        if f.section == DROP and self._drop_impact_left > 0.0 and not self.force:
            pal_drop = build_harmony_palette(base_hue, active_harmony, artnet)
            c_drop = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail, harmony_colors=pal_drop)
            progress = 1.0 - (self._drop_impact_left / DROP_IMPACT_BEATS)
            paths = _drop_impact_effect(c_drop, progress, self._drop_impact_variant)
            if progress > 0.78:
                paths = _morph_paths(paths, SCENES[self.scene](c_drop), (progress - 0.78) / 0.22)
            paths = _apply_split_color_harmony(paths, pal_drop, artnet)
            return _optimize_path_order(paths), DROP_IMPACT_NAMES[self._drop_impact_variant]

        elapsed_beats = f.beat_count - self._swap_start_beat
        swap_active = (
            self.swap_mode is not None
            and self.swap_seq_name in SWAP_SEQUENCES
            and 0 <= elapsed_beats < self.swap_beats_total
        )

        if swap_active:
            seq = SWAP_SEQUENCES[self.swap_seq_name]
            slot = seq[elapsed_beats % len(seq)]
            if slot != self._active_slot:
                self._active_slot = slot
                self._slot_morph_left = 0.0 if self.swap_instant else SWAP_MORPH_BEATS

            if self.swap_mode == "color":
                active_hue = base_hue if slot == 1 else (base_hue + self.swap_hue_offset)
                pal = build_harmony_palette(active_hue, active_harmony, artnet)
                c = Ctx(f, bt, active_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail, harmony_colors=pal)
                paths = SCENES[self.scene](c)
                if not self.swap_instant and self._slot_morph_left > 0.0:
                    prev_hue = (base_hue + self.swap_hue_offset) if slot == 1 else base_hue
                    pal_prev = build_harmony_palette(prev_hue, active_harmony, artnet)
                    c_prev = Ctx(f, bt, prev_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail, harmony_colors=pal_prev)
                    paths = _morph_paths(SCENES[self.scene](c_prev), paths, 1.0 - (self._slot_morph_left / SWAP_MORPH_BEATS))
            else:
                pal = build_harmony_palette(base_hue, active_harmony, artnet)
                c = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail, harmony_colors=pal)
                primary_name = self.scene if slot == 1 else self.swap_scene
                other_name = self.swap_scene if slot == 1 else self.scene
                paths = SCENES[primary_name](c)
                if not self.swap_instant and self._slot_morph_left > 0.0 and other_name in SCENES:
                    paths = _morph_paths(SCENES[other_name](c), paths, 1.0 - (self._slot_morph_left / SWAP_MORPH_BEATS))
        else:
            pal = build_harmony_palette(base_hue, active_harmony, artnet)
            c = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail, harmony_colors=pal)
            paths = SCENES[self.scene](c)

        if self._morph_left > 0.0 and self.prev_scene in SCENES:
            paths = _morph_paths(SCENES[self.prev_scene](c), paths, 1.0 - (self._morph_left / MORPH_BEATS))

        paths = _apply_modifier(paths, self.modifier, c)

        burst_active = (
            f.section == BREAKDOWN
            and not self.force
            and self._burst_active_left > 0.0
        )
        if burst_active:
            burst_prog = 1.0 - (self._burst_active_left / BREAKDOWN_BURST_DURATION_BEATS)
            paths = _apply_breakdown_transient_burst(paths, burst_prog, self._burst_params, c)

        paths = _apply_split_color_harmony(paths, pal, artnet)

        if swap_active and self.swap_mode == "effect":
            mode_tag = "cut" if self.swap_instant else "morph"
            label = f"{self.scene}<>{self.swap_scene} [{mode_tag} {self.swap_seq_name}]"
        elif swap_active and self.swap_mode == "color":
            mode_tag = "cut" if self.swap_instant else "morph"
            label = f"{self.scene} [color-{mode_tag} {self.swap_seq_name}]"
        else:
            label = self.scene

        if burst_active:
            label = f"{label}+white:{self._burst_params.get('style', '')}"
        elif self.modifier:
            label = f"{label}+{self.modifier}"
        if artnet is not None and artnet.connected:
            label = f"{label} · ART-NET"
        return _optimize_path_order(paths), label
