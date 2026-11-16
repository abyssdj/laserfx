"""LaserFX Effect Engine: coherent aerial choreography (Liquid Sky, Tunnels, Cones, Fans,
Hot Beams, Waves), nearest-neighbor path optimization, and RGB-safe morphing.

Coordinates are in [-1, 1]; colours are RGB in 0..1.
Designed for single or twin X-mirrored projectors.
"""
from __future__ import annotations

import colorsys
import math
import time
from dataclasses import dataclass

import numpy as np

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
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)), dtype=float)


def line(a, b, rgb, dwell: int = 0) -> Path:
    return Path(np.array([a, b], dtype=float), rgb, dwell)


def hot_dot(pt, rgb=WHITE, dwell: int = 2) -> Path:
    """Crisp stationary aerial hot-beam pip."""
    x = float(np.clip(pt[0], -LIM, LIM))
    y = float(np.clip(pt[1], -LIM, LIM))
    dx = 0.006 if x < (LIM - 0.02) else -0.006
    return Path(np.array([[x, y], [x + dx, y]], dtype=float), rgb, dwell)


def dotted_line(a, b, rgb, dots: int = 5, dot_len: float = 0.036, phase: float = 0.0) -> list[Path]:
    """Evenly spaced aerial beam pips along segment a -> b."""
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
    """Cleanly separated dashed aerial segments."""
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
    """Ray from (ox, oy) at `ang` (0 = straight up), clipped cleanly to [-bound, bound]."""
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
    """Full-span aerial sheet passing through (cx, cy) perpendicular to `angle`, clipped to [-span, span]."""
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
    """Closed geometric polygon (triangle, square, hexagon, circle) with clean vertices."""
    angs = np.linspace(0.0, TAU, sides + 1) + rot
    xs = np.clip(cx + rx * np.cos(angs), -LIM, LIM)
    ys = np.clip(cy + ry * np.sin(angs), -LIM, LIM)
    return Path(np.stack([xs, ys], axis=1), rgb, dwell=0)


def _reverse_path(p: Path) -> Path:
    pts = p.pts[::-1].copy()
    rgb = p.rgb[::-1].copy() if np.asarray(p.rgb).ndim == 2 else p.rgb
    return Path(pts, rgb, p.dwell)


def _optimize_path_order(paths: list[Path], tol: float = 0.025) -> list[Path]:
    """Deduplicates overlapping lines and sorts remaining paths via nearest-neighbor
    with bidirectional reversal to minimize blanked galvo travel."""
    if not paths:
        return []

    deduped: list[Path] = []
    sigs: list[tuple[np.ndarray, np.ndarray]] = []
    for p in paths:
        if len(p.pts) == 2:
            a, b = p.pts[0], p.pts[1]
            if float(np.hypot(b[0] - a[0], b[1] - a[1])) >= 0.02:
                dup = False
                for ka, kb in sigs:
                    d_same = max(float(np.hypot(a[0] - ka[0], a[1] - ka[1])), float(np.hypot(b[0] - kb[0], b[1] - kb[1])))
                    d_flip = max(float(np.hypot(a[0] - kb[0], a[1] - kb[1])), float(np.hypot(b[0] - ka[0], b[1] - ka[1])))
                    if min(d_same, d_flip) < tol:
                        dup = True
                        break
                if dup:
                    continue
                sigs.append((a, b))
        deduped.append(p)

    if len(deduped) <= 1:
        return deduped

    remaining = list(deduped)
    ordered: list[Path] = [remaining.pop(0)]
    cur_pos = ordered[0].pts[-1]

    while remaining:
        best_idx = 0
        best_rev = False
        best_dist = 1e9
        for idx, cand in enumerate(remaining):
            d_fwd = float(np.hypot(cand.pts[0, 0] - cur_pos[0], cand.pts[0, 1] - cur_pos[1]))
            d_rev = float(np.hypot(cand.pts[-1, 0] - cur_pos[0], cand.pts[-1, 1] - cur_pos[1]))
            if d_fwd < best_dist:
                best_dist = d_fwd
                best_idx = idx
                best_rev = False
            if d_rev < best_dist:
                best_dist = d_rev
                best_idx = idx
                best_rev = True
        chosen = remaining.pop(best_idx)
        if best_rev:
            chosen = _reverse_path(chosen)
        ordered.append(chosen)
        cur_pos = chosen.pts[-1]

    return ordered


# ---------------------------------------------------------------- smooth musical motion helpers
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


def snare_swell(c: Ctx) -> float:
    return min(max(getattr(c.f, "snare", 0.0), 0.0), 1.0)


def strobe_gate(c: Ctx, div: float = 4.0, duty: float = 0.48) -> bool:
    return (not c.strobe) or ((c.bt * div) % 1.0) < duty


PALETTES = [
    (0.54, 0.84, 0.85),   # Ice Cyan + Electric Magenta
    (0.00, 0.56, 0.12),   # Pure Crimson + Crisp White
    (0.76, 0.11, 0.90),   # Deep UV Violet + Solar Amber
    (0.36, 0.55, 0.15),   # Acid Emerald + Ice White
    (0.62, 0.04, 0.95),   # Cobalt Blue + Laser Orange
]


@dataclass
class Ctx:
    f: Features
    bt: float
    hue: float
    rng: np.random.Generator
    rt: float = 0.0
    strobe: bool = True
    detail: float = 0.5

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
        p_idx = int(round(abs(self.hue * 10))) % len(PALETTES)
        h_pri, h_acc, s_acc = PALETTES[p_idx]
        if idx % 2 == 0:
            return hsv(h_pri, max(1.0 - white_boost, 0.0), v)
        return hsv(h_acc, max(s_acc - white_boost, 0.0), v)


# ======================================================================== CALIBRATION TEST PATTERNS

TEST_PATTERNS = {
    "ilda_grid": "ILDA Tuning Grid",
    "corner_box": "Corner Sharpness Box",
    "rgb_align": "RGB Color Alignment",
}


def render_test_pattern(kind: str, bt: float) -> list[Path]:
    """Built-in scanner & color alignment test patterns (modeled after Pangolin QuickShow test frames)."""
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
        # Alternating R, G, B vertical bars + white frame to align Color Shift (Galvo Lag) & Blanking Delay
        s = 0.82
        out = [
            Path(np.array([[-s, -s], [s, -s], [s, s], [-s, s], [-s, -s]], float), WHITE, dwell=0),
        ]
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

    # Default: "ilda_grid" — outer border, inner circle, crosshairs, and corner boxes
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
    col_acc = c.palette(0, v=fade, white_boost=0.55)
    eased = math.sin(min(progress, 1.0) * math.pi * 0.5)
    out: list[Path] = []

    if variant == 0:
        n_rays = c.count(6, 8)
        spin = 0.28 * progress
        reach = 0.45 + 1.85 * eased
        for i in range(n_rays):
            ang = spin + (i / max(n_rays, 1)) * TAU
            p0, p1 = ray(0.0, 0.0, ang, reach)
            out.append(line(p0, p1, col_main if i % 2 == 0 else col_acc, dwell=0))

    elif variant == 1:
        for idx, frac in enumerate((0.35, 0.85)):
            y = 0.14 + 0.74 * eased * frac
            out.append(line((-LIM, y), (LIM, y), col_main if idx == 0 else col_acc, dwell=0))
            out.append(line((LIM, -y), (-LIM, -y), col_main if idx == 0 else col_acc, dwell=0))

    elif variant == 2:
        for idx, r0 in enumerate((0.25, 0.55)):
            r = min(0.18 + 0.74 * eased + r0 * 0.35, LIM)
            out.append(polygon_path(0.0, 0.0, r, r * 0.88, 16, progress * 0.4, col_main if idx == 0 else col_acc))

    else:
        span = 0.15 + 0.76 * eased
        out.extend([
            line((-span, -LIM), (-span, LIM), col_main, dwell=0),
            line((span, LIM), (span, -LIM), col_main, dwell=0),
            line((-LIM, span), (LIM, span), col_acc, dwell=0),
            line((LIM, -span), (-LIM, -span), col_acc, dwell=0),
        ])

    return out


# ======================================================================== BREAKDOWN SCENES (ATMOSPHERIC LIQUID SKY)

def sc_liquid_sky_drift(c: Ctx) -> list[Path]:
    planes = c.count(1, 2)
    tilt = 0.20 * smooth_wave(c.bt, 16.0)
    y_base = 0.26 + 0.16 * smooth_wave(c.bt, 32.0)
    out: list[Path] = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * 0.34
        y0 = float(np.clip(y_base + gap - tilt, -LIM, LIM))
        y1 = float(np.clip(y_base + gap + tilt, -LIM, LIM))
        out.append(line((-LIM, y0), (LIM, y1), c.palette(p, v=0.84), dwell=0))
    return out


def sc_morphing_plane_sky(c: Ctx) -> list[Path]:
    morph = smooth_wave(c.bt, 16.0)
    y_center = 0.22 + 0.14 * smooth_wave(c.bt, 24.0)
    tilt = 0.34 * morph
    out = [
        line((-LIM, float(np.clip(y_center - tilt, -LIM, LIM))),
             (LIM, float(np.clip(y_center + tilt, -LIM, LIM))),
             c.palette(0, v=0.84), dwell=0),
    ]
    if c.detail >= 0.35:
        out.append(
            line((-LIM, float(np.clip(y_center + tilt * 0.8 - 0.32, -LIM, LIM))),
                 (LIM, float(np.clip(y_center - tilt * 0.8 - 0.32, -LIM, LIM))),
                 c.palette(1, v=0.78), dwell=0)
        )
    return out


def sc_breathe_horizon_sky(c: Ctx) -> list[Path]:
    spread = 0.16 + 0.26 * (0.5 + 0.5 * smooth_wave(c.bt, 16.0))
    y_mid = 0.20 + 0.12 * smooth_wave(c.bt, 32.0)
    rock = 0.12 * smooth_wave(c.bt, 12.0)
    out: list[Path] = []
    for idx, sign in enumerate((-1.0, 1.0)):
        if idx == 1 and c.detail < 0.3:
            break
        y0 = float(np.clip(y_mid + sign * spread - rock, -LIM, LIM))
        y1 = float(np.clip(y_mid + sign * spread + rock, -LIM, LIM))
        out.append(line((-LIM, y0), (LIM, y1), c.palette(idx, v=0.80), dwell=0))
    return out


def sc_velvet_sky_sweep(c: Ctx) -> list[Path]:
    u = beat_sweep(c.bt, beats_per_pass=8.0, alternate=True)
    x_center = -0.72 + 1.44 * u
    blades = c.count(1, 2)
    width = 0.32 + 0.08 * smooth_wave(c.bt, 16.0)
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * width
        x = float(np.clip(x_center + off, -LIM, LIM))
        out.append(line((x, -LIM), (x, LIM), c.palette(i % 2, v=0.80), dwell=0))
    return out


def sc_silk_wave_sky(c: Ctx) -> list[Path]:
    x = np.linspace(-LIM, LIM, 26)
    ph = c.bt * TAU / 16.0
    y = np.clip(0.28 + 0.14 * smooth_wave(c.bt, 24.0) + 0.16 * np.sin(1.4 * math.pi * x + ph), -LIM, LIM)
    return [Path(np.stack([x, y], axis=1), c.palette(0, v=0.84), dwell=0)]


def sc_dual_horizon_mirage(c: Ctx) -> list[Path]:
    ang = 0.28 * smooth_wave(c.bt, 20.0)
    y_base = 0.24 + 0.12 * smooth_wave(c.bt, 32.0)
    return [
        line((-LIM, y_base - ang), (LIM, y_base + ang), c.palette(0, v=0.82), dwell=0),
        line((-LIM, y_base + ang * 0.6 - 0.34), (LIM, y_base - ang * 0.6 - 0.34), c.palette(1, v=0.76), dwell=0),
    ]


def sc_floating_dot_constellation(c: Ctx) -> list[Path]:
    y_sky = 0.10 + 0.12 * smooth_wave(c.bt, 24.0)
    out: list[Path] = [line((-LIM, y_sky), (LIM, y_sky), c.palette(0, v=0.76), dwell=0)]
    n_stars = c.count(5, 7)
    for s in range(n_stars):
        u = (s + 0.5) / n_stars
        x = -0.82 + 1.64 * u + 0.05 * math.sin(c.bt * TAU / 16.0 + s)
        y = 0.44 + 0.24 * math.sin(c.bt * TAU / 20.0 + s * 1.7)
        col = WHITE * 0.85 if s % 2 == 0 else c.palette(1, v=0.84)
        out.append(hot_dot((x, y), col, dwell=2))
    return out


def sc_slow_v_canopy_sky(c: Ctx) -> list[Path]:
    y_apex = 0.12 + 0.16 * smooth_wave(c.bt, 24.0)
    wing_lift = 0.28 + 0.22 * smooth_wave(c.bt, 16.0)
    y_wing = float(np.clip(y_apex + wing_lift, -LIM, LIM))
    pts = np.array([[-LIM, y_wing], [0.0, y_apex], [LIM, y_wing]], float)
    return [Path(pts, c.palette(0, v=0.84), dwell=0)]


def sc_aurora_sky_curtain(c: Ctx) -> list[Path]:
    x = np.linspace(-LIM, LIM, 26)
    out: list[Path] = []
    for layer in range(c.count(1, 2)):
        ph = c.bt * TAU / 18.0 + layer * 2.1
        y = 0.34 + 0.12 * smooth_wave(c.bt, 28.0) + 0.14 * np.sin(1.4 * math.pi * x + ph) - layer * 0.34
        pts = np.stack([x, np.clip(y, -LIM, LIM)], axis=1)
        out.append(Path(pts, c.palette(layer, v=0.82), dwell=0))
    return out


def _apply_synth_accent(paths: list[Path], age_beats: float, kind: int, c: Ctx) -> list[Path]:
    if age_beats < 0.0 or age_beats >= 1.0:
        return paths
    env = math.sin(min(age_beats * 4.0, 1.0) * math.pi * 0.5) * ((1.0 - age_beats) ** 1.4)
    if env <= 0.03:
        return paths
    col = WHITE * (0.72 * env)
    if kind % 2 == 0:
        y = -0.45 + 0.42 * age_beats
        tilt = 0.12 * (1.0 if kind % 4 == 0 else -1.0)
        return paths + [line((-LIM, y - tilt), (LIM, y + tilt), col, dwell=0)]
    ox, oy = -0.36, -0.32
    r = 0.14 + 0.26 * age_beats
    burst = [hot_dot((ox + r * math.cos(k * TAU / 4.0 + age_beats * 0.8),
                      oy + r * 0.7 * math.sin(k * TAU / 4.0 + age_beats * 0.8)), col, dwell=1)
             for k in range(4)]
    return paths + burst


# ======================================================================== MAIN & DROP SCENES (COHERENT, ARCHITECTURAL GEOMETRY)

def sc_concentric_tunnel(c: Ctx) -> list[Path]:
    """Classic Pangolin-style expanding concentric laser tunnel (circle/octagon)."""
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
        col = c.palette(r_idx % 2, v=1.0, white_boost=0.65 * pump if r_idx == 0 else 0.0)
        out.append(polygon_path(cx, cy, rad, rad * 0.88, 16, rot + r_idx * 0.2, col))
    return out


def sc_rotating_laser_cone(c: Ctx) -> list[Path]:
    """3D perspective rotating aerial cone / spirograph vortex."""
    pump = beat_pump(c)
    n_pts = 32
    t = np.linspace(0.0, TAU, n_pts)
    tilt_x = 0.32 * smooth_wave(c.bt, 8.0)
    tilt_y = 0.26 * smooth_wave(c.bt, 12.0)
    rx = 0.56 + 0.24 * pump
    ry = 0.48 + 0.20 * pump
    cx = 0.16 * math.sin(c.bt * TAU / 16.0)
    cy = 0.12 * math.cos(c.bt * TAU / 16.0)
    xs = np.clip(cx + rx * np.cos(t) + tilt_x * np.sin(2.0 * t + c.bt), -LIM, LIM)
    ys = np.clip(cy + ry * np.sin(t) + tilt_y * np.cos(2.0 * t - c.bt), -LIM, LIM)
    col = c.palette(0, v=1.0, white_boost=0.55 * pump)
    out = [Path(np.stack([xs, ys], axis=1), col, dwell=0)]
    if c.detail >= 0.45:
        xs2 = np.clip(cx + (rx * 0.58) * np.cos(t + 0.5), -LIM, LIM)
        ys2 = np.clip(cy + (ry * 0.58) * np.sin(t + 0.5), -LIM, LIM)
        out.append(Path(np.stack([xs2, ys2], axis=1), c.palette(1, v=0.9), dwell=0))
    return out


def sc_split_direction_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u_top = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    u_bot = 1.0 - u_top
    x_top = -0.82 + 1.64 * u_top
    x_bot = -0.82 + 1.64 * u_bot
    blades = c.count(2, 2)
    gap = 0.18 + 0.05 * pump
    out: list[Path] = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        xt = float(np.clip(x_top + off, -LIM, LIM))
        xb = float(np.clip(x_bot + off, -LIM, LIM))
        out.append(line((xt, 0.10), (xt, LIM), c.palette(0, white_boost=0.65 * pump if i == 0 else 0.0), dwell=0))
        out.append(line((xb, -LIM), (xb, -0.10), c.palette(1, white_boost=0.65 * pump if i == 0 else 0.0), dwell=0))
    return out


def sc_half_plane_stagger(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u1 = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    u2 = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)
    y_left = -0.82 + 1.64 * u1
    y_right = -0.82 + 1.64 * (1.0 - u2)
    x_top = -0.84 + 1.68 * u2
    x_bot = -0.84 + 1.68 * (1.0 - u1)
    return [
        line((-LIM, y_left), (-0.10, y_left + 0.08), c.palette(0, white_boost=0.6 * pump), dwell=0),
        line((LIM, y_right), (0.10, y_right - 0.08), c.palette(1, white_boost=0.6 * pump), dwell=0),
        line((x_top, 0.10), (x_top, LIM), c.palette(1, white_boost=0.45 * pump), dwell=0),
        line((x_bot, -LIM), (x_bot, -0.10), c.palette(0, white_boost=0.45 * pump), dwell=0),
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
        col = c.palette(i % 2, v=1.0, white_boost=0.75 * pump if i == 0 else 0.0)
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
        col = c.palette(i % 2, v=1.0, white_boost=0.65 * pump if i == 1 else 0.0)
        out.append(line((-LIM, y0), (LIM, y1), col, dwell=0))
    return out


def sc_dotted_fan_burst(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    n = c.count(5, 7, max(c.energy, 0.45))
    aim = 0.22 + 0.38 * smooth_wave(c.bt, 8.0)
    spread = 0.44 + 0.20 * pump
    ox = -0.54
    out: list[Path] = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        reach = 1.75 + 0.25 * math.cos(u * math.pi) + 0.18 * pump
        p0, p1 = ray(ox, -LIM, ang, reach)
        col = c.palette(i % 2, v=0.95)
        if i % 2 == 1:
            out.extend(dotted_line(p0, p1, col, dots=4, dot_len=0.04, phase=c.bt * 2.0))
        else:
            out.append(line(p0, p1, col, dwell=0))
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
        col = WHITE if (r == 0 and pump > 0.35) else c.palette(r % 2, v=1.0)
        out.extend(dotted_line((-LIM, y0), (LIM, y1), col, dots=dots_per_row, dot_len=0.038, phase=c.bt * 1.5 + r * 0.35))
    return out


def sc_drop_hyper_strobe(c: Ctx) -> list[Path]:
    div = 4.0 if (int(c.bt / 2) % 2 == 1) else 2.0
    if not strobe_gate(c, div=div, duty=0.48):
        return []
    pump = beat_pump(c)
    step = int(c.bt * div)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    out: list[Path] = []
    if step % 3 == 0:
        x = -0.80 + 1.60 * u
        out.append(line((x, -LIM), (x, LIM), WHITE, dwell=0))
        x2 = float(np.clip(x - 0.24 if x > -0.56 else x + 0.24, -LIM, LIM))
        out.append(line((x2, LIM), (x2, -LIM), c.palette(0), dwell=0))
    elif step % 3 == 1:
        y_top = 0.18 + 0.72 * u
        y_bot = -0.18 - 0.72 * u
        out.append(line((-LIM, y_top), (LIM, y_top), c.palette(1, white_boost=0.75 * pump), dwell=0))
        out.append(line((LIM, y_bot), (-LIM, y_bot), WHITE, dwell=0))
    else:
        n = c.count(5, 6)
        aim = 0.20 + 0.42 * u
        for i in range(n):
            frac = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(-0.56, -LIM, aim + frac * 0.92, 2.3)
            out.append(line(p0, p1, c.palette(i % 2, white_boost=0.55), dwell=0))
    return out


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
        col = c.palette(i % 2, v=1.0, white_boost=0.70 * pump if i == 0 else 0.0)
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
        col = c.palette(i % 2, v=1.0, white_boost=0.70 * pump if i == 0 else 0.0)
        out.append(line((-LIM, y0), (LIM, y1), col, dwell=0))
    return out


def sc_orbital_plane_sweep(c: Ctx) -> list[Path]:
    """Smoothly rotating dual aerial planes (replaces erratic random angle jumps)."""
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
            col = c.palette(i % 2, v=1.0, white_boost=0.75 * pump if i == 0 else 0.0)
            out.append(line(seg[0], seg[1], col, dwell=0))
    return out


def sc_multi_plane_cross_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    n_planes = c.count(2, 2)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    v = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)
    out: list[Path] = []
    for i in range(n_planes):
        pos = (u + i / max(n_planes, 1)) % 1.0
        x = -0.86 + 1.72 * pos
        out.append(line((x, -LIM), (x, LIM), c.palette(0, v=1.0, white_boost=0.6 * pump if i == 0 else 0.0), dwell=0))
    for j in range(n_planes):
        pos = (1.0 - v + j / max(n_planes, 1)) % 1.0
        y = -0.86 + 1.72 * pos
        out.append(line((-LIM, y), (LIM, y), c.palette(1, v=1.0, white_boost=0.6 * pump if j == 0 else 0.0), dwell=0))
    return out


def sc_twin_scissor_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    snare = snare_swell(c)
    n = c.count(4, 5, max(c.energy, 0.45))
    swing = 0.40 * smooth_wave(c.bt, 8.0)
    out: list[Path] = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = 0.32 + swing + u * (0.36 + 0.16 * pump)
        p0, p1 = ray(-0.74, -LIM, ang, 2.2)
        out.append(line(p0, p1, c.palette(0, v=1.0), dwell=0))
    for i in range(max(3, n - 1)):
        u = (i / max(n - 2, 1)) - 0.5
        ang = math.pi - (0.34 - swing * 0.85 + u * (0.34 + 0.16 * snare))
        p0, p1 = ray(-0.28, LIM, ang, 2.2)
        out.append(line(p0, p1, c.palette(1, v=0.88 + 0.12 * snare), dwell=0))
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
        col = c.palette(p % 2, v=1.0, white_boost=0.75 * pump if p == 0 else 0.0)
        out.append(line((-LIM, y0), (LIM, y1), col, dwell=0))
    return out


def sc_quadrant_mirror_sweep(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    sweep_ang = 0.22 + 0.96 * u
    p0, p1 = ray(-LIM, -LIM, sweep_ang, 1.9 + 0.3 * pump)
    p2, p3 = ray(-LIM, LIM, math.pi - sweep_ang, 1.9 + 0.3 * pump)
    return [
        line(p0, p1, c.palette(0, white_boost=0.65 * pump), dwell=0),
        line(p3, p2, c.palette(1, white_boost=0.65 * pump), dwell=0),
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
            col = c.palette(i % 2, white_boost=0.70 * pump if i == 0 else 0.0)
            out.append(line(seg[0], seg[1], col, dwell=0))
    return out


def sc_crisscross_ribbon_planes(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_a = -0.72 + 1.44 * u
    y_b = 0.72 - 1.44 * u
    return [
        line((-LIM, y_a), (LIM, float(np.clip(y_a + 0.38, -LIM, LIM))), c.palette(0, white_boost=0.6 * pump), dwell=0),
        line((LIM, y_b), (-LIM, float(np.clip(y_b + 0.38, -LIM, LIM))), c.palette(1, white_boost=0.6 * pump), dwell=0),
    ]


def sc_alternating_horizon_gates(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    gap = 0.20 + 0.68 * (0.5 - 0.5 * math.cos(math.pi * c.bt))
    out = [
        line((-LIM, gap), (LIM, gap), c.palette(0, white_boost=0.70 * pump), dwell=0),
        line((LIM, -gap), (-LIM, -gap), c.palette(1, white_boost=0.70 * pump), dwell=0),
    ]
    if c.detail >= 0.45:
        out.extend(dotted_line((-0.88, 0.0), (0.88, 0.0), WHITE * (0.55 + 0.45 * pump), dots=4, dot_len=0.035, phase=c.bt * 2.0))
    return out


def sc_prism_dot_fan(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    n = c.count(4, 6)
    aim = 0.28 + 0.38 * smooth_wave(c.bt, 6.0)
    spread = 0.44 + 0.20 * pump
    out: list[Path] = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(-0.62, -LIM, ang, 2.25)
        out.append(line(p0, p1, c.palette(i % 2), dwell=0))
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
        line((x1, -LIM), (x1, LIM), c.palette(0, white_boost=0.8 * pump), dwell=0),
        line((x2, LIM), (x2, -LIM), c.palette(1, white_boost=0.8 * pump), dwell=0),
        line((-LIM, y1), (LIM, y1), WHITE if pump > 0.45 else c.palette(0), dwell=0),
    ]


def sc_horizontal_plane_blast(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    eased = math.sin(frac * math.pi * 0.5)
    pairs = c.count(2, 2)
    out: list[Path] = []
    for p in range(pairs):
        y = 0.18 + 0.70 * eased * ((p + 1) / pairs)
        col = c.palette(p % 2, white_boost=0.75 * pump if p == 0 else 0.0)
        out.append(line((-LIM, y), (LIM, y), col, dwell=0))
        out.append(line((LIM, -y), (-LIM, -y), col, dwell=0))
    return out


def sc_hot_beam_matrix(c: Ctx) -> list[Path]:
    """Synchronized aerial hot-beam matrix (classic club/festival look)."""
    pump = beat_pump(c)
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
            col = WHITE if (col_i % 2 == 0 and pump > 0.45) else c.palette((r + col_i) % 2)
            out.append(hot_dot((x, y), col, dwell=2))
    return out


def sc_quad_corner_lasers(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    blades = c.count(2, 3)
    out: list[Path] = []
    for i in range(blades):
        ang = 0.24 + 0.80 * u + (i - 1) * 0.20
        p0, p1 = ray(-0.90, -0.90, ang, 2.5)
        out.append(line(p0, p1, c.palette(i % 2, white_boost=0.75 * pump), dwell=0))
    return out


def sc_strobe_tunnel_ribs(c: Ctx) -> list[Path]:
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    layers = c.count(2, 3)
    cx = -0.18 * smooth_wave(c.bt, 8.0)
    out: list[Path] = []
    for l in range(layers):
        s = 0.26 + 0.64 * (((frac + l / layers) % 1.0))
        x0 = max(cx - s, -LIM)
        x1 = min(cx + s, LIM)
        y0 = -s * 0.85
        y1 = s * 0.85
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)
        out.append(Path(pts, c.palette(l % 2, white_boost=0.65 * pump), dwell=0))
    return out


# ======================================================================== BUILD SECTION SCENES

def sc_build_converge_sweep(c: Ctx) -> list[Path]:
    b = c.f.build
    n = c.count(5, 7, max(b, 0.45))
    sweep = (0.44 * (1.0 - 0.65 * (b ** 2))) * math.sin(c.rt * TAU / 4.0)
    bias = 0.30 * (1.0 - 0.7 * b)
    spread = 0.52 * (1.0 - 0.76 * (b ** 2)) + 0.08
    ox = -0.56 * (1.0 - 0.70 * b)
    out: list[Path] = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = bias + sweep + u * 2.0 * spread
        p0, p1 = ray(ox, -LIM, ang, 2.3)
        col = c.palette(i % 2, v=0.78 + 0.22 * b, white_boost=0.85 * (b ** 2))
        out.append(line(p0, p1, col, dwell=0))
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
        col = c.palette(p % 2, v=0.8 + 0.2 * b, white_boost=0.8 * (b ** 2))
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
        col = c.palette(i % 2, v=0.78 + 0.22 * b, white_boost=0.85 * (b ** 2))
        out.append(line((x, -LIM), (x, LIM), col, dwell=0))
    return out


def sc_build_accelerating_bars(c: Ctx) -> list[Path]:
    b = c.f.build
    u = beat_sweep(c.rt, beats_per_pass=0.5, alternate=True)
    span = 0.86 * (1.0 - 0.65 * b) + 0.14
    x = -span + 2.0 * span * u
    col = c.palette(0, v=0.8 + 0.2 * b, white_boost=0.9 * (b ** 2))
    return [
        line((x, -LIM), (x, LIM), col, dwell=0),
        line((-x, LIM), (-x, -LIM), WHITE * (0.45 + 0.55 * b), dwell=0),
    ]


def sc_build_dot_convergence(c: Ctx) -> list[Path]:
    b = c.f.build
    n_dots = c.count(8, 12)
    r = 0.86 * (1.0 - 0.74 * b) + 0.12
    ox = -0.28 * (1.0 - b)
    out: list[Path] = []
    for i in range(n_dots):
        ang = c.rt * 2.2 + (i / n_dots) * TAU
        px = ox + r * math.cos(ang)
        py = r * math.sin(ang)
        col = WHITE if (b > 0.65 or i % 2 == 0) else c.palette(i % 2)
        out.append(hot_dot((px, py), col, dwell=1))
    return out


def sc_build_split_pinch(c: Ctx) -> list[Path]:
    b = c.f.build
    y_gap = 0.84 * (1.0 - 0.80 * b) + 0.10
    wobble = 0.14 * (1.0 - b) * math.sin(c.rt * TAU / 2.0)
    col = c.palette(1, v=0.8 + 0.2 * b, white_boost=0.85 * (b ** 2))
    return [
        line((-LIM, y_gap - wobble), (LIM, y_gap + wobble), col, dwell=0),
        line((LIM, -y_gap + wobble), (-LIM, -y_gap - wobble), col, dwell=0),
    ]


# ======================================================================== SECONDARY MODIFIERS

def _apply_modifier(paths: list[Path], mod: str, c: Ctx) -> list[Path]:
    if not mod or not paths:
        return paths

    if mod == "fast_strobe_16th":
        div = 4.0 if (int(c.bt) % 2 == 1) else 2.0
        if not strobe_gate(c, div=div, duty=0.5):
            return []
        return [Path(p.pts, np.clip(p.rgb * 0.5 + WHITE * 0.5, 0.0, 1.0), p.dwell) for p in paths]

    if mod == "beat_rotate":
        ang = 0.30 * smooth_wave(c.bt, 4.0)
        ca, sa = math.cos(ang), math.sin(ang)
        out: list[Path] = []
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
        pump = beat_pump(c)
        scale = 0.86 + 0.14 * pump
        return [Path(np.clip(p.pts * scale, -LIM, LIM), p.rgb, p.dwell) for p in paths]

    if mod == "counter_sweep_plane":
        u = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)
        y = -0.84 + 1.68 * u
        extra = [line((-LIM, y), (LIM, y), c.palette(1, v=0.9, white_boost=0.5 * beat_pump(c)), dwell=0)]
        return paths + extra

    if mod == "hat_packet_chop":
        if c.hi < 0.38:
            return paths
        out = []
        for idx, p in enumerate(paths):
            if len(p.pts) == 2 and float(np.hypot(*(p.pts[1] - p.pts[0]))) > 0.05:
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
    x = np.interp(u, s, pts[:, 0])
    y = np.interp(u, s, pts[:, 1])
    return np.stack([x, y], axis=1)


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
        pts0 = _resample_poly(p0.pts, v_count)
        pts1 = _resample_poly(p1.pts, v_count)
        pts = pts0 * (1.0 - t) + pts1 * t

        c0 = _resample_rgb(p0.rgb, v_count)
        c1 = _resample_rgb(p1.rgb, v_count)
        rgb = c0 * (1.0 - t) + c1 * t
        dwell = int(round(p0.dwell * (1.0 - t) + p1.dwell * t))
        morphed.append(Path(pts, rgb, dwell))
    return morphed


SCENES = {
    "concentric_tunnel": sc_concentric_tunnel,
    "rotating_laser_cone": sc_rotating_laser_cone,
    "hot_beam_matrix": sc_hot_beam_matrix,
    "orbital_plane_sweep": sc_orbital_plane_sweep,
    "liquid_sky_drift": sc_liquid_sky_drift,
    "morphing_plane_sky": sc_morphing_plane_sky,
    "breathe_horizon_sky": sc_breathe_horizon_sky,
    "velvet_sky_sweep": sc_velvet_sky_sweep,
    "silk_wave_sky": sc_silk_wave_sky,
    "dual_horizon_mirage": sc_dual_horizon_mirage,
    "floating_dot_constellation": sc_floating_dot_constellation,
    "slow_v_canopy_sky": sc_slow_v_canopy_sky,
    "aurora_sky_curtain": sc_aurora_sky_curtain,
    "split_direction_sweep": sc_split_direction_sweep,
    "half_plane_stagger": sc_half_plane_stagger,
    "double_time_whip": sc_double_time_whip,
    "half_time_glide": sc_half_time_glide,
    "dotted_fan_burst": sc_dotted_fan_burst,
    "dot_matrix_sweep": sc_dot_matrix_sweep,
    "drop_hyper_strobe": sc_drop_hyper_strobe,
    "full_width_wall": sc_full_width_wall,
    "full_height_sheet": sc_full_height_sheet,
    "multi_plane_cross_sweep": sc_multi_plane_cross_sweep,
    "twin_scissor_sweep": sc_twin_scissor_sweep,
    "drop_x_blades": sc_drop_x_blades,
    "quadrant_mirror_sweep": sc_quadrant_mirror_sweep,
    "pendulum_beam_bar": sc_pendulum_beam_bar,
    "crisscross_ribbon_planes": sc_crisscross_ribbon_planes,
    "alternating_horizon_gates": sc_alternating_horizon_gates,
    "prism_dot_fan": sc_prism_dot_fan,
    "hyper_bar_ricochet": sc_hyper_bar_ricochet,
    "horizontal_plane_blast": sc_horizontal_plane_blast,
    "quad_corner_lasers": sc_quad_corner_lasers,
    "strobe_tunnel_ribs": sc_strobe_tunnel_ribs,
    "build_converge_sweep": sc_build_converge_sweep,
    "build_rising_sheet": sc_build_rising_sheet,
    "build_prism_tighten": sc_build_prism_tighten,
    "build_accelerating_bars": sc_build_accelerating_bars,
    "build_dot_convergence": sc_build_dot_convergence,
    "build_split_pinch": sc_build_split_pinch,
    # Legacy aliases
    "tunnel": sc_concentric_tunnel,
    "tunnel_fast": sc_strobe_tunnel_ribs,
    "beam_fan": sc_dotted_fan_burst,
    "starburst": sc_orbital_plane_sweep,
    "starburst_fast": sc_hot_beam_matrix,
    "lissajous": sc_rotating_laser_cone,
    "scope": sc_dot_matrix_sweep,
    "spiral": sc_rotating_laser_cone,
    "spiral_slow": sc_velvet_sky_sweep,
    "grid": sc_hot_beam_matrix,
    "drop_blast": sc_horizontal_plane_blast,
    "breath": sc_breathe_horizon_sky,
    "riser": sc_build_prism_tighten,
    "ring_pulse": sc_concentric_tunnel,
    "sky_fan": sc_morphing_plane_sky,
    "sky_cross": sc_dual_horizon_mirage,
    "scan_sweep": sc_silk_wave_sky,
    "build_fan": sc_build_converge_sweep,
    "build_scan": sc_build_rising_sheet,
    "strobe_poly": sc_drop_hyper_strobe,
    "strobe_beams": sc_hyper_bar_ricochet,
    "strobe_burst": sc_hot_beam_matrix,
}

# Quick-Cue live performance pads (inspired by Pangolin QuickShow cue grid)
QUICK_CUES: list[tuple[str, str]] = [
    ("liquid_sky_drift", "Liquid Sky"),
    ("dotted_fan_burst", "Beam Fan"),
    ("concentric_tunnel", "Tunnel"),
    ("hot_beam_matrix", "Hot Beams"),
    ("rotating_laser_cone", "Laser Cone"),
    ("multi_plane_cross_sweep", "Cross Planes"),
    ("silk_wave_sky", "Wave Sheet"),
    ("alternating_horizon_gates", "Horizon Gates"),
]

POOLS = {
    MAIN: [
        "concentric_tunnel",
        "rotating_laser_cone",
        "split_direction_sweep",
        "half_plane_stagger",
        "dotted_fan_burst",
        "dot_matrix_sweep",
        "half_time_glide",
        "full_width_wall",
        "full_height_sheet",
        "orbital_plane_sweep",
        "multi_plane_cross_sweep",
        "twin_scissor_sweep",
        "quadrant_mirror_sweep",
        "pendulum_beam_bar",
        "crisscross_ribbon_planes",
        "alternating_horizon_gates",
        "prism_dot_fan",
    ],
    DROP: [
        "drop_hyper_strobe",
        "concentric_tunnel",
        "hot_beam_matrix",
        "double_time_whip",
        "hyper_bar_ricochet",
        "horizontal_plane_blast",
        "quad_corner_lasers",
        "strobe_tunnel_ribs",
        "split_direction_sweep",
        "orbital_plane_sweep",
        "half_plane_stagger",
        "dot_matrix_sweep",
        "multi_plane_cross_sweep",
        "drop_x_blades",
    ],
    BREAKDOWN: [
        "liquid_sky_drift",
        "morphing_plane_sky",
        "breathe_horizon_sky",
        "velvet_sky_sweep",
        "silk_wave_sky",
        "dual_horizon_mirage",
        "floating_dot_constellation",
        "slow_v_canopy_sky",
        "aurora_sky_curtain",
    ],
    BUILD: [
        "build_prism_tighten",
        "build_rising_sheet",
        "build_converge_sweep",
        "build_accelerating_bars",
        "build_dot_convergence",
        "build_split_pinch",
    ],
    SILENCE: [],
}

MODIFIER_POOLS = {
    MAIN: ["beat_rotate", "stereo_pan_shift", "kick_zoom_pump", "counter_sweep_plane", "hat_packet_chop"],
    DROP: ["fast_strobe_16th", "beat_rotate", "counter_sweep_plane", "hat_packet_chop"],
    BREAKDOWN: ["stereo_pan_shift"],
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
    """High-FPS FX engine producing coherent aerial beam choreography with optimized galvo order."""

    def __init__(self, seed: int = 1, force_scene: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene, self.prev_scene = None, None, None
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

        self._melodic_avg: float = 0.0
        self._prev_melodic: float = 0.0
        self._synth_accent_age: float = 99.0
        self._synth_accent_kind: int = 0

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
        chance = 0.55 if self.section == DROP else (0.42 if self.section == MAIN else 0.25)
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
        use_color_swap = float(self.rng.random()) < 0.50

        if use_color_swap:
            self.swap_mode = "color"
            step = int(self.rng.integers(1, len(PALETTES)))
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

    def update(self, f: Features):
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
            base_hue = 0.0

        dbt = bt - self._last_bt
        self._last_bt = bt
        if 0 < dbt < 1:
            self.rt += dbt * (1 + 4 * f.build if f.section == BUILD else 1)
            if self._morph_left > 0.0:
                self._morph_left = max(0.0, self._morph_left - dbt)
            if self._slot_morph_left > 0.0:
                self._slot_morph_left = max(0.0, self._slot_morph_left - dbt)
            if self._drop_impact_left > 0.0:
                self._drop_impact_left = max(0.0, self._drop_impact_left - dbt)
            self._synth_accent_age += dbt

        melodic = 0.65 * f.mid + 0.35 * f.high + 0.25 * getattr(f, "snare", 0.0)
        self._melodic_avg = 0.92 * self._melodic_avg + 0.08 * melodic
        if (
            f.section == BREAKDOWN
            and self._synth_accent_age > 1.25
            and melodic > max(self._melodic_avg * 1.18, 0.28)
            and melodic > self._prev_melodic + 0.03
        ):
            self._synth_accent_age = 0.0
            self._synth_accent_kind += 1
        self._prev_melodic = melodic

        if not self.scene or (f.section == SILENCE and not self.force):
            return [], None

        if f.section == DROP and self._drop_impact_left > 0.0 and not self.force:
            c_drop = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
            progress = 1.0 - (self._drop_impact_left / DROP_IMPACT_BEATS)
            paths = _drop_impact_effect(c_drop, progress, self._drop_impact_variant)
            if progress > 0.78:
                blend = (progress - 0.78) / 0.22
                paths = _morph_paths(paths, SCENES[self.scene](c_drop), blend)
            return _optimize_path_order(paths), DROP_IMPACT_NAMES[self._drop_impact_variant]

        elapsed_beats = f.beat_count - self._swap_start_beat
        swap_active = (
            self.swap_mode is not None
            and self.swap_seq_name in SWAP_SEQUENCES
            and 0 <= elapsed_beats < self.swap_beats_total
        )

        active_hue = base_hue
        if swap_active:
            seq = SWAP_SEQUENCES[self.swap_seq_name]
            slot = seq[elapsed_beats % len(seq)]
            if slot != self._active_slot:
                self._active_slot = slot
                self._slot_morph_left = 0.0 if self.swap_instant else SWAP_MORPH_BEATS

            if self.swap_mode == "color":
                active_hue = base_hue if slot == 1 else (base_hue + self.swap_hue_offset)
                c = Ctx(f, bt, active_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
                paths = SCENES[self.scene](c)
                if not self.swap_instant and self._slot_morph_left > 0.0:
                    prev_hue = (base_hue + self.swap_hue_offset) if slot == 1 else base_hue
                    c_prev = Ctx(f, bt, prev_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
                    alpha = 1.0 - (self._slot_morph_left / SWAP_MORPH_BEATS)
                    paths = _morph_paths(SCENES[self.scene](c_prev), paths, alpha)
            else:
                c = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
                primary_name = self.scene if slot == 1 else self.swap_scene
                other_name = self.swap_scene if slot == 1 else self.scene
                paths = SCENES[primary_name](c)
                if not self.swap_instant and self._slot_morph_left > 0.0 and other_name in SCENES:
                    alpha = 1.0 - (self._slot_morph_left / SWAP_MORPH_BEATS)
                    paths = _morph_paths(SCENES[other_name](c), paths, alpha)
        else:
            c = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
            paths = SCENES[self.scene](c)

        if self._morph_left > 0.0 and self.prev_scene in SCENES:
            alpha = 1.0 - (self._morph_left / MORPH_BEATS)
            prev_paths = SCENES[self.prev_scene](c)
            paths = _morph_paths(prev_paths, paths, alpha)

        paths = _apply_modifier(paths, self.modifier, c)

        if f.section == BREAKDOWN and self._synth_accent_age < 1.0 and not self.force:
            paths = _apply_synth_accent(paths, self._synth_accent_age, self._synth_accent_kind, c)

        if swap_active and self.swap_mode == "effect":
            mode_tag = "cut" if self.swap_instant else "morph"
            label = f"{self.scene}<>{self.swap_scene} [{mode_tag} {self.swap_seq_name}]"
        elif swap_active and self.swap_mode == "color":
            mode_tag = "cut" if self.swap_instant else "morph"
            label = f"{self.scene} [color-{mode_tag} {self.swap_seq_name}]"
        else:
            label = self.scene

        if self.modifier:
            label = f"{label}+{self.modifier}"
        return _optimize_path_order(paths), label