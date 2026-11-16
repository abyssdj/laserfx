"""Laser effects: coherent, sharp aerial beam geometry, clean single-vector planes,
smooth RGB-safe morphing, and high-FPS sub-frame beat interpolation.

Coordinates are in [-1, 1]; colours are RGB in 0..1.
Designed for a single output feeding twin projectors where Projector 2 is X-mirrored (x -> -x).
"""
from __future__ import annotations

import colorsys
import math
import time
from dataclasses import dataclass

import numpy as np

from .audio import BREAKDOWN, BUILD, DROP, MAIN, SILENCE, Features

TAU = 2 * math.pi
WHITE = np.array([1.0, 1.0, 1.0], dtype=float)
LIM = 0.94  # uniform scan boundary so all corners and sweeps share clean margins


@dataclass
class Path:
    pts: np.ndarray            # (V, 2)
    rgb: np.ndarray            # (3,) or (V, 3)
    dwell: int = 0


def hsv(h, s=1.0, v=1.0):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)))


def line(a, b, rgb, dwell=0):
    return Path(np.array([a, b], float), rgb, dwell)


def hot_dot(pt, rgb=WHITE, dwell=1):
    """Creates a crisp stationary aerial laser pip."""
    x = float(np.clip(pt[0], -LIM, LIM))
    y = float(np.clip(pt[1], -LIM, LIM))
    x2 = min(x + 0.008, LIM) if x < (LIM - 0.02) else x - 0.008
    return Path(np.array([[x, y], [x2, y]], float), rgb, dwell)


def dotted_line(a, b, rgb, dots=5, dot_len=0.04, phase=0.0):
    """Renders a clean, well-spaced series of crisp aerial laser pips along segment a -> b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vec = b - a
    total = float(np.hypot(vec[0], vec[1]))
    if total < 1e-4:
        return []
    frac_len = min(dot_len / total, 0.42 / max(dots, 1))
    out = []
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


def dashed_beam(a, b, rgb, segments=3, duty=0.64, phase=0.0):
    """Cleanly separated dashed aerial beam segments."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = []
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


def ray(ox, oy, ang, length=2.6, bound=LIM):
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
    """Full-span laser plane passing through (cx, cy) perpendicular to `angle`, clipped to [-span, span]."""
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


def _dedupe_exact_lines(paths: list[Path], tol: float = 0.03) -> list[Path]:
    """Removes only true duplicate overlapping 2-point lines while preserving crossing beams and polygons."""
    kept: list[Path] = []
    sigs: list[tuple[np.ndarray, np.ndarray]] = []
    for p in paths:
        if len(p.pts) != 2:
            kept.append(p)
            continue
        a, b = p.pts[0], p.pts[1]
        if float(np.hypot(b[0] - a[0], b[1] - a[1])) < 0.02:
            kept.append(p)
            continue
        dup = False
        for ka, kb in sigs:
            d_same = max(float(np.hypot(a[0] - ka[0], a[1] - ka[1])), float(np.hypot(b[0] - kb[0], b[1] - kb[1])))
            d_flip = max(float(np.hypot(a[0] - kb[0], a[1] - kb[1])), float(np.hypot(b[0] - ka[0], b[1] - ka[1])))
            if min(d_same, d_flip) < tol:
                dup = True
                break
        if not dup:
            kept.append(p)
            sigs.append((a, b))
    return kept


def _zig(i: int, a, b):
    return (a, b) if i % 2 == 0 else (b, a)


# ---------------------------------------------------------------- smooth musical motion helpers
def smooth_wave(bt: float, period_beats: float) -> float:
    return -math.cos(bt * TAU / period_beats)


def beat_sweep(bt: float, beats_per_pass: float = 1.0, alternate: bool = True) -> float:
    if alternate:
        return 0.5 - 0.5 * math.cos(math.pi * bt / beats_per_pass)
    idx = int(math.floor(bt / beats_per_pass))
    frac = (bt / beats_per_pass) - idx
    return 0.5 - 0.5 * math.cos(math.pi * frac)


def beat_hash(idx: int, salt: int = 0) -> float:
    v = math.sin(idx * 127.1 + salt * 311.7) * 43758.5453123
    return v - math.floor(v)


def beat_pump(c: Ctx) -> float:
    return min(max(0.55 * c.pulse + 0.45 * c.kick, 0.0), 1.0)


def snare_swell(c: Ctx) -> float:
    return min(max(getattr(c.f, "snare", 0.0), 0.0), 1.0)


def strobe_gate(c: Ctx, div: float = 4.0, duty: float = 0.48) -> bool:
    return (not c.strobe) or ((c.bt * div) % 1.0) < duty


PALETTES = [
    (0.54, 0.83, 0.85),   # Ice Cyan + Electric Magenta
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
    def energy(self):
        return min(max(self.f.level, 0.0), 1.0)

    @property
    def pulse(self):
        return self.f.beat_pulse

    @property
    def kick(self):
        return self.f.kick

    @property
    def hi(self):
        return min(max(getattr(self.f, "hi_flux", 0.0), 0.0), 1.0)

    def count(self, lo, hi, x=1.0):
        return int(round(lo + (hi - lo) * min(max(self.detail * x, 0.0), 1.0)))

    def palette(self, idx: int = 0, v: float = 1.0, white_boost: float = 0.0):
        p_idx = int(round(abs(self.hue * 10))) % len(PALETTES)
        h_pri, h_acc, s_acc = PALETTES[p_idx]
        if idx % 2 == 0:
            return hsv(h_pri, max(1.0 - white_boost, 0.0), v)
        return hsv(h_acc, max(s_acc - white_boost, 0.0), v)


# ======================================================================== DROP OPENING EXPLOSIONS (4 VARIATIONS)

DROP_IMPACT_NAMES = [
    "drop_white_starburst",
    "drop_white_plane_burst",
    "drop_white_dot_explosion",
    "drop_white_sweep_bars",
]


def _drop_impact_effect(c: Ctx, progress: float, variant: int) -> list[Path]:
    fade = max(0.0, 1.0 - 0.30 * progress)
    col_main = WHITE * fade
    col_soft = WHITE * (0.72 * fade)
    eased = math.sin(min(progress, 1.0) * math.pi * 0.5)
    out = []

    if variant == 0:
        ox, oy = -0.32, -0.10
        n_rays = c.count(6, 8)
        spin = 0.35 * progress
        reach = 0.50 + 1.80 * min(progress * 1.4, 1.0)
        for i in range(n_rays):
            ang = spin + (i / max(n_rays, 1)) * TAU
            p0, p1 = ray(ox, oy, ang, reach)
            if progress < 0.55:
                out.append(line(*_zig(i, p0, p1), col_main, dwell=0))
            else:
                out.extend(dotted_line(p0, p1, col_main, dots=4, dot_len=0.045, phase=progress * 2.0))

    elif variant == 1:
        pairs = c.count(2, 3)
        for p in range(pairs):
            frac = (p + 0.5) / pairs
            y_off = 0.16 + 0.72 * eased * frac
            slant = 0.14 * (1.0 - eased) * (1.0 if p % 2 == 0 else -1.0)
            for sign in (-1.0, 1.0):
                y0 = float(np.clip(sign * y_off - slant, -LIM, LIM))
                y1 = float(np.clip(sign * y_off + slant, -LIM, LIM))
                out.append(line((-LIM, y0), (LIM, y1), col_main, dwell=0))

    elif variant == 2:
        ox, oy = -0.25, 0.05
        n_arms = c.count(6, 8)
        dots_per_arm = c.count(3, 4)
        for arm in range(n_arms):
            base_a = (arm / n_arms) * TAU + progress * 0.9
            for k in range(dots_per_arm):
                r = (0.18 + 0.86 * eased) * ((k + 1) / dots_per_arm)
                a = base_a + k * 0.18 * progress
                px = float(np.clip(ox + r * math.cos(a), -LIM, LIM))
                py = float(np.clip(oy + r * math.sin(a), -LIM, LIM))
                out.append(hot_dot((px, py), col_main if k == dots_per_arm - 1 else col_soft, dwell=1))

    else:
        u = beat_sweep(progress * 2.0, beats_per_pass=1.0, alternate=True)
        x_a = -0.88 + 1.76 * u
        x_b = 0.88 - 1.76 * u
        y_c = -0.82 + 1.64 * eased
        out.append(line((x_a, -LIM), (x_a, LIM), col_main, dwell=0))
        out.append(line((x_b, LIM), (x_b, -LIM), col_main, dwell=0))
        out.append(line((-LIM, y_c), (LIM, y_c), col_soft, dwell=0))
        out.append(line((LIM, -y_c), (-LIM, -y_c), col_soft, dwell=0))

    return out


# ======================================================================== ATMOSPHERIC BREAKDOWN SCENES (LIQUID SKY)

def sc_liquid_sky_drift(c: Ctx):
    planes = c.count(1, 2)
    tilt = 0.22 * smooth_wave(c.bt, 16.0)
    y_base = 0.24 * smooth_wave(c.bt, 32.0)

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * 0.36
        y0 = float(np.clip(y_base + gap - tilt, -LIM, LIM))
        y1 = float(np.clip(y_base + gap + tilt, -LIM, LIM))
        a, b = _zig(p, (-LIM, y0), (LIM, y1))
        out.append(line(a, b, c.palette(p, v=0.82), dwell=0))
    return out


def sc_morphing_plane_sky(c: Ctx):
    morph = 0.5 + 0.5 * smooth_wave(c.bt, 16.0)
    y_center = 0.22 + 0.16 * smooth_wave(c.bt, 24.0)
    tilt = 0.36 * morph

    p0_a = (-LIM, float(np.clip(y_center - tilt, -LIM, LIM)))
    p0_b = (LIM, float(np.clip(y_center + tilt, -LIM, LIM)))
    p1_a = (-LIM, float(np.clip(y_center + tilt * 0.75 - 0.34, -LIM, LIM)))
    p1_b = (LIM, float(np.clip(y_center - tilt * 0.75 - 0.34, -LIM, LIM)))

    out = [line(p0_a, p0_b, c.palette(0, v=0.84), dwell=0)]
    if c.detail >= 0.35:
        out.append(line(p1_b, p1_a, c.palette(1, v=0.78), dwell=0))
    return out


def sc_breathe_horizon_sky(c: Ctx):
    spread = 0.18 + 0.28 * (0.5 + 0.5 * smooth_wave(c.bt, 16.0))
    y_mid = 0.18 + 0.14 * smooth_wave(c.bt, 32.0)
    rock = 0.14 * smooth_wave(c.bt, 12.0)

    out = []
    for idx, sign in enumerate((-1.0, 1.0)):
        if idx == 1 and c.detail < 0.3:
            break
        y0 = float(np.clip(y_mid + sign * spread - rock, -LIM, LIM))
        y1 = float(np.clip(y_mid + sign * spread + rock, -LIM, LIM))
        a, b = _zig(idx, (-LIM, y0), (LIM, y1))
        out.append(line(a, b, c.palette(idx, v=0.80), dwell=0))
    return out


def sc_velvet_sky_sweep(c: Ctx):
    u = beat_sweep(c.bt, beats_per_pass=8.0, alternate=True)
    x_center = -0.74 + 1.48 * u
    blades = c.count(1, 2)
    width = 0.34 + 0.10 * smooth_wave(c.bt, 16.0)

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * width
        x = float(np.clip(x_center + off, -LIM, LIM))
        a, b = _zig(i, (x, -LIM), (x, LIM))
        out.append(line(a, b, c.palette(i % 2, v=0.80), dwell=0))
    return out


def sc_silk_wave_sky(c: Ctx):
    x = np.linspace(-LIM, LIM, 24)
    ph = c.bt * TAU / 16.0
    y = np.clip(0.28 + 0.16 * smooth_wave(c.bt, 24.0) + 0.18 * np.sin(1.5 * math.pi * x + ph), -LIM, LIM)
    pts = np.stack([x, y], 1)
    return [Path(pts, c.palette(0, v=0.82), dwell=0)]


def sc_dual_horizon_mirage(c: Ctx):
    ang = 0.30 * smooth_wave(c.bt, 20.0)
    y_base = 0.24 + 0.12 * smooth_wave(c.bt, 32.0)
    return [
        line((-LIM, y_base - ang), (LIM, y_base + ang), c.palette(0, v=0.80), dwell=0),
        line((LIM, y_base - ang * 0.6 - 0.34), (-LIM, y_base + ang * 0.6 - 0.34), c.palette(1, v=0.76), dwell=0),
    ]


def sc_floating_dot_constellation(c: Ctx):
    y_sky = 0.08 + 0.14 * smooth_wave(c.bt, 24.0)
    out = [line((-LIM, y_sky), (LIM, y_sky), c.palette(0, v=0.74), dwell=0)]
    n_stars = c.count(5, 7)
    for s in range(n_stars):
        u = (s + 0.5) / n_stars
        x = -0.84 + 1.68 * u + 0.06 * math.sin(c.bt * TAU / 16.0 + s)
        y = 0.44 + 0.28 * math.sin(c.bt * TAU / 20.0 + s * 1.7)
        col = WHITE * 0.84 if s % 2 == 0 else c.palette(1, v=0.84)
        out.append(hot_dot((x, y), col, dwell=1))
    return out


def sc_slow_v_canopy_sky(c: Ctx):
    y_apex = 0.10 + 0.18 * smooth_wave(c.bt, 24.0)
    wing_lift = 0.30 + 0.24 * smooth_wave(c.bt, 16.0)
    y_wing = float(np.clip(y_apex + wing_lift, -LIM, LIM))
    # Continuous V-polyline for crisp center vertex without blanking jump
    pts = np.array([[-LIM, y_wing], [0.0, y_apex], [LIM, y_wing]], float)
    return [Path(pts, c.palette(0, v=0.82), dwell=0)]


def sc_aurora_sky_curtain(c: Ctx):
    x = np.linspace(-LIM, LIM, 24)
    out = []
    for layer in range(c.count(1, 2)):
        ph = c.bt * TAU / 18.0 + layer * 2.1
        y = 0.34 + 0.12 * smooth_wave(c.bt, 28.0) + 0.14 * np.sin(1.4 * math.pi * x + ph) - layer * 0.34
        pts = np.stack([x, np.clip(y, -LIM, LIM)], 1)
        if layer == 1:
            pts = pts[::-1]
        out.append(Path(pts, c.palette(layer, v=0.80), dwell=0))
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
        tilt = 0.14 * (1.0 if kind % 4 == 0 else -1.0)
        return paths + [line((-LIM, y - tilt), (LIM, y + tilt), col, dwell=0)]
    else:
        ox, oy = -0.38, -0.34
        r = 0.14 + 0.28 * age_beats
        burst = []
        for k in range(4):
            a = (k / 4.0) * TAU + age_beats * 0.8
            burst.append(hot_dot((ox + r * math.cos(a), oy + r * 0.7 * math.sin(a)), col, dwell=1))
        return paths + burst


# ======================================================================== MAIN & DROP SCENES (COHERENT, CRISP GEOMETRY)

def sc_split_direction_sweep(c: Ctx):
    pump = beat_pump(c)
    u_top = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    u_bot = 1.0 - u_top

    x_top = -0.84 + 1.68 * u_top
    x_bot = -0.84 + 1.68 * u_bot
    blades = c.count(2, 2)
    gap = 0.18 + 0.06 * pump

    out = []
    idx = 0
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        xt = float(np.clip(x_top + off, -LIM, LIM))
        xb = float(np.clip(x_bot + off, -LIM, LIM))
        out.append(line(*_zig(idx, (xt, 0.10), (xt, LIM)), c.palette(0, white_boost=0.70 * pump if i == 0 else 0.0), dwell=0))
        idx += 1
        out.append(line(*_zig(idx, (xb, -LIM), (xb, -0.10)), c.palette(1, white_boost=0.70 * pump if i == 0 else 0.0), dwell=0))
        idx += 1
    return out


def sc_half_plane_stagger(c: Ctx):
    pump = beat_pump(c)
    u1 = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    u2 = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)

    y_left = -0.84 + 1.68 * u1
    y_right = -0.84 + 1.68 * (1.0 - u2)
    x_top = -0.86 + 1.72 * u2
    x_bot = -0.86 + 1.72 * (1.0 - u1)

    return [
        line((-LIM, y_left), (-0.10, y_left + 0.10), c.palette(0, white_boost=0.65 * pump), dwell=0),
        line((LIM, y_right), (0.10, y_right - 0.10), c.palette(1, white_boost=0.65 * pump), dwell=0),
        line((x_top, 0.10), (x_top, LIM), c.palette(1, white_boost=0.50 * pump), dwell=0),
        line((x_bot, -LIM), (x_bot, -0.10), c.palette(0, white_boost=0.50 * pump), dwell=0),
    ]


def sc_double_time_whip(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    x_center = -0.82 + 1.64 * u
    slant = 0.22 * math.sin(c.bt * TAU)
    blades = c.count(2, 2)
    gap = 0.20 + 0.06 * pump

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        x0 = float(np.clip(x_center + off - slant, -LIM, LIM))
        x1 = float(np.clip(x_center + off + slant, -LIM, LIM))
        col = c.palette(i % 2, v=1.0, white_boost=0.80 * pump if i == 0 else 0.0)
        out.append(line(*_zig(i, (x0, -LIM), (x1, LIM)), col, dwell=0))
    return out


def sc_half_time_glide(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)
    y_center = -0.76 + 1.52 * u
    slant = 0.20 * smooth_wave(c.bt, 8.0)
    blades = c.count(2, 3)
    gap = 0.20 + 0.08 * pump

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        y0 = float(np.clip(y_center + off - slant, -LIM, LIM))
        y1 = float(np.clip(y_center + off + slant, -LIM, LIM))
        col = c.palette(i % 2, v=1.0, white_boost=0.70 * pump if i == 1 else 0.0)
        out.append(line(*_zig(i, (-LIM, y0), (LIM, y1)), col, dwell=0))
    return out


def sc_dotted_fan_burst(c: Ctx):
    pump = beat_pump(c)
    n = c.count(5, 7, max(c.energy, 0.45))
    aim = 0.26 + 0.42 * smooth_wave(c.bt, 8.0)
    spread = 0.44 + 0.22 * pump
    ox = -0.58

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        reach = 1.70 + 0.30 * math.cos(u * math.pi) + 0.20 * pump
        p0, p1 = ray(ox, -LIM, ang, reach)
        col = c.palette(i % 2, v=0.95)
        if i % 2 == 1:
            out.extend(dotted_line(p0, p1, col, dots=4, dot_len=0.045, phase=c.bt * 2.0))
        else:
            out.append(line(*_zig(i, p0, p1), col, dwell=0))
    return out


def sc_dot_matrix_sweep(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_center = -0.74 + 1.48 * u
    rows = c.count(2, 3)
    dots_per_row = c.count(5, 7)
    slant = 0.16 * smooth_wave(c.bt, 4.0)

    out = []
    for r in range(rows):
        off = (r - (rows - 1) / 2.0) * (0.24 + 0.10 * pump)
        y0 = float(np.clip(y_center + off - slant, -LIM, LIM))
        y1 = float(np.clip(y_center + off + slant, -LIM, LIM))
        col = WHITE if (r == 0 and pump > 0.35) else c.palette(r % 2, v=1.0)
        out.extend(dotted_line((-LIM, y0), (LIM, y1), col, dots=dots_per_row, dot_len=0.04, phase=c.bt * 1.5 + r * 0.35))
    return out


def sc_drop_hyper_strobe(c: Ctx):
    div = 4.0 if (int(c.bt / 2) % 2 == 1) else 2.0
    if not strobe_gate(c, div=div, duty=0.48):
        return []

    pump = beat_pump(c)
    step = int(c.bt * div)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    out = []

    if step % 3 == 0:
        x = -0.82 + 1.64 * u
        out.append(line((x, -LIM), (x, LIM), WHITE, dwell=0))
        x2 = float(np.clip(x - 0.22 if x > -0.60 else x + 0.22, -LIM, LIM))
        out.append(line((x2, LIM), (x2, -LIM), c.palette(0), dwell=0))
    elif step % 3 == 1:
        y_top = 0.18 + 0.74 * u
        y_bot = -0.18 - 0.74 * u
        out.append(line((-LIM, y_top), (LIM, float(np.clip(y_top + 0.10, -LIM, LIM))), c.palette(1, white_boost=0.8 * pump), dwell=0))
        out.append(line((LIM, y_bot), (-LIM, float(np.clip(y_bot - 0.10, -LIM, LIM))), WHITE, dwell=0))
    else:
        n = c.count(5, 6)
        aim = 0.25 + 0.45 * u
        for i in range(n):
            frac = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(-0.62, -LIM, aim + frac * 0.95, 2.3)
            out.append(line(*_zig(i, p0, p1), c.palette(i % 2, white_boost=0.6), dwell=0))
    return out


def sc_full_width_wall(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    x_center = -0.78 + 1.56 * u
    blades = c.count(2, 2)
    thickness = 0.18 + 0.06 * pump
    tilt = 0.08 * math.sin(c.bt * math.pi)

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * thickness
        x0 = float(np.clip(x_center + off - tilt, -LIM, LIM))
        x1 = float(np.clip(x_center + off + tilt, -LIM, LIM))
        col = c.palette(i % 2, v=1.0, white_boost=0.75 * pump if i == 0 else 0.0)
        out.append(line(*_zig(i, (x0, -LIM), (x1, LIM)), col, dwell=0))
    return out


def sc_full_height_sheet(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_center = -0.78 + 1.56 * u
    slant = 0.16 * math.sin(c.bt * math.pi)
    blades = c.count(2, 2)
    gap = 0.20 + 0.06 * pump

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        y0 = float(np.clip(y_center + off - slant, -LIM, LIM))
        y1 = float(np.clip(y_center + off + slant, -LIM, LIM))
        col = c.palette(i % 2, v=1.0, white_boost=0.75 * pump if i == 0 else 0.0)
        out.append(line(*_zig(i, (-LIM, y0), (LIM, y1)), col, dwell=0))
    return out


def sc_random_dir_sweep(c: Ctx):
    pump = beat_pump(c)
    beat_idx = int(math.floor(c.bt))
    frac = c.bt - beat_idx
    u = 0.5 - 0.5 * math.cos(math.pi * frac)

    dir_idx = int(beat_hash(beat_idx, salt=11) * 8)
    angle = dir_idx * (TAU / 8.0)

    dist = -0.90 + 1.80 * u
    cx, cy = dist * math.cos(angle), dist * math.sin(angle)

    blades = c.count(2, 2)
    spacing = 0.20 + 0.06 * pump
    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * spacing
        px = cx + off * math.cos(angle)
        py = cy + off * math.sin(angle)
        seg = clipped_plane(px, py, angle, span=LIM)
        if seg is not None:
            col = c.palette(beat_idx % 2 if i == 0 else 1, v=1.0, white_boost=0.8 * pump if i == 0 else 0.0)
            out.append(line(*_zig(i, seg[0], seg[1]), col, dwell=0))
    return out


def sc_multi_plane_cross_sweep(c: Ctx):
    pump = beat_pump(c)
    n_planes = c.count(2, 2)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    v = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)

    out = []
    idx = 0
    for i in range(n_planes):
        phase_off = i / max(n_planes, 1)
        pos = ((u + phase_off) % 1.0)
        x = -0.88 + 1.76 * pos
        col = c.palette(0, v=1.0, white_boost=0.65 * pump if i == 0 else 0.0)
        out.append(line(*_zig(idx, (x, -LIM), (x, LIM)), col, dwell=0))
        idx += 1

    for j in range(n_planes):
        phase_off = j / max(n_planes, 1)
        pos = ((1.0 - v + phase_off) % 1.0)
        y = -0.88 + 1.76 * pos
        slant = 0.12 * (1.0 if j % 2 == 0 else -1.0)
        y0 = float(np.clip(y - slant, -LIM, LIM))
        y1 = float(np.clip(y + slant, -LIM, LIM))
        col = c.palette(1, v=1.0, white_boost=0.65 * pump if j == 0 else 0.0)
        out.append(line(*_zig(idx, (-LIM, y0), (LIM, y1)), col, dwell=0))
        idx += 1
    return out


def sc_random_multi_blade(c: Ctx):
    pump = beat_pump(c)
    beat_idx = int(math.floor(c.bt))
    frac = c.bt - beat_idx
    u = 0.5 - 0.5 * math.cos(math.pi * frac)

    n_blades = c.count(2, 3)
    out = []
    for b in range(n_blades):
        base_ang = beat_hash(beat_idx, salt=30) * TAU
        ang = base_ang + b * (TAU / max(n_blades, 1))
        flip = -1.0 if beat_hash(beat_idx, salt=90 + b * 13) > 0.5 else 1.0
        dist = flip * (-0.90 + 1.80 * u)
        cx, cy = dist * math.cos(ang), dist * math.sin(ang)
        seg = clipped_plane(cx, cy, ang, span=LIM)
        if seg is not None:
            col = c.palette(b % 2, v=1.0, white_boost=0.8 * pump if b == 0 else 0.0)
            out.append(line(*_zig(b, seg[0], seg[1]), col, dwell=0))
    return out


def sc_twin_scissor_sweep(c: Ctx):
    pump = beat_pump(c)
    snare = snare_swell(c)
    n = c.count(4, 5, max(c.energy, 0.45))
    swing = 0.44 * smooth_wave(c.bt, 8.0)

    out = []
    idx = 0
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = 0.34 + swing + u * (0.38 + 0.18 * pump)
        p0, p1 = ray(-0.78, -LIM, ang, 2.2)
        out.append(line(*_zig(idx, p0, p1), c.palette(0, v=1.0), dwell=0))
        idx += 1

    for i in range(max(3, n - 1)):
        u = (i / max(n - 2, 1)) - 0.5
        ang = math.pi - (0.36 - swing * 0.85 + u * (0.36 + 0.18 * snare))
        p0, p1 = ray(-0.28, LIM, ang, 2.2)
        out.append(line(*_zig(idx, p0, p1), c.palette(1, v=0.85 + 0.15 * snare), dwell=0))
        idx += 1
    return out


def sc_drop_x_blades(c: Ctx):
    pump = beat_pump(c)
    planes = c.count(2, 3)
    tilt = 0.54 * smooth_wave(c.bt, 4.0)
    y_sweep = 0.28 * smooth_wave(c.bt, 8.0)

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.24 + 0.14 * pump)
        y0 = float(np.clip(y_sweep + gap - tilt, -LIM, LIM))
        y1 = float(np.clip(y_sweep + gap + tilt, -LIM, LIM))
        col = c.palette(p % 2, v=1.0, white_boost=0.8 * pump if p == 0 else 0.0)
        out.append(line(*_zig(p, (-LIM, y0), (LIM, y1)), col, dwell=0))
    return out


def sc_quadrant_mirror_sweep(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    sweep_ang = 0.20 + 1.00 * u
    p0, p1 = ray(-LIM, -LIM, sweep_ang, 1.9 + 0.3 * pump)
    p2, p3 = ray(-LIM, LIM, math.pi - sweep_ang, 1.9 + 0.3 * pump)
    return [
        line(p0, p1, c.palette(0, white_boost=0.7 * pump), dwell=0),
        line(p3, p2, c.palette(1, white_boost=0.7 * pump), dwell=0),
    ]


def sc_pendulum_beam_bar(c: Ctx):
    pump = beat_pump(c)
    ang = 0.85 * smooth_wave(c.bt, 4.0)
    bars = c.count(2, 2)
    spacing = 0.24 + 0.08 * pump
    out = []
    for i in range(bars):
        off = (i - (bars - 1) / 2.0) * spacing
        seg = clipped_plane(-0.22 + off * math.cos(ang), off * math.sin(ang), ang, span=LIM)
        if seg is not None:
            col = c.palette(i % 2, white_boost=0.75 * pump if i == 0 else 0.0)
            out.append(line(*_zig(i, seg[0], seg[1]), col, dwell=0))
    return out


def sc_crisscross_ribbon_planes(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_a = -0.75 + 1.50 * u
    y_b = 0.75 - 1.50 * u
    return [
        line((-LIM, y_a), (LIM, float(np.clip(y_a + 0.40, -LIM, LIM))), c.palette(0, white_boost=0.6 * pump), dwell=0),
        line((LIM, y_b), (-LIM, float(np.clip(y_b + 0.40, -LIM, LIM))), c.palette(1, white_boost=0.6 * pump), dwell=0),
    ]


def sc_alternating_horizon_gates(c: Ctx):
    pump = beat_pump(c)
    gap = 0.20 + 0.70 * (0.5 - 0.5 * math.cos(math.pi * c.bt))
    out = [
        line((-LIM, gap), (LIM, gap), c.palette(0, white_boost=0.75 * pump), dwell=0),
        line((LIM, -gap), (-LIM, -gap), c.palette(1, white_boost=0.75 * pump), dwell=0),
    ]
    if c.detail >= 0.4:
        out.extend(dotted_line((-0.90, 0.0), (0.90, 0.0), WHITE * (0.55 + 0.45 * pump), dots=4, dot_len=0.035, phase=c.bt * 2.0))
    return out


def sc_prism_dot_fan(c: Ctx):
    pump = beat_pump(c)
    n = c.count(4, 6)
    aim = 0.30 + 0.42 * smooth_wave(c.bt, 6.0)
    spread = 0.44 + 0.22 * pump
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(-0.66, -LIM, ang, 2.25)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2), dwell=0))
    return out


def sc_hyper_bar_ricochet(c: Ctx):
    pump = beat_pump(c)
    u1 = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    u2 = beat_sweep(c.bt + 0.25, beats_per_pass=0.5, alternate=True)
    x1 = -0.88 + 1.76 * u1
    x2 = 0.88 - 1.76 * u2
    if abs(x1 - x2) < 0.20:
        x2 = float(np.clip(x1 + (0.22 if x1 < 0 else -0.22), -LIM, LIM))
    y1 = -0.88 + 1.76 * u2
    return [
        line((x1, -LIM), (x1, LIM), c.palette(0, white_boost=0.85 * pump), dwell=0),
        line((x2, LIM), (x2, -LIM), c.palette(1, white_boost=0.85 * pump), dwell=0),
        line((-LIM, y1), (LIM, y1), WHITE if pump > 0.45 else c.palette(0), dwell=0),
    ]


def sc_horizontal_plane_blast(c: Ctx):
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    eased = math.sin(frac * math.pi * 0.5)
    pairs = c.count(2, 2)
    out = []
    idx = 0
    for p in range(pairs):
        y = 0.18 + 0.72 * eased * ((p + 1) / pairs)
        col = c.palette(p % 2, white_boost=0.8 * pump if p == 0 else 0.0)
        out.append(line(*_zig(idx, (-LIM, y), (LIM, y)), col, dwell=0))
        idx += 1
        out.append(line(*_zig(idx, (-LIM, -y), (LIM, -y)), col, dwell=0))
        idx += 1
    return out


def sc_shatter_dot_storm(c: Ctx):
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    r = 0.20 + 0.74 * frac
    n_dots = c.count(10, 14)
    ox, oy = -0.35 * math.cos(c.bt * TAU / 8.0), 0.0
    out = []
    for i in range(n_dots):
        ang = c.bt * 1.4 + (i / n_dots) * TAU
        rad = r * (0.55 if i % 2 == 0 else 1.0)
        px = ox + rad * math.cos(ang)
        py = oy + rad * math.sin(ang)
        col = WHITE if (i % 3 == 0 or pump > 0.5) else c.palette(i % 2)
        out.append(hot_dot((px, py), col, dwell=1))
    return out


def sc_quad_corner_lasers(c: Ctx):
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    blades = c.count(2, 3)
    out = []
    for i in range(blades):
        ang = 0.22 + 0.85 * u + (i - 1) * 0.22
        p0, p1 = ray(-0.92, -0.92, ang, 2.5)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2, white_boost=0.8 * pump), dwell=0))
    return out


def sc_strobe_tunnel_ribs(c: Ctx):
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    layers = c.count(2, 3)
    cx = -0.22 * smooth_wave(c.bt, 8.0)
    out = []
    for l in range(layers):
        s = 0.26 + 0.65 * (((frac + l / layers) % 1.0))
        x0 = max(cx - s, -LIM)
        x1 = min(cx + s, LIM)
        y0 = -s * 0.85
        y1 = s * 0.85
        pts = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0],
        ], float)
        out.append(Path(pts, c.palette(l % 2, white_boost=0.7 * pump), dwell=0))
    return out


# ======================================================================== BUILD SECTION SCENES

def sc_build_converge_sweep(c: Ctx):
    b = c.f.build
    n = c.count(5, 7, max(b, 0.45))
    sweep = (0.48 * (1.0 - 0.62 * (b ** 2))) * math.sin(c.rt * TAU / 4.0)
    bias = 0.34 * (1.0 - 0.7 * b)
    spread = 0.54 * (1.0 - 0.74 * (b ** 2)) + 0.10
    ox = -0.60 * (1.0 - 0.65 * b)

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = bias + sweep + u * 2.0 * spread
        p0, p1 = ray(ox, -LIM, ang, 2.3)
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.85 * (b ** 2))
        out.append(line(*_zig(i, p0, p1), col, dwell=0))
    return out


def sc_build_rising_sheet(c: Ctx):
    b = c.f.build
    planes = c.count(2, 2)
    y_rise = -0.55 + 1.35 * b
    tilt = (0.30 * (1.0 - 0.5 * b)) * math.sin(c.rt * TAU / 4.0)

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.30 * (1.0 - 0.4 * b) + 0.14)
        y0 = float(np.clip(y_rise + gap - tilt, -LIM, LIM))
        y1 = float(np.clip(y_rise + gap + tilt, -LIM, LIM))
        col = c.palette(p % 2, v=0.8 + 0.2 * b, white_boost=0.8 * (b ** 2))
        out.append(line(*_zig(p, (-LIM, y0), (LIM, y1)), col, dwell=0))
    return out


def sc_build_prism_tighten(c: Ctx):
    b = c.f.build
    u = beat_sweep(c.rt, beats_per_pass=1.0, alternate=True)
    x_center = -0.84 + 1.68 * u
    blades = c.count(2, 3)
    width = 0.48 * (1.0 - 0.55 * b) + 0.18
    out = []
    for i in range(blades):
        off = (i / max(blades - 1, 1) - 0.5) * width
        x = float(np.clip(x_center + off, -LIM, LIM))
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.85 * (b ** 2))
        out.append(line(*_zig(i, (x, -LIM), (x, LIM)), col, dwell=0))
    return out


def sc_build_accelerating_bars(c: Ctx):
    b = c.f.build
    u = beat_sweep(c.rt, beats_per_pass=0.5, alternate=True)
    span = 0.88 * (1.0 - 0.65 * b) + 0.14
    x = -span + 2.0 * span * u
    col = c.palette(0, v=0.8 + 0.2 * b, white_boost=0.9 * (b ** 2))
    return [
        line((x, -LIM), (x, LIM), col, dwell=0),
        line((-x, LIM), (-x, -LIM), WHITE * (0.45 + 0.55 * b), dwell=0),
    ]


def sc_build_dot_convergence(c: Ctx):
    b = c.f.build
    n_dots = c.count(8, 12)
    r = 0.88 * (1.0 - 0.74 * b) + 0.12
    ox = -0.32 * (1.0 - b)
    out = []
    for i in range(n_dots):
        ang = c.rt * 2.2 + (i / n_dots) * TAU
        px = ox + r * math.cos(ang)
        py = r * math.sin(ang)
        col = WHITE if (b > 0.65 or i % 2 == 0) else c.palette(i % 2)
        out.append(hot_dot((px, py), col, dwell=1))
    return out


def sc_build_split_pinch(c: Ctx):
    b = c.f.build
    y_gap = 0.86 * (1.0 - 0.80 * b) + 0.10
    wobble = 0.16 * (1.0 - b) * math.sin(c.rt * TAU / 2.0)
    col = c.palette(1, v=0.8 + 0.2 * b, white_boost=0.85 * (b ** 2))
    return [
        line((-LIM, y_gap - wobble), (LIM, y_gap + wobble), col, dwell=0),
        line((LIM, -y_gap + wobble), (-LIM, -y_gap - wobble), col, dwell=0),
    ]


# ======================================================================== SECONDARY OVERLAY LAYER

def _apply_modifier(paths: list[Path], mod: str, c: Ctx) -> list[Path]:
    if not mod or not paths:
        return paths

    if mod == "fast_strobe_16th":
        div = 4.0 if (int(c.bt) % 2 == 1) else 2.0
        if not strobe_gate(c, div=div, duty=0.5):
            return []
        out = []
        for p in paths:
            rgb = np.clip(p.rgb * 0.5 + WHITE * 0.5, 0.0, 1.0)
            out.append(Path(p.pts, rgb, p.dwell))
        return out

    if mod == "white_tip_dots":
        dots = []
        for p in paths:
            if len(p.pts) == 2 and float(np.hypot(*(p.pts[1] - p.pts[0]))) > 0.05:
                dots.append(hot_dot(p.pts[1], WHITE, dwell=1))
        return paths + dots

    if mod == "beat_rotate":
        ang = 0.36 * smooth_wave(c.bt, 4.0)
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for p in paths:
            x, y = p.pts[:, 0], p.pts[:, 1]
            nx = np.clip(x * ca - y * sa, -LIM, LIM)
            ny = np.clip(x * sa + y * ca, -LIM, LIM)
            out.append(Path(np.stack([nx, ny], 1), p.rgb, p.dwell))
        return out

    if mod == "stereo_pan_shift":
        dx = 0.20 * smooth_wave(c.bt, 8.0)
        out = []
        for p in paths:
            pts = p.pts.copy()
            pts[:, 0] = np.clip(pts[:, 0] + dx, -LIM, LIM)
            out.append(Path(pts, p.rgb, p.dwell))
        return out

    if mod == "kick_zoom_pump":
        pump = beat_pump(c)
        scale = 0.86 + 0.14 * pump
        out = []
        for p in paths:
            pts = np.clip(p.pts * scale, -LIM, LIM)
            out.append(Path(pts, p.rgb, p.dwell))
        return out

    if mod == "counter_sweep_plane":
        u = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)
        y = -0.86 + 1.72 * u
        extra = [line((-LIM, y), (LIM, y), c.palette(1, v=0.9, white_boost=0.5 * beat_pump(c)), dwell=0)]
        return paths + extra

    if mod == "hat_packet_chop":
        if c.hi < 0.35:
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
    return np.stack([x, y], 1)


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
    morphed = []
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
    "random_dir_sweep": sc_random_dir_sweep,
    "multi_plane_cross_sweep": sc_multi_plane_cross_sweep,
    "random_multi_blade": sc_random_multi_blade,
    "twin_scissor_sweep": sc_twin_scissor_sweep,
    "drop_x_blades": sc_drop_x_blades,
    "quadrant_mirror_sweep": sc_quadrant_mirror_sweep,
    "pendulum_beam_bar": sc_pendulum_beam_bar,
    "crisscross_ribbon_planes": sc_crisscross_ribbon_planes,
    "alternating_horizon_gates": sc_alternating_horizon_gates,
    "prism_dot_fan": sc_prism_dot_fan,
    "hyper_bar_ricochet": sc_hyper_bar_ricochet,
    "horizontal_plane_blast": sc_horizontal_plane_blast,
    "shatter_dot_storm": sc_shatter_dot_storm,
    "quad_corner_lasers": sc_quad_corner_lasers,
    "strobe_tunnel_ribs": sc_strobe_tunnel_ribs,
    "build_converge_sweep": sc_build_converge_sweep,
    "build_rising_sheet": sc_build_rising_sheet,
    "build_prism_tighten": sc_build_prism_tighten,
    "build_accelerating_bars": sc_build_accelerating_bars,
    "build_dot_convergence": sc_build_dot_convergence,
    "build_split_pinch": sc_build_split_pinch,
    # Legacy aliases
    "tunnel": sc_strobe_tunnel_ribs,
    "tunnel_fast": sc_double_time_whip,
    "beam_fan": sc_dotted_fan_burst,
    "starburst": sc_random_dir_sweep,
    "starburst_fast": sc_shatter_dot_storm,
    "lissajous": sc_split_direction_sweep,
    "scope": sc_dot_matrix_sweep,
    "spiral": sc_half_plane_stagger,
    "spiral_slow": sc_velvet_sky_sweep,
    "grid": sc_half_plane_stagger,
    "drop_blast": sc_horizontal_plane_blast,
    "breath": sc_breathe_horizon_sky,
    "riser": sc_build_prism_tighten,
    "ring_pulse": sc_full_width_wall,
    "sky_fan": sc_morphing_plane_sky,
    "sky_cross": sc_dual_horizon_mirage,
    "scan_sweep": sc_silk_wave_sky,
    "build_fan": sc_build_converge_sweep,
    "build_scan": sc_build_rising_sheet,
    "strobe_poly": sc_drop_hyper_strobe,
    "strobe_beams": sc_hyper_bar_ricochet,
    "strobe_burst": sc_shatter_dot_storm,
}

POOLS = {
    MAIN: [
        "split_direction_sweep",
        "half_plane_stagger",
        "dotted_fan_burst",
        "dot_matrix_sweep",
        "half_time_glide",
        "full_width_wall",
        "full_height_sheet",
        "random_dir_sweep",
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
        "double_time_whip",
        "hyper_bar_ricochet",
        "horizontal_plane_blast",
        "shatter_dot_storm",
        "quad_corner_lasers",
        "strobe_tunnel_ribs",
        "split_direction_sweep",
        "random_multi_blade",
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
    MAIN: ["white_tip_dots", "beat_rotate", "stereo_pan_shift", "kick_zoom_pump", "counter_sweep_plane", "hat_packet_chop"],
    DROP: ["fast_strobe_16th", "white_tip_dots", "beat_rotate", "counter_sweep_plane", "hat_packet_chop"],
    BREAKDOWN: ["stereo_pan_shift"],
    BUILD: ["white_tip_dots", "kick_zoom_pump"],
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
    """High-FPS FX engine producing clean, sharp, non-overlapping beam paths."""

    def __init__(self, seed=1, force_scene=None):
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
        chance = 0.65 if self.section == DROP else (0.50 if self.section == MAIN else 0.30)
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

        if self.section not in (MAIN, DROP):
            return

        swap_chance = 0.70 if self.section == DROP else 0.55
        if float(self.rng.random()) > swap_chance:
            return

        if not self._seq_bag:
            self._seq_bag = list(SWAP_SEQ_NAMES)
            self.rng.shuffle(self._seq_bag)
        seq_name = self._seq_bag.pop(0)

        self.swap_instant = bool(float(self.rng.random()) < 0.50)
        use_color_swap = bool(self.force) or (float(self.rng.random()) < 0.45)

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
            if f.section == DROP:
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

        if not self.scene or f.section == SILENCE:
            return [], None

        if f.section == DROP and self._drop_impact_left > 0.0:
            c_drop = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
            progress = 1.0 - (self._drop_impact_left / DROP_IMPACT_BEATS)
            paths = _drop_impact_effect(c_drop, progress, self._drop_impact_variant)
            if progress > 0.78:
                blend = (progress - 0.78) / 0.22
                paths = _morph_paths(paths, SCENES[self.scene](c_drop), blend)
            return _dedupe_exact_lines(paths), DROP_IMPACT_NAMES[self._drop_impact_variant]

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

        if f.section == BREAKDOWN and self._synth_accent_age < 1.0:
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
        return _dedupe_exact_lines(paths), label