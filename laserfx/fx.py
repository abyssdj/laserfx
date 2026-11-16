"""Laser effects: phrase-locked instant vs. morphed effect swaps, colour swaps, varied 1-2 beat
white drop impact explosions (horizontal plane bursts, dot explosions, fast sweeping bars, radial
bursts), slow atmospheric Liquid Sky breakdowns with white depth dots & 1-beat piano/synth
white accents, and expanded effect pools across all sections.

Coordinates are in [-1, 1]; colours are RGB in 0..1.
Designed for a single output feeding twin projectors where Projector 2 is X-mirrored (x -> -x).
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np

from .audio import BREAKDOWN, BUILD, DROP, MAIN, SILENCE, Features

TAU = 2 * math.pi
WHITE = np.array([1.0, 1.0, 1.0], dtype=float)


@dataclass
class Path:
    pts: np.ndarray            # (V, 2)
    rgb: np.ndarray            # (3,) or (V, 3)
    dwell: int = 0             # extra points held at vertices (sharp corners / bright aerial dots)


def hsv(h, s=1.0, v=1.0):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)))


def line(a, b, rgb, dwell=2):
    return Path(np.array([a, b], float), rgb, dwell)


def hot_dot(pt, rgb=WHITE, dwell=4):
    """Creates a piercing aerial laser dot (tiny micro-segment with heavy vertex dwell)."""
    x, y = float(pt[0]), float(pt[1])
    x2 = min(x + 0.008, 0.98) if x < 0.96 else x - 0.008
    return Path(np.array([[x, y], [x2, y]], float), rgb, dwell)


def line_with_dots(a, b, rgb, n_dots=4, dot_rgb=WHITE, dwell=2, dot_dwell=3):
    """Draws a continuous laser plane with subtle white accent dots distributed along its length for depth."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    out = [line(a_arr, b_arr, rgb, dwell=dwell)]
    for i in range(n_dots):
        t = (i + 0.5) / max(n_dots, 1)
        pt = a_arr + (b_arr - a_arr) * t
        out.append(hot_dot(pt, dot_rgb, dwell=dot_dwell))
    return out


def dotted_line(a, b, rgb, dots=5, dot_len=0.035, phase=0.0):
    """Renders a line as a series of crisp, bright aerial laser dots/pips."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    vec = b - a
    total = float(np.hypot(vec[0], vec[1]))
    if total < 1e-4:
        return []
    frac_len = min(dot_len / total, 0.45 / max(dots, 1))
    out = []
    for i in range(dots):
        t0 = ((i + (phase % 1.0)) / dots) % 1.0
        t1 = min(t0 + frac_len, 1.0)
        if t1 <= t0 + 0.004:
            continue
        p0 = a + vec * t0
        p1 = a + vec * t1
        if i % 2 == 1:
            p0, p1 = p1, p0
        out.append(line(p0, p1, rgb, dwell=3))
    return out


def dashed_beam(a, b, rgb, segments=4, duty=0.65, phase=0.0):
    """Smoothly gliding dashed aerial beam segments."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = []
    for i in range(segments):
        t0 = ((i + phase) % segments) / segments
        t1 = min(t0 + duty / segments, 1.0)
        if t1 <= t0 + 0.03:
            continue
        p0 = a + (b - a) * t0
        p1 = a + (b - a) * t1
        if i % 2 == 1:
            p0, p1 = p1, p0
        out.append(line(p0, p1, rgb, dwell=1))
    return out


def ray(ox, oy, ang, length=2.6):
    """Ray from (ox, oy) at `ang` (0 = straight up), clipped cleanly to [-1, 1]."""
    dx, dy, t = math.sin(ang), math.cos(ang), length
    if dx > 1e-9:
        t = min(t, (1.0 - ox) / dx)
    elif dx < -1e-9:
        t = min(t, (-1.0 - ox) / dx)
    if dy > 1e-9:
        t = min(t, (1.0 - oy) / dy)
    elif dy < -1e-9:
        t = min(t, (-1.0 - oy) / dy)
    return (ox, oy), (ox + dx * t, oy + dy * t)


def clipped_plane(cx: float, cy: float, angle: float, span: float = 0.96):
    """Creates a full-span laser plane passing through (cx, cy) perpendicular to `angle`, clipped to [-span, span]."""
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
    if t_max <= t_min + 0.05:
        return None
    return (cx + tx * t_min, cy + ty * t_min), (cx + tx * t_max, cy + ty * t_max)


# ---------------------------------------------------------------- smooth musical motion helpers
def smooth_wave(bt: float, period_beats: float) -> float:
    """Smooth -1..+1 cosine wave locked to exact beat multiples."""
    return -math.cos(bt * TAU / period_beats)


def beat_sweep(bt: float, beats_per_pass: float = 1.0, alternate: bool = True) -> float:
    """Smooth 0..1 sweep across `beats_per_pass` (supports 0.5x double-time and 2.0x half-time)."""
    idx = int(math.floor(bt / beats_per_pass))
    frac = (bt / beats_per_pass) - idx
    u = 0.5 - 0.5 * math.cos(math.pi * frac)
    if alternate and (idx % 2 == 1):
        return 1.0 - u
    return u


def beat_hash(idx: int, salt: int = 0) -> float:
    """Deterministic pseudo-random float in [0, 1) for beat `idx`."""
    v = math.sin(idx * 127.1 + salt * 311.7) * 43758.5453123
    return v - math.floor(v)


def beat_pump(c: Ctx) -> float:
    """Smooth musical punch combining beat_pulse and kick."""
    return min(max(0.55 * c.pulse + 0.45 * c.kick, 0.0), 1.0)


def snare_swell(c: Ctx) -> float:
    """Smooth snare/clap accent envelope."""
    return min(max(getattr(c.f, "snare", 0.0), 0.0), 1.0)


def strobe_gate(c: Ctx, div: float = 4.0, duty: float = 0.45) -> bool:
    """Fast beat-synced strobe gate (respects photosensitivity `c.strobe` toggle)."""
    return (not c.strobe) or ((c.bt * div) % 1.0) < duty


def _zig(i: int, a, b):
    return (a, b) if i % 2 == 0 else (b, a)


# High-contrast modern festival palettes: (primary_hue, accent_hue, accent_sat)
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
    """Solo pure-white impact variations on the first 1-2 beats of a drop."""
    fade = max(0.0, 1.0 - 0.30 * progress)
    col_main = WHITE * fade
    col_soft = WHITE * (0.68 * fade)
    eased = math.sin(min(progress, 1.0) * math.pi * 0.5)
    out = []

    if variant == 0:
        # Variation 0: Radial white starburst shattering into an expanding dot ring
        ox, oy = -0.32, -0.10
        n_rays = c.count(6, 9)
        spin = 0.35 * progress
        reach = 0.45 + 1.85 * min(progress * 1.4, 1.0)
        for i in range(n_rays):
            ang = spin + (i / max(n_rays, 1)) * TAU
            p0, p1 = ray(ox, oy, ang, reach)
            if progress < 0.55:
                out.append(line(*_zig(i, p0, p1), col_main, dwell=2))
            else:
                out.extend(dotted_line(p0, p1, col_main, dots=4, dot_len=0.045, phase=progress * 2.0))
            out.append(hot_dot(p1, WHITE, dwell=4))
        n_dots = c.count(8, 12)
        r_ring = 0.25 + 0.72 * eased
        for d in range(n_dots):
            a = -spin * 1.5 + (d / n_dots) * TAU
            px = float(np.clip(ox + r_ring * math.cos(a), -0.96, 0.96))
            py = float(np.clip(oy + r_ring * math.sin(a), -0.96, 0.96))
            out.append(hot_dot((px, py), col_soft if d % 2 else col_main, dwell=4))

    elif variant == 1:
        # Variation 1: Horizontal white plane burst blasting outward from center horizon to top & bottom
        pairs = c.count(2, 4)
        for p in range(pairs):
            frac = (p + 1) / pairs
            y_off = (0.10 + 0.85 * eased) * frac
            slant = 0.14 * (1.0 - eased) * (1.0 if p % 2 == 0 else -1.0)
            for sign in (-1.0, 1.0):
                y0 = float(np.clip(sign * y_off - slant, -0.96, 0.96))
                y1 = float(np.clip(sign * y_off + slant, -0.96, 0.96))
                out.extend(line_with_dots((-0.96, y0), (0.96, y1), col_main, n_dots=4, dot_rgb=WHITE, dwell=2, dot_dwell=3))

    elif variant == 2:
        # Variation 2: High-density white dot explosion erupting outward in concentric spirals
        ox, oy = -0.25, 0.05
        n_arms = c.count(6, 10)
        dots_per_arm = c.count(3, 5)
        for arm in range(n_arms):
            base_a = (arm / n_arms) * TAU + progress * 0.9
            for k in range(dots_per_arm):
                r = (0.12 + 0.95 * eased) * ((k + 1) / dots_per_arm)
                a = base_a + k * 0.18 * progress
                px = float(np.clip(ox + r * math.cos(a), -0.96, 0.96))
                py = float(np.clip(oy + r * math.sin(a), -0.96, 0.96))
                out.append(hot_dot((px, py), col_main if k == dots_per_arm - 1 else col_soft, dwell=4))

    else:
        # Variation 3: Fast sweeping white vertical & horizontal bars whipping across the whole frame
        u = beat_sweep(progress * 2.0, beats_per_pass=1.0, alternate=True)
        x_a = -0.92 + 1.84 * u
        x_b = 0.92 - 1.84 * u
        y_c = -0.88 + 1.76 * eased
        out.append(line((x_a, -0.96), (x_a, 0.96), col_main, dwell=2))
        out.append(line((x_b, 0.96), (x_b, -0.96), col_main, dwell=2))
        out.extend(line_with_dots((-0.96, y_c), (0.96, y_c), col_soft, n_dots=5, dot_rgb=WHITE, dwell=2, dot_dwell=3))
        out.extend(line_with_dots((0.96, -y_c), (-0.96, -y_c), col_soft, n_dots=5, dot_rgb=WHITE, dwell=2, dot_dwell=3))

    return out


# ======================================================================== ATMOSPHERIC BREAKDOWN SCENES (LOW LINE COUNT)

def sc_liquid_sky_drift(c: Ctx):
    """Slow, ultra-relaxed Liquid Sky planes (1-2 lines) drifting over 16 beats with white depth dots."""
    planes = c.count(1, 2)
    tilt = 0.24 * smooth_wave(c.bt, 16.0)
    y_base = 0.34 * smooth_wave(c.bt, 32.0)
    dot_col = WHITE * 0.72

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * 0.26
        y0 = float(np.clip(y_base + gap - tilt, -0.92, 0.92))
        y1 = float(np.clip(y_base + gap + tilt, -0.92, 0.92))
        a, b = _zig(p, (-0.95, y0), (0.95, y1))
        out.extend(line_with_dots(a, b, c.palette(p, v=0.76), n_dots=4, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    return out


def sc_morphing_plane_sky(c: Ctx):
    """Two slow sky planes that smoothly morph their angle between horizontal canopy and diagonal V-sky."""
    morph = 0.5 + 0.5 * smooth_wave(c.bt, 16.0)
    y_center = 0.26 + 0.18 * smooth_wave(c.bt, 24.0)
    tilt = 0.42 * morph
    dot_col = WHITE * 0.75

    p0_a = (-0.95, float(np.clip(y_center - tilt, -0.92, 0.92)))
    p0_b = (0.95, float(np.clip(y_center + tilt, -0.92, 0.92)))
    p1_a = (-0.95, float(np.clip(y_center + tilt * 0.75 - 0.18, -0.92, 0.92)))
    p1_b = (0.95, float(np.clip(y_center - tilt * 0.75 - 0.18, -0.92, 0.92)))

    out = []
    out.extend(line_with_dots(p0_a, p0_b, c.palette(0, v=0.78), n_dots=5, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    if c.detail >= 0.35:
        out.extend(line_with_dots(p1_b, p1_a, c.palette(1, v=0.72), n_dots=4, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    return out


def sc_breathe_horizon_sky(c: Ctx):
    """Single/dual wide sky sheet that breathes open and closed slowly over 16 beats with white dots."""
    spread = 0.08 + 0.24 * (0.5 + 0.5 * smooth_wave(c.bt, 16.0))
    y_mid = 0.28 + 0.16 * smooth_wave(c.bt, 32.0)
    rock = 0.14 * smooth_wave(c.bt, 12.0)
    dot_col = WHITE * 0.70

    out = []
    for idx, sign in enumerate((-1.0, 1.0)):
        if idx == 1 and c.detail < 0.3:
            break
        y0 = float(np.clip(y_mid + sign * spread - rock, -0.92, 0.92))
        y1 = float(np.clip(y_mid + sign * spread + rock, -0.92, 0.92))
        a, b = _zig(idx, (-0.95, y0), (0.95, y1))
        out.extend(line_with_dots(a, b, c.palette(idx, v=0.75), n_dots=4, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    return out


def sc_velvet_sky_sweep(c: Ctx):
    """Ultra-slow vertical sheet (1-2 lines) drifting across the room over 8 beats with white depth points."""
    u = beat_sweep(c.bt, beats_per_pass=8.0, alternate=True)
    x_center = -0.82 + 1.64 * u
    blades = c.count(1, 2)
    width = 0.20 + 0.08 * smooth_wave(c.bt, 16.0)
    dot_col = WHITE * 0.70

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * width
        x = float(np.clip(x_center + off, -0.94, 0.94))
        a, b = _zig(i, (x, -0.92), (x, 0.92))
        out.extend(line_with_dots(a, b, c.palette(i % 2, v=0.74), n_dots=4, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    return out


def sc_silk_wave_sky(c: Ctx):
    """Single slow undulating sky ribbon with white accent pearls along the crest."""
    x = np.linspace(-0.94, 0.94, 48)
    ph = c.bt * TAU / 16.0
    y = np.clip(0.32 + 0.16 * smooth_wave(c.bt, 24.0) + 0.15 * np.sin(1.6 * math.pi * x + ph), -0.92, 0.92)
    pts = np.stack([x, y], 1)
    out = [Path(pts, c.palette(0, v=0.76), dwell=0)]
    dot_col = WHITE * 0.72
    for idx in (8, 18, 29, 39):
        out.append(hot_dot(pts[idx], dot_col, dwell=3))
    return out


def sc_dual_horizon_mirage(c: Ctx):
    """Two ultra-slow counter-tilting sky planes (2 lines) with white midpoint & tip pearls."""
    ang = 0.34 * smooth_wave(c.bt, 20.0)
    y_base = 0.30 + 0.14 * smooth_wave(c.bt, 32.0)
    dot_col = WHITE * 0.72
    out = []
    out.extend(line_with_dots((-0.95, y_base - ang), (0.95, y_base + ang), c.palette(0, v=0.75), n_dots=5, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    out.extend(line_with_dots((0.95, y_base - ang * 0.6 - 0.16), (-0.95, y_base + ang * 0.6 - 0.16), c.palette(1, v=0.70), n_dots=4, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    return out


def sc_floating_dot_constellation(c: Ctx):
    """Single whisper-thin horizon line beneath a slow drifting constellation of white & pastel aerial dots."""
    y_sky = 0.18 + 0.14 * smooth_wave(c.bt, 24.0)
    out = line_with_dots((-0.95, y_sky), (0.95, y_sky), c.palette(0, v=0.68), n_dots=3, dot_rgb=WHITE * 0.65, dwell=1, dot_dwell=2)
    n_stars = c.count(5, 8)
    for s in range(n_stars):
        u = (s + 0.5) / n_stars
        x = -0.86 + 1.72 * u + 0.08 * math.sin(c.bt * TAU / 16.0 + s)
        y = 0.42 + 0.34 * math.sin(c.bt * TAU / 20.0 + s * 1.7)
        col = WHITE * 0.78 if s % 2 == 0 else c.palette(1, v=0.80)
        out.append(hot_dot((float(np.clip(x, -0.94, 0.94)), float(np.clip(y, -0.90, 0.92))), col, dwell=3))
    return out


def sc_slow_v_canopy_sky(c: Ctx):
    """Wide V-shaped atmospheric canopy (2 lines) slowly opening and breathing over 16 beats."""
    y_apex = 0.14 + 0.18 * smooth_wave(c.bt, 24.0)
    wing_lift = 0.28 + 0.24 * smooth_wave(c.bt, 16.0)
    y_wing = float(np.clip(y_apex + wing_lift, -0.90, 0.92))
    dot_col = WHITE * 0.74
    out = []
    out.extend(line_with_dots((-0.95, y_wing), (0.0, y_apex), c.palette(0, v=0.76), n_dots=3, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    out.extend(line_with_dots((0.0, y_apex), (0.95, y_wing), c.palette(1, v=0.76), n_dots=3, dot_rgb=dot_col, dwell=1, dot_dwell=3))
    return out


def sc_aurora_sky_curtain(c: Ctx):
    """Two gently undulating overhead sky curtains drifting in slow phase opposition with white depth dots."""
    x = np.linspace(-0.94, 0.94, 44)
    out = []
    dot_col = WHITE * 0.70
    for layer in range(c.count(1, 2)):
        ph = c.bt * TAU / 18.0 + layer * 2.1
        y = 0.34 + 0.14 * smooth_wave(c.bt, 28.0) + 0.13 * np.sin(1.4 * math.pi * x + ph) - layer * 0.18
        pts = np.stack([x, np.clip(y, -0.92, 0.92)], 1)
        if layer == 1:
            pts = pts[::-1]
        out.append(Path(pts, c.palette(layer, v=0.74), dwell=0))
        for idx in (10, 22, 34):
            out.append(hot_dot(pts[idx], dot_col, dwell=3))
    return out


def _apply_synth_accent(paths: list[Path], age_beats: float, kind: int, c: Ctx) -> list[Path]:
    """Subtle 1-beat white accent line or micro-burst triggered by piano chords / synth plucks in breakdowns."""
    if age_beats < 0.0 or age_beats >= 1.0:
        return paths
    env = math.sin(min(age_beats * 4.0, 1.0) * math.pi * 0.5) * ((1.0 - age_beats) ** 1.4)
    if env <= 0.03:
        return paths
    col = WHITE * (0.68 * env)

    if kind % 2 == 0:
        y = 0.18 + 0.36 * age_beats
        tilt = 0.16 * (1.0 if kind % 4 == 0 else -1.0)
        return paths + line_with_dots((-0.92, y - tilt), (0.92, y + tilt), col, n_dots=3, dot_rgb=col, dwell=1, dot_dwell=2)
    else:
        ox, oy = -0.38, 0.34
        r = 0.08 + 0.28 * age_beats
        burst = []
        for k in range(4):
            a = (k / 4.0) * TAU + age_beats * 0.8
            px = float(np.clip(ox + r * math.cos(a), -0.94, 0.94))
            py = float(np.clip(oy + r * 0.7 * math.sin(a), -0.94, 0.94))
            burst.append(hot_dot((px, py), col, dwell=3))
        return paths + burst


# ======================================================================== MAIN & DROP SCENES

def sc_split_direction_sweep(c: Ctx):
    """Upper and lower halves sweep in OPPOSITE directions across the entire width on every beat."""
    pump = beat_pump(c)
    u_top = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    u_bot = 1.0 - u_top

    x_top = -0.92 + 1.84 * u_top
    x_bot = -0.92 + 1.84 * u_bot
    blades = c.count(2, 3)
    gap = 0.04 + 0.05 * pump

    out = []
    idx = 0
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        xt = float(np.clip(x_top + off, -0.96, 0.96))
        xb = float(np.clip(x_bot + off, -0.96, 0.96))
        is_core = (i == blades // 2)
        out.append(line(*_zig(idx, (xt, 0.06), (xt, 0.96)), c.palette(0, white_boost=0.75 * pump if is_core else 0.0), dwell=2))
        idx += 1
        out.append(line(*_zig(idx, (xb, -0.96), (xb, -0.06)), c.palette(1, white_boost=0.75 * pump if is_core else 0.0), dwell=2))
        idx += 1
    return out


def sc_half_plane_stagger(c: Ctx):
    """Half-length horizontal & vertical planes sweeping staggered zones across the frame."""
    pump = beat_pump(c)
    u1 = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    u2 = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)

    y_left = -0.88 + 1.76 * u1
    y_right = -0.88 + 1.76 * (1.0 - u2)
    x_top = -0.90 + 1.80 * u2
    x_bot = -0.90 + 1.80 * (1.0 - u1)

    return [
        line((-0.96, y_left), (-0.06, y_left + 0.10), c.palette(0, white_boost=0.65 * pump), dwell=2),
        line((0.96, y_right), (0.06, y_right - 0.10), c.palette(1, white_boost=0.65 * pump), dwell=2),
        line((x_top, 0.06), (x_top, 0.96), c.palette(1, white_boost=0.5 * pump), dwell=2),
        line((x_bot, -0.96), (x_bot, -0.06), c.palette(0, white_boost=0.5 * pump), dwell=2),
    ]


def sc_double_time_whip(c: Ctx):
    """Double-time (2x BPM) fast sweeps: slashes across the entire width twice per beat."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    x_center = -0.92 + 1.84 * u
    slant = 0.24 * (1.0 if int(c.bt * 2) % 2 == 0 else -1.0)
    blades = c.count(2, 3)
    gap = 0.045 + 0.05 * pump

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        x0 = float(np.clip(x_center + off - slant, -0.96, 0.96))
        x1 = float(np.clip(x_center + off + slant, -0.96, 0.96))
        is_core = (i == blades // 2)
        col = c.palette(i % 2, v=1.0, white_boost=0.85 * pump if is_core else 0.0)
        out.append(line(*_zig(i, (x0, -0.96), (x1, 0.96)), col, dwell=2))
    return out


def sc_half_time_glide(c: Ctx):
    """Half-time (0.5x BPM) heavy sweep: takes 2 full beats per pass with wide multi-beam planes."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)
    y_center = -0.88 + 1.76 * u
    slant = 0.22 * smooth_wave(c.bt, 8.0)
    blades = c.count(3, 5)
    gap = 0.06 + 0.08 * pump

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        y0 = float(np.clip(y_center + off - slant, -0.96, 0.96))
        y1 = float(np.clip(y_center + off + slant, -0.96, 0.96))
        is_core = (i == blades // 2)
        col = c.palette(i % 2, v=1.0, white_boost=0.7 * pump if is_core else 0.0)
        out.append(line(*_zig(i, (-0.96, y0), (0.96, y1)), col, dwell=2))
    return out


def sc_dotted_fan_burst(c: Ctx):
    """Modern fan with piercing WHITE HOT-DOTS on every beam tip + dotted grating sub-beams."""
    pump = beat_pump(c)
    n = c.count(5, 8, max(c.energy, 0.45))
    aim = 0.26 + 0.48 * smooth_wave(c.bt, 8.0)
    spread = 0.34 + 0.26 * pump
    ox = -0.58

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        reach = 1.65 + 0.35 * math.cos(u * math.pi) + 0.25 * pump
        p0, p1 = ray(ox, -0.96, ang, reach)
        col = c.palette(i % 2, v=0.95)
        if i % 2 == 1:
            out.extend(dotted_line(p0, p1, col, dots=5, dot_len=0.04, phase=c.bt * 2.0))
        else:
            out.append(line(*_zig(i, p0, p1), col, dwell=1))
        out.append(hot_dot(p1, WHITE, dwell=4))
    return out


def sc_dot_matrix_sweep(c: Ctx):
    """Sweeping dotted laser curtain: rows of bright white & colored aerial dots sweeping on every beat."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_center = -0.88 + 1.76 * u
    rows = c.count(2, 4)
    dots_per_row = c.count(5, 8)
    slant = 0.18 * smooth_wave(c.bt, 4.0)

    out = []
    for r in range(rows):
        off = (r - (rows - 1) / 2.0) * (0.14 + 0.14 * pump)
        y0 = float(np.clip(y_center + off - slant, -0.95, 0.95))
        y1 = float(np.clip(y_center + off + slant, -0.95, 0.95))
        col = WHITE if (r == 0 and pump > 0.35) else c.palette(r % 2, v=1.0)
        out.extend(dotted_line((-0.95, y0), (0.95, y1), col, dots=dots_per_row, dot_len=0.035, phase=c.bt * 1.5 + r * 0.3))
    return out


def sc_drop_hyper_strobe(c: Ctx):
    """High-energy drop strobe cue: 1/8th & 1/16th note gated white-core sweeps and split slashes."""
    div = 4.0 if (int(c.bt / 2) % 2 == 1) else 2.0
    if not strobe_gate(c, div=div, duty=0.48):
        return []

    pump = beat_pump(c)
    step = int(c.bt * div)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    out = []

    if step % 3 == 0:
        x = -0.92 + 1.84 * u
        out.append(line((x, -0.96), (x, 0.96), WHITE, dwell=3))
        out.append(line((max(x - 0.08, -0.96), -0.96), (max(x - 0.08, -0.96), 0.96), c.palette(0), dwell=2))
    elif step % 3 == 1:
        y_top = 0.10 + 0.82 * u
        y_bot = -0.10 - 0.82 * u
        out.append(line((-0.96, y_top), (0.96, y_top + 0.12), c.palette(1, white_boost=0.8 * pump), dwell=2))
        out.append(line((0.96, y_bot), (-0.96, y_bot - 0.12), WHITE, dwell=2))
    else:
        n = c.count(5, 7)
        aim = 0.25 + 0.50 * u
        for i in range(n):
            frac = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(-0.62, -0.96, aim + frac * 0.85, 2.3)
            out.append(line(*_zig(i, p0, p1), c.palette(i % 2, white_boost=0.6), dwell=1))
            out.append(hot_dot(p1, WHITE, dwell=4))
    return out


def sc_full_width_wall(c: Ctx):
    """Tall vertical laser wall sweeping the ENTIRE width (-0.96 to +0.96) on every single beat."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    x_center = -0.92 + 1.84 * u
    blades = c.count(2, 4)
    thickness = 0.03 + 0.05 * pump
    tilt = 0.08 * math.sin(c.bt * math.pi)

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * thickness
        x0 = float(np.clip(x_center + off - tilt, -0.96, 0.96))
        x1 = float(np.clip(x_center + off + tilt, -0.96, 0.96))
        is_core = (i == blades // 2)
        col = c.palette(0 if is_core else 1, v=1.0, white_boost=0.75 * pump if is_core else 0.0)
        out.append(line(*_zig(i, (x0, -0.96), (x1, 0.96)), col, dwell=2))
    return out


def sc_full_height_sheet(c: Ctx):
    """Wide horizontal Liquid Sky sheet sweeping the ENTIRE height (-0.95 to +0.95) on every beat."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_center = -0.90 + 1.80 * u
    slant = 0.16 * (1.0 if int(c.bt) % 2 == 0 else -1.0)
    blades = c.count(2, 4)
    gap = 0.035 + 0.06 * pump

    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * gap
        y0 = float(np.clip(y_center + off - slant, -0.96, 0.96))
        y1 = float(np.clip(y_center + off + slant, -0.96, 0.96))
        is_core = (i == blades // 2)
        col = c.palette(1 if is_core else 0, v=1.0, white_boost=0.75 * pump if is_core else 0.0)
        out.append(line(*_zig(i, (-0.96, y0), (0.96, y1)), col, dwell=2))
    return out


def sc_random_dir_sweep(c: Ctx):
    """Picks a new random sweep direction every beat across the whole frame."""
    pump = beat_pump(c)
    beat_idx = int(math.floor(c.bt))
    frac = c.bt - beat_idx
    u = 0.5 - 0.5 * math.cos(math.pi * frac)

    dir_idx = int(beat_hash(beat_idx, salt=11) * 8)
    angle = dir_idx * (TAU / 8.0)

    dist = -1.05 + 2.10 * u
    cx, cy = dist * math.cos(angle), dist * math.sin(angle)

    blades = c.count(2, 3)
    spacing = 0.045 + 0.055 * pump
    out = []
    for i in range(blades):
        off = (i - (blades - 1) / 2.0) * spacing
        px = cx + off * math.cos(angle)
        py = cy + off * math.sin(angle)
        seg = clipped_plane(px, py, angle, span=0.96)
        if seg is not None:
            is_core = (i == blades // 2)
            col = c.palette(beat_idx % 2 if is_core else (i % 2), v=1.0, white_boost=0.8 * pump if is_core else 0.0)
            out.append(line(*_zig(i, seg[0], seg[1]), col, dwell=2))
    return out


def sc_multi_plane_cross_sweep(c: Ctx):
    """Multiple wide & tall planes sweeping simultaneously in opposing directions across the entire venue."""
    pump = beat_pump(c)
    n_planes = c.count(2, 3)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    v = beat_sweep(c.bt, beats_per_pass=2.0, alternate=True)

    out = []
    idx = 0
    for i in range(n_planes):
        phase_off = i / max(n_planes, 1)
        pos = ((u + phase_off) % 1.0)
        x = -0.92 + 1.84 * pos
        col = c.palette(0, v=1.0, white_boost=0.65 * pump if i == 0 else 0.0)
        out.append(line(*_zig(idx, (x, -0.96), (x, 0.96)), col, dwell=2))
        idx += 1

    for j in range(n_planes):
        phase_off = j / max(n_planes, 1)
        pos = ((1.0 - v + phase_off) % 1.0)
        y = -0.92 + 1.84 * pos
        slant = 0.12 * (1.0 if j % 2 == 0 else -1.0)
        y0 = float(np.clip(y - slant, -0.96, 0.96))
        y1 = float(np.clip(y + slant, -0.96, 0.96))
        col = c.palette(1, v=1.0, white_boost=0.65 * pump if j == 0 else 0.0)
        out.append(line(*_zig(idx, (-0.96, y0), (0.96, y1)), col, dwell=2))
        idx += 1
    return out


def sc_random_multi_blade(c: Ctx):
    """Fires 2 to 4 independent full-frame planes on each beat, each sweeping a random angle & direction."""
    pump = beat_pump(c)
    beat_idx = int(math.floor(c.bt))
    frac = c.bt - beat_idx
    u = 0.5 - 0.5 * math.cos(math.pi * frac)

    n_blades = c.count(2, 4)
    out = []
    for b in range(n_blades):
        ang = beat_hash(beat_idx, salt=30 + b * 17) * TAU
        flip = -1.0 if beat_hash(beat_idx, salt=90 + b * 13) > 0.5 else 1.0
        dist = flip * (-1.0 + 2.0 * u)
        cx, cy = dist * math.cos(ang), dist * math.sin(ang)
        seg = clipped_plane(cx, cy, ang, span=0.96)
        if seg is not None:
            col = c.palette(b % 2, v=1.0, white_boost=0.8 * pump if b == 0 else 0.0)
            out.append(line(*_zig(b, seg[0], seg[1]), col, dwell=2))
    return out


def sc_twin_scissor_sweep(c: Ctx):
    """Opposing floor and ceiling beam fans with white endpoint hot-dots gliding in counter-phase."""
    pump = beat_pump(c)
    snare = snare_swell(c)
    n = c.count(4, 6, max(c.energy, 0.45))
    swing = 0.46 * smooth_wave(c.bt, 8.0)

    out = []
    idx = 0
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = 0.34 + swing + u * (0.30 + 0.18 * pump)
        p0, p1 = ray(-0.78, -0.96, ang, 2.2)
        out.append(line(*_zig(idx, p0, p1), c.palette(0, v=1.0), dwell=1))
        out.append(hot_dot(p1, WHITE, dwell=3))
        idx += 1

    for i in range(max(3, n - 1)):
        u = (i / max(n - 2, 1)) - 0.5
        ang = math.pi - (0.36 - swing * 0.85 + u * (0.28 + 0.18 * snare))
        p0, p1 = ray(-0.28, 0.96, ang, 2.2)
        out.append(line(*_zig(idx, p0, p1), c.palette(1, v=0.85 + 0.15 * snare), dwell=1))
        idx += 1
    return out


def sc_drop_x_blades(c: Ctx):
    """High-speed sweeping aerial X-sheets that slice up and down on 2-beat sweeps."""
    pump = beat_pump(c)
    planes = c.count(3, 5)
    tilt = 0.58 * smooth_wave(c.bt, 4.0)
    y_sweep = 0.35 * smooth_wave(c.bt, 8.0)

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.14 + 0.16 * pump)
        y0 = float(np.clip(y_sweep + gap - tilt, -0.95, 0.95))
        y1 = float(np.clip(y_sweep + gap + tilt, -0.95, 0.95))
        col = c.palette(p % 2, v=1.0, white_boost=0.8 * pump if p == 0 else 0.0)
        out.append(line(*_zig(p, (-0.96, y0), (0.96, y1)), col, dwell=2))
    return out


# ---------------- NEW MAIN & DROP SCENES FOR MAXIMUM VARIATION

def sc_quadrant_mirror_sweep(c: Ctx):
    """Four corner-anchored beam blades sweeping inward/outward in synchronized quadrants."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    sweep_ang = 0.18 + 1.05 * u
    out = []
    # Bottom-left & top-left origins (X-mirrored unit adds right corners automatically)
    p0, p1 = ray(-0.94, -0.94, sweep_ang, 1.9 + 0.3 * pump)
    out.append(line(p0, p1, c.palette(0, white_boost=0.7 * pump), dwell=2))
    out.append(hot_dot(p1, WHITE, dwell=4))

    p2, p3 = ray(-0.94, 0.94, math.pi - sweep_ang, 1.9 + 0.3 * pump)
    out.append(line(p3, p2, c.palette(1, white_boost=0.7 * pump), dwell=2))
    out.append(hot_dot(p3, WHITE, dwell=4))
    return out


def sc_pendulum_beam_bar(c: Ctx):
    """Wide rotating aerial light bar pivoting smoothly like a giant pendulum across the center."""
    pump = beat_pump(c)
    ang = 0.85 * smooth_wave(c.bt, 4.0)
    bars = c.count(2, 4)
    spacing = 0.09 + 0.08 * pump
    out = []
    for i in range(bars):
        off = (i - (bars - 1) / 2.0) * spacing
        seg = clipped_plane(-0.22 + off * math.cos(ang), off * math.sin(ang), ang, span=0.95)
        if seg is not None:
            col = c.palette(i % 2, white_boost=0.75 * pump if i == 0 else 0.0)
            out.append(line(*_zig(i, seg[0], seg[1]), col, dwell=2))
    return out


def sc_crisscross_ribbon_planes(c: Ctx):
    """Two counter-sweeping diagonal laser ribbons with glowing white depth dots along their ridges."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=1.0, alternate=True)
    y_a = -0.75 + 1.50 * u
    y_b = 0.75 - 1.50 * u
    out = []
    out.extend(line_with_dots((-0.95, y_a), (0.95, float(np.clip(y_a + 0.42, -0.95, 0.95))), c.palette(0, white_boost=0.6 * pump), n_dots=4, dot_rgb=WHITE, dwell=2, dot_dwell=3))
    out.extend(line_with_dots((0.95, y_b), (-0.95, float(np.clip(y_b + 0.42, -0.95, 0.95))), c.palette(1, white_boost=0.6 * pump), n_dots=4, dot_rgb=WHITE, dwell=2, dot_dwell=3))
    return out


def sc_alternating_horizon_gates(c: Ctx):
    """Beat-alternating wide horizontal gates that open from center on odd beats and close on even beats."""
    pump = beat_pump(c)
    beat_idx = int(math.floor(c.bt))
    frac = c.bt - beat_idx
    eased = 0.5 - 0.5 * math.cos(math.pi * frac)
    gap = (0.12 + 0.78 * eased) if (beat_idx % 2 == 0) else (0.90 - 0.78 * eased)
    out = [
        line((-0.96, gap), (0.96, gap), c.palette(0, white_boost=0.75 * pump), dwell=2),
        line((0.96, -gap), (-0.96, -gap), c.palette(1, white_boost=0.75 * pump), dwell=2),
    ]
    if c.detail >= 0.4:
        out.extend(dotted_line((-0.90, 0.0), (0.90, 0.0), WHITE * (0.5 + 0.5 * pump), dots=6, dot_len=0.03, phase=c.bt * 2.0))
    return out


def sc_prism_dot_fan(c: Ctx):
    """Asymmetric prism fan with alternating solid beams, dotted trails, and double white tip dots."""
    pump = beat_pump(c)
    n = c.count(5, 8)
    aim = 0.30 + 0.44 * smooth_wave(c.bt, 6.0)
    spread = 0.36 + 0.22 * pump
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(-0.66, -0.95, ang, 2.25)
        col = c.palette(i % 2)
        out.append(line(*_zig(i, p0, p1), col, dwell=1))
        out.append(hot_dot(p1, WHITE, dwell=4))
        mid = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
        out.append(hot_dot(mid, WHITE * 0.85, dwell=3))
    return out


def sc_hyper_bar_ricochet(c: Ctx):
    """Drop fast sweeping bars: 3 to 5 tall & wide bars ricocheting at 2x BPM across the entire stage."""
    pump = beat_pump(c)
    u1 = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    u2 = beat_sweep(c.bt + 0.25, beats_per_pass=0.5, alternate=True)
    x1 = -0.92 + 1.84 * u1
    x2 = 0.92 - 1.84 * u2
    y1 = -0.90 + 1.80 * u2
    out = [
        line((x1, -0.96), (x1, 0.96), c.palette(0, white_boost=0.85 * pump), dwell=2),
        line((x2, 0.96), (x2, -0.96), c.palette(1, white_boost=0.85 * pump), dwell=2),
        line((-0.96, y1), (0.96, y1), WHITE if pump > 0.45 else c.palette(0), dwell=2),
    ]
    return out


def sc_horizontal_plane_blast(c: Ctx):
    """Drop horizontal plane burst: stacked wide planes erupting outward from the center on every beat."""
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    eased = math.sin(frac * math.pi * 0.5)
    pairs = c.count(2, 3)
    out = []
    idx = 0
    for p in range(pairs):
        scale = (p + 1) / pairs
        y = (0.08 + 0.84 * eased) * scale
        col = c.palette(p % 2, white_boost=0.8 * pump if p == 0 else 0.0)
        out.append(line(*_zig(idx, (-0.96, y), (0.96, y)), col, dwell=2))
        idx += 1
        out.append(line(*_zig(idx, (-0.96, -y), (0.96, -y)), col, dwell=2))
        idx += 1
    return out


def sc_shatter_dot_storm(c: Ctx):
    """Drop dot explosion storm: fast expanding rings and sweeping rows of piercing white & neon dots."""
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    r = 0.15 + 0.85 * frac
    n_dots = c.count(10, 16)
    ox, oy = -0.35 * math.cos(c.bt * TAU / 8.0), 0.0
    out = []
    for i in range(n_dots):
        ang = c.bt * 1.4 + (i / n_dots) * TAU
        rad = r * (0.55 if i % 2 == 0 else 1.0)
        px = float(np.clip(ox + rad * math.cos(ang), -0.96, 0.96))
        py = float(np.clip(oy + rad * math.sin(ang), -0.96, 0.96))
        col = WHITE if (i % 3 == 0 or pump > 0.5) else c.palette(i % 2)
        out.append(hot_dot((px, py), col, dwell=4))
    return out


def sc_quad_corner_lasers(c: Ctx):
    """High-speed corner beam whips firing across the diagonal on every beat."""
    pump = beat_pump(c)
    u = beat_sweep(c.bt, beats_per_pass=0.5, alternate=True)
    blades = c.count(2, 3)
    out = []
    for i in range(blades):
        ang = 0.20 + 0.95 * u + (i - 1) * 0.14
        p0, p1 = ray(-0.92, -0.92, ang, 2.5)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2, white_boost=0.8 * pump), dwell=2))
        out.append(hot_dot(p1, WHITE, dwell=4))
    return out


def sc_strobe_tunnel_ribs(c: Ctx):
    """Fast-pulsing geometric aerial ribs that expand on every beat with white corner dots."""
    pump = beat_pump(c)
    frac = c.bt - math.floor(c.bt)
    layers = c.count(2, 3)
    cx = -0.28 * smooth_wave(c.bt, 8.0)
    out = []
    for l in range(layers):
        s = 0.25 + 0.68 * (((frac + l / layers) % 1.0))
        pts = np.array([
            [max(cx - s, -0.95), -s * 0.85],
            [min(cx + s, 0.95), -s * 0.85],
            [min(cx + s, 0.95), s * 0.85],
            [max(cx - s, -0.95), s * 0.85],
            [max(cx - s, -0.95), -s * 0.85],
        ], float)
        out.append(Path(pts, c.palette(l % 2, white_boost=0.7 * pump), dwell=2))
    return out


# ======================================================================== BUILD SECTION SCENES

def sc_build_converge_sweep(c: Ctx):
    """Smoothly accelerating dotted fan sweep that narrows into a bright center beam at the drop."""
    b = c.f.build
    n = c.count(6, 10, max(b, 0.45))
    sweep = (0.50 * (1.0 - 0.65 * (b ** 2))) * math.sin(c.rt * TAU / 4.0)
    bias = 0.34 * (1.0 - 0.7 * b)
    spread = 0.50 * (1.0 - 0.78 * (b ** 2)) + 0.05
    ox = -0.60 * (1.0 - 0.65 * b)

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = bias + sweep + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.3)
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.85 * (b ** 2))
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
        out.append(hot_dot(p1, WHITE, dwell=3))
    return out


def sc_build_rising_sheet(c: Ctx):
    """Horizontal Liquid Sky planes that sweep upward toward the ceiling as tension builds."""
    b = c.f.build
    planes = c.count(2, 4)
    y_rise = -0.55 + 1.35 * b
    tilt = (0.32 * (1.0 - 0.5 * b)) * math.sin(c.rt * TAU / 4.0)

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.24 * (1.0 - 0.5 * b) + 0.06)
        y0 = float(np.clip(y_rise + gap - tilt, -0.94, 0.94))
        y1 = float(np.clip(y_rise + gap + tilt, -0.94, 0.94))
        col = c.palette(p % 2, v=0.8 + 0.2 * b, white_boost=0.8 * (b ** 2))
        out.append(line(*_zig(p, (-0.96, y0), (0.96, y1)), col, dwell=2))
    return out


def sc_build_prism_tighten(c: Ctx):
    """Accelerating full-width vertical beam wall sweeping faster as the drop approaches."""
    b = c.f.build
    u = beat_sweep(c.rt, beats_per_pass=1.0, alternate=True)
    x_center = -0.90 + 1.80 * u
    blades = c.count(3, 6)
    width = 0.24 * (1.0 - 0.7 * b) + 0.04
    out = []
    for i in range(blades):
        off = (i / max(blades - 1, 1) - 0.5) * width
        x = float(np.clip(x_center + off, -0.95, 0.95))
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.85 * (b ** 2))
        out.append(line(*_zig(i, (x, -0.95), (x, 0.95)), col, dwell=2))
    return out


def sc_build_accelerating_bars(c: Ctx):
    """Opposing horizontal & vertical bars that accelerate inward toward center impact at drop."""
    b = c.f.build
    u = beat_sweep(c.rt, beats_per_pass=0.5, alternate=True)
    span = 0.92 * (1.0 - 0.75 * b) + 0.06
    x = -span + 2.0 * span * u
    col = c.palette(0, v=0.8 + 0.2 * b, white_boost=0.9 * (b ** 2))
    return [
        line((x, -0.95), (x, 0.95), col, dwell=2),
        line((-x, 0.95), (-x, -0.95), WHITE * (0.4 + 0.6 * b), dwell=2),
    ]


def sc_build_dot_convergence(c: Ctx):
    """Swirling ring of white & colored dots spiraling inward into a tight high-energy core."""
    b = c.f.build
    n_dots = c.count(8, 14)
    r = 0.90 * (1.0 - 0.82 * b) + 0.05
    ox = -0.32 * (1.0 - b)
    out = []
    for i in range(n_dots):
        ang = c.rt * 2.2 + (i / n_dots) * TAU
        px = float(np.clip(ox + r * math.cos(ang), -0.95, 0.95))
        py = float(np.clip(r * math.sin(ang), -0.95, 0.95))
        col = WHITE if (b > 0.65 or i % 2 == 0) else c.palette(i % 2)
        out.append(hot_dot((px, py), col, dwell=4))
    return out


def sc_build_split_pinch(c: Ctx):
    """Top and bottom horizontal sheets pinching toward the center horizon as build peaks."""
    b = c.f.build
    y_gap = 0.90 * (1.0 - 0.85 * b) + 0.04
    wobble = 0.18 * (1.0 - b) * math.sin(c.rt * TAU / 2.0)
    col = c.palette(1, v=0.8 + 0.2 * b, white_boost=0.85 * (b ** 2))
    return [
        line((-0.95, y_gap - wobble), (0.95, y_gap + wobble), col, dwell=2),
        line((0.95, -y_gap + wobble), (-0.95, -y_gap - wobble), col, dwell=2),
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
            if len(p.pts) == 2:
                dots.append(hot_dot(p.pts[1], WHITE, dwell=4))
        return paths + dots

    if mod == "beat_rotate":
        ang = 0.38 * smooth_wave(c.bt, 4.0)
        ca, sa = math.cos(ang), math.sin(ang)
        out = []
        for p in paths:
            x, y = p.pts[:, 0], p.pts[:, 1]
            nx = np.clip(x * ca - y * sa, -0.98, 0.98)
            ny = np.clip(x * sa + y * ca, -0.98, 0.98)
            out.append(Path(np.stack([nx, ny], 1), p.rgb, p.dwell))
        return out

    if mod == "stereo_pan_shift":
        dx = 0.22 * smooth_wave(c.bt, 8.0)
        out = []
        for p in paths:
            pts = p.pts.copy()
            pts[:, 0] = np.clip(pts[:, 0] + dx, -0.98, 0.98)
            out.append(Path(pts, p.rgb, p.dwell))
        return out

    if mod == "kick_zoom_pump":
        pump = beat_pump(c)
        scale = 0.86 + 0.16 * pump
        out = []
        for p in paths:
            pts = np.clip(p.pts * scale, -0.98, 0.98)
            out.append(Path(pts, p.rgb, p.dwell))
        return out

    if mod == "counter_sweep_plane":
        u = beat_sweep(c.bt + 0.5, beats_per_pass=1.0, alternate=True)
        y = -0.88 + 1.76 * u
        extra = [line((-0.96, y), (0.96, y), c.palette(1, v=0.9, white_boost=0.5 * beat_pump(c)), dwell=2)]
        return paths + extra

    if mod == "hat_packet_chop":
        if c.hi < 0.35:
            return paths
        out = []
        for idx, p in enumerate(paths):
            if len(p.pts) == 2:
                out.extend(dashed_beam(p.pts[0], p.pts[1], p.rgb, segments=3, duty=0.68, phase=c.bt * 2.0 + idx * 0.25))
            else:
                out.append(p)
        return out

    return paths


# ======================================================================== SMOOTH PRESET MORPHING

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

        c0 = np.broadcast_to(p0.rgb, (v_count, 3))
        c1 = np.broadcast_to(p1.rgb, (v_count, 3))
        rgb = c0 * (1.0 - t) + c1 * t
        dwell = int(round(p0.dwell * (1.0 - t) + p1.dwell * t))
        morphed.append(Path(pts, rgb, dwell))
    return morphed


SCENES = {
    # Atmospheric Breakdown Liquid Sky & Morphing Sky presets (9 scenes)
    "liquid_sky_drift": sc_liquid_sky_drift,
    "morphing_plane_sky": sc_morphing_plane_sky,
    "breathe_horizon_sky": sc_breathe_horizon_sky,
    "velvet_sky_sweep": sc_velvet_sky_sweep,
    "silk_wave_sky": sc_silk_wave_sky,
    "dual_horizon_mirage": sc_dual_horizon_mirage,
    "floating_dot_constellation": sc_floating_dot_constellation,
    "slow_v_canopy_sky": sc_slow_v_canopy_sky,
    "aurora_sky_curtain": sc_aurora_sky_curtain,
    # Main & Drop presets (19 scenes)
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
    # Build presets (6 scenes)
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

# BPM-locked rhythmic swap sequences (1 = primary scene/colour, 2 = partner scene/colour)
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
    """Switches scenes every 8 or 16 beats with BPM-locked effect swaps (instant cut or morphed, chosen
    at the start of the swap sequence) OR colour swaps (1-1-1-2, 1-2-1-2, 1-1-2-2), varied 1-2 beat white
    drop impact explosions, 1-beat piano/synth white breakdown accents, and a secondary modifier layer."""

    def __init__(self, seed=1, force_scene=None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene, self.prev_scene = None, None, None
        self.swap_mode: str | None = None          # "effect", "color", or None
        self.swap_instant: bool = False            # Decided once at start of swap sequence
        self.swap_scene: str | None = None
        self.swap_hue_offset: float = 0.2
        self.swap_seq_name: str | None = None
        self.swap_beats_total: int = 0
        self._swap_start_beat: int = 0
        self._active_slot: int = 1
        self._slot_morph_left: float = 0.0

        # Drop impact white explosion timer & variant
        self._drop_impact_left: float = 0.0
        self._drop_impact_variant: int = 0
        self._drop_impact_bag: list[int] = []

        # Piano/synth melodic onset detector for 1-beat white breakdown accents
        self._melodic_avg: float = 0.0
        self._prev_melodic: float = 0.0
        self._synth_accent_age: float = 99.0
        self._synth_accent_kind: int = 0

        self.modifier: str | None = None
        self._morph_left = 0.0
        self.hue, self.bars, self._last_beat = 0.0, 0, 0
        self.rt, self._last_bt, self.strobe_enabled, self.detail = 0.0, 0.0, True, 0.5
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
        """Configures an 8- or 16-beat BPM-locked EFFECT swap (instant or morphed) or COLOUR swap phrase."""
        self.swap_mode = None
        self.swap_instant = False
        self.swap_scene = None
        self.swap_seq_name = None
        self.swap_beats_total = 0
        self._active_slot = 1
        self._slot_morph_left = 0.0

        # Keep breakdowns slow and unbroken; run effect/colour swaps in MAIN and DROP
        if self.section not in (MAIN, DROP):
            return

        swap_chance = 0.70 if self.section == DROP else 0.55
        if float(self.rng.random()) > swap_chance:
            return

        if not self._seq_bag:
            self._seq_bag = list(SWAP_SEQ_NAMES)
            self.rng.shuffle(self._seq_bag)
        seq_name = self._seq_bag.pop(0)

        # Decide once at the start of the swap sequence whether transitions are instant cuts or morphed
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

        bt = f.beat_count + f.beat_phase
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

        # Detect piano / synth chord or pluck transients during BREAKDOWN for 1-beat subtle white accent
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

        # First 1-2 beats of DROP: varied solo pure-white impact effect before switching into regular patterns
        if f.section == DROP and self._drop_impact_left > 0.0:
            c_drop = Ctx(f, bt, base_hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
            progress = 1.0 - (self._drop_impact_left / DROP_IMPACT_BEATS)
            paths = _drop_impact_effect(c_drop, progress, self._drop_impact_variant)
            if progress > 0.78:
                blend = (progress - 0.78) / 0.22
                paths = _morph_paths(paths, SCENES[self.scene](c_drop), blend)
            return paths, DROP_IMPACT_NAMES[self._drop_impact_variant]

        # Evaluate BPM-locked swap phrase (either "effect" swap or "color" swap)
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
            else:  # self.swap_mode == "effect"
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

        # Phrase-boundary smooth morph from previous scene
        if self._morph_left > 0.0 and self.prev_scene in SCENES:
            alpha = 1.0 - (self._morph_left / MORPH_BEATS)
            prev_paths = SCENES[self.prev_scene](c)
            paths = _morph_paths(prev_paths, paths, alpha)

        paths = _apply_modifier(paths, self.modifier, c)

        # Add subtle 1-beat white accent line or dot burst on piano/synth hits during BREAKDOWN
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
        return paths, label
