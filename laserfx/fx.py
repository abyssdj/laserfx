"""Laser effects: 26 phrase-locked beam scenes, smooth preset morphing, and secondary beat-synced modifiers.

Coordinates are in [-1, 1]; colours are RGB in 0..1.
Designed for a single output feeding twin projectors where Projector 2 is X-mirrored (x -> -x):
- Deliberate X-offsets and asymmetric tilts allow Unit A and X-mirrored Unit B to sweep across
  each other in wide stereo symmetry without overlapping onto identical beams.
- Patterns change cleanly every 8 or 16 beats using a non-repeating shuffle bag, with 1.5-beat
  smooth geometric morphing between presets and a secondary beat-synced modifier layer.
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

import numpy as np

from .audio import BREAKDOWN, BUILD, DROP, MAIN, SILENCE, Features

TAU = 2 * math.pi


@dataclass
class Path:
    pts: np.ndarray            # (V, 2)
    rgb: np.ndarray            # (3,) or (V, 3)
    dwell: int = 0             # extra points held at vertices (sharp corners / bright aerial tips)


def hsv(h, s=1.0, v=1.0):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)))


def line(a, b, rgb, dwell=2):
    return Path(np.array([a, b], float), rgb, dwell)


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


# ---------------------------------------------------------------- smooth musical motion helpers
def smooth_wave(bt: float, period_beats: float) -> float:
    """Smooth -1..+1 cosine wave locked to exact beat multiples (4, 8, 16 beats)."""
    return -math.cos(bt * TAU / period_beats)


def glide_step(bt: float, step_beats: float = 2.0, smoothness: float = 4.5) -> float:
    """Smoothly glides to a new beat-locked angle every `step_beats` without jarring teleportation."""
    x = bt / step_beats
    idx = math.floor(x)
    frac = x - idx
    eased = 1.0 - math.exp(-smoothness * frac) * (1.0 - frac)
    return idx + eased


def beat_pump(c: Ctx) -> float:
    """Smooth musical punch combining beat_pulse and kick."""
    return min(max(0.55 * c.pulse + 0.45 * c.kick, 0.0), 1.0)


def snare_swell(c: Ctx) -> float:
    """Smooth snare/clap accent envelope."""
    return min(max(getattr(c.f, "snare", 0.0), 0.0), 1.0)


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
        p_idx = int(abs(self.hue * 10)) % len(PALETTES)
        h_pri, h_acc, s_acc = PALETTES[p_idx]
        if idx % 2 == 0:
            return hsv(h_pri, max(1.0 - white_boost, 0.0), v)
        return hsv(h_acc, max(s_acc - white_boost, 0.0), v)


# ======================================================================== MAIN SECTION SCENES (8)

def sc_sweeping_curtain(c: Ctx):
    """Wide, fluid 8-beat sweeping beam fan that breathes open on every kick."""
    pump = beat_pump(c)
    n = c.count(6, 10, max(c.energy, 0.45))
    aim = 0.24 + 0.52 * smooth_wave(c.bt, 8.0)
    spread = 0.32 + 0.24 * pump
    ox = -0.58

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.5)
        is_edge = (i == 0 or i == n - 1)
        col = c.palette(1 if is_edge else (i % 2), v=1.0, white_boost=0.55 * pump if is_edge else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


def sc_liquid_horizon(c: Ctx):
    """Smoothly tilting 'Liquid Sky' ceiling sheet that rocks to the 4-beat bar and pulses on kicks."""
    pump = beat_pump(c)
    planes = c.count(2, 4, max(c.energy, 0.4))
    tilt = 0.42 * smooth_wave(c.bt, 8.0)
    y_center = 0.28 * smooth_wave(c.bt, 16.0)

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.14 + 0.16 * pump)
        y0 = float(np.clip(y_center + gap - tilt, -0.94, 0.94))
        y1 = float(np.clip(y_center + gap + tilt, -0.94, 0.94))
        col = c.palette(p, v=0.9 + 0.1 * pump, white_boost=0.45 * pump if p == 0 else 0.0)
        out.append(line(*_zig(p, (-0.96, y0), (0.96, y1)), col, dwell=2))
    return out


def sc_twin_scissor_sweep(c: Ctx):
    """Opposing floor and ceiling beam fans gliding smoothly in counter-phase."""
    pump = beat_pump(c)
    snare = snare_swell(c)
    n = c.count(4, 7, max(c.energy, 0.45))
    swing = 0.46 * smooth_wave(c.bt, 8.0)

    out = []
    idx = 0
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = 0.34 + swing + u * (0.30 + 0.18 * pump)
        p0, p1 = ray(-0.78, -0.96, ang, 2.5)
        out.append(line(*_zig(idx, p0, p1), c.palette(0, v=1.0), dwell=1))
        idx += 1

    for i in range(max(3, n - 1)):
        u = (i / max(n - 2, 1)) - 0.5
        ang = math.pi - (0.36 - swing * 0.85 + u * (0.28 + 0.18 * snare))
        p0, p1 = ray(-0.28, 0.96, ang, 2.5)
        out.append(line(*_zig(idx, p0, p1), c.palette(1, v=0.85 + 0.15 * snare, white_boost=0.4 * snare), dwell=1))
        idx += 1
    return out


def sc_chevron_wave(c: Ctx):
    """Fluid aerial V-wedges that glide vertically and open/close smoothly with the bass."""
    pump = beat_pump(c)
    layers = c.count(3, 5, max(c.energy, 0.45))
    morph = smooth_wave(c.bt, 8.0)
    cx = -0.36 * math.cos(c.bt * TAU / 16.0)
    wing_w = 0.50 + 0.16 * pump

    out = []
    for i in range(layers):
        u = i / max(layers - 1, 1)
        apex_y = -0.35 + 0.70 * u + 0.18 * smooth_wave(c.bt, 4.0)
        wing_y = float(np.clip(apex_y + morph * (0.36 + 0.18 * pump), -0.95, 0.95))
        apex_y = float(np.clip(apex_y, -0.95, 0.95))
        w = wing_w * (0.45 + 0.55 * u)

        pts = np.array([[max(cx - w, -0.96), wing_y], [cx, apex_y], [min(cx + w, 0.96), wing_y]], float)
        if i % 2 == 1:
            pts = pts[::-1]
        col = c.palette(i, v=0.65 + 0.35 * (1.0 - u), white_boost=0.4 * pump if i == 0 else 0.0)
        out.append(Path(pts, col, dwell=3))
    return out


def sc_apex_corridor(c: Ctx):
    """Twin off-axis perspective beam corridors that glide smoothly across the venue."""
    pump = beat_pump(c)
    n = c.count(4, 6)
    vp_x = -0.40 + 0.18 * smooth_wave(c.bt, 16.0)
    vp_y = 0.20 * smooth_wave(c.bt, 8.0)
    shift = (c.bt * 0.5) % 1.0
    rot = (math.pi / 4.0) + 0.25 * smooth_wave(c.bt, 16.0)

    out = []
    for idx in range(n):
        u = (idx + shift) / n
        r = min(0.06 + 0.54 * (u ** 1.5), 0.62) * (1.0 + 0.07 * pump)
        cx = vp_x * (1.0 - u * 0.4)
        cy = vp_y * (1.0 - u * 0.4)
        a = rot + np.arange(5) * TAU / 4.0
        pts = np.stack([cx + r * np.sin(a), cy + r * np.cos(a)], 1)
        col = c.palette(idx, v=min(1.0, u * 2.0), white_boost=0.4 * pump if idx == n - 1 else 0.0)
        out.append(Path(pts, col, dwell=3))
    return out


def sc_rolling_waves(c: Ctx):
    """Smooth undulating laser wave ribbon that rolls continuously across both projectors."""
    pump = beat_pump(c)
    waves = c.count(2, 3)
    x = np.linspace(-0.95, 0.95, 56)
    y_offset = 0.32 * smooth_wave(c.bt, 16.0)
    amp = 0.18 + 0.16 * pump

    out = []
    for w in range(waves):
        phase = c.bt * TAU / 4.0 + w * 1.1
        y = np.clip(y_offset + (w - 0.5) * 0.24 + amp * np.sin(2.2 * math.pi * x + phase), -0.94, 0.94)
        pts = np.stack([x, y], 1)
        if w % 2 == 1:
            pts = pts[::-1]
        out.append(Path(pts, c.palette(w, v=1.0, white_boost=0.35 * pump if w == 0 else 0.0), dwell=0))
    return out


def sc_helix_ribbon(c: Ctx):
    """Intertwined double-helix laser ribbon drifting diagonally across the stereo field."""
    pump = beat_pump(c)
    t = np.linspace(-0.92, 0.92, 56)
    phase = c.bt * TAU / 6.0
    amp = 0.22 + 0.14 * pump
    out = []
    for strand in (0, 1):
        w = amp * np.sin(2.5 * math.pi * t + phase + strand * math.pi)
        # Diagonal axis ensures X-mirrored projector forms a rich woven lattice
        x = np.clip(t * 0.85 - 0.15 + w * 0.35, -0.95, 0.95)
        y = np.clip(t * 0.72 + w * 0.85, -0.95, 0.95)
        pts = np.stack([x, y], 1)
        if strand == 1:
            pts = pts[::-1]
        out.append(Path(pts, c.palette(strand, v=1.0, white_boost=0.35 * pump if strand == 0 else 0.0), dwell=0))
    return out


def sc_prism_pendulum(c: Ctx):
    """Twin suspended beam arrays swinging like synchronized pendulums from the upper rig."""
    pump = beat_pump(c)
    n = c.count(5, 8, max(c.energy, 0.45))
    swing = math.pi + 0.48 * smooth_wave(c.bt, 8.0)
    spread = 0.28 + 0.22 * pump
    ox = -0.52

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(ox, 0.96, swing + u * 2.0 * spread, 2.5)
        col = c.palette(i % 2, v=1.0, white_boost=0.5 * pump if i == n // 2 else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


# ======================================================================== DROP SECTION SCENES (8)

def sc_drop_power_sweep(c: Ctx):
    """Fast 4-beat sweeping power fan that whips smoothly from side to side and flares white on kicks."""
    pump = beat_pump(c)
    n = c.count(6, 10, max(c.energy, 0.5))
    aim = 0.25 + 0.62 * smooth_wave(c.bt, 4.0)
    spread = 0.28 + 0.34 * pump
    ox = -0.62

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.5)
        is_lead = (i == 0 or i == n - 1 or i == n // 2)
        col = c.palette(1 if is_lead else (i % 2), v=1.0, white_boost=0.85 * pump if is_lead else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
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


def sc_drop_fan_pump(c: Ctx):
    """Dual-origin sweeping fans (floor + ceiling) that pump wide open on every kick."""
    pump = beat_pump(c)
    n = c.count(4, 7)
    glide = glide_step(c.bt, step_beats=2.0, smoothness=5.0)
    aim = 0.28 + 0.42 * math.sin(glide * math.pi * 0.5)
    spread = 0.20 + 0.42 * pump

    out = []
    idx = 0
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(-0.75, -0.96, aim + u * 2.0 * spread, 2.5)
        out.append(line(*_zig(idx, p0, p1), c.palette(0, white_boost=0.7 * pump), dwell=2))
        idx += 1
    for i in range(max(3, n - 1)):
        u = (i / max(n - 2, 1)) - 0.5
        p0, p1 = ray(-0.25, 0.96, math.pi - (aim + u * 2.0 * spread), 2.5)
        out.append(line(*_zig(idx, p0, p1), c.palette(1, white_boost=0.7 * pump), dwell=2))
        idx += 1
    return out


def sc_drop_vertical_sweep(c: Ctx):
    """Sweeping vertical beam wall that glides back and forth across the room every 4 beats."""
    pump = beat_pump(c)
    n = c.count(5, 8)
    x_center = -0.50 + 0.34 * smooth_wave(c.bt, 4.0)
    width = 0.22 + 0.24 * pump

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        x = float(np.clip(x_center + u * width, -0.96, 0.96))
        col = c.palette(i % 2, v=1.0, white_boost=0.85 * pump if i == n // 2 else 0.0)
        out.append(line(*_zig(i, (x, -0.95), (x, 0.95)), col, dwell=2))
    return out


def sc_drop_arch_pulse(c: Ctx):
    """Sweeping cathedral arch beams that fan outward from the corners on every beat."""
    pump = beat_pump(c)
    n = c.count(5, 9)
    sweep = 0.35 + 0.42 * smooth_wave(c.bt, 4.0)
    spread = 0.28 + 0.26 * pump

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = sweep + u * 2.0 * spread
        p0, p1 = ray(-0.92, -0.96, ang, 2.6)
        col = c.palette(i % 2, v=1.0, white_boost=0.75 * pump if i == 0 else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


def sc_drop_saw_slash(c: Ctx):
    """Relentless 2-beat unidirectional sawtooth laser slash across the venue."""
    pump = beat_pump(c)
    n = c.count(5, 8)
    ramp = (c.bt / 2.0) % 1.0
    aim = -0.35 + 1.15 * (ramp ** 0.8)
    spread = 0.20 + 0.24 * pump

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(-0.65, -0.96, aim + u * 2.0 * spread, 2.5)
        col = c.palette(i % 2, v=1.0, white_boost=0.8 * pump if i == n - 1 else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


def sc_drop_laser_cage(c: Ctx):
    """Intersecting horizontal & vertical laser cage bars that shift smoothly on beats."""
    pump = beat_pump(c)
    nh = c.count(2, 4)
    nv = c.count(2, 4)
    y_shift = 0.26 * smooth_wave(c.bt, 4.0)
    x_shift = -0.48 + 0.28 * smooth_wave(c.bt, 4.0)

    out = []
    idx = 0
    for i in range(nh):
        y = float(np.clip(y_shift + (i - (nh - 1) / 2.0) * (0.34 + 0.14 * pump), -0.92, 0.92))
        out.append(line(*_zig(idx, (-0.95, y - 0.12), (0.95, y + 0.12)), c.palette(0, white_boost=0.6 * pump), dwell=2))
        idx += 1
    for j in range(nv):
        x = float(np.clip(x_shift + (j - (nv - 1) / 2.0) * (0.28 + 0.12 * pump), -0.94, 0.94))
        out.append(line(*_zig(idx, (x, -0.95), (x, 0.95)), c.palette(1, white_boost=0.6 * pump), dwell=2))
        idx += 1
    return out


def sc_drop_double_whip(c: Ctx):
    """Twin parallel sweeping fans originating from separate floor points, sweeping in tight unison."""
    pump = beat_pump(c)
    n = c.count(3, 5)
    aim = 0.22 + 0.52 * smooth_wave(c.bt, 4.0)
    spread = 0.20 + 0.24 * pump

    out = []
    idx = 0
    for f_idx, ox in enumerate((-0.82, -0.28)):
        for i in range(n):
            u = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(ox, -0.96, aim + u * 2.0 * spread, 2.5)
            col = c.palette(f_idx, v=1.0, white_boost=0.75 * pump if i == n // 2 else 0.0)
            out.append(line(*_zig(idx, p0, p1), col, dwell=2))
            idx += 1
    return out


# ======================================================================== BREAKDOWN SECTION SCENES (6)

def sc_sky_canopy(c: Ctx):
    """Ultra-smooth 16-beat overhead canopy sweep across the venue."""
    v = 0.72 + 0.25 * c.energy
    n = c.count(6, 10)
    aim = 0.28 + 0.42 * smooth_wave(c.bt, 16.0)
    spread = 0.46 + 0.18 * smooth_wave(c.bt, 8.0)

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(-0.62, -0.96, ang, 2.5)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2, v=v), dwell=1))
    return out


def sc_slow_horizon_drift(c: Ctx):
    """Slowly raising and lowering dual Liquid Sky planes that drift over 16 beats."""
    planes = c.count(2, 4)
    tilt = 0.28 * smooth_wave(c.bt, 16.0)
    y_base = 0.38 * smooth_wave(c.bt, 32.0)

    out = []
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * 0.22
        y0 = float(np.clip(y_base + gap - tilt, -0.94, 0.94))
        y1 = float(np.clip(y_base + gap + tilt, -0.94, 0.94))
        col = c.palette(p, v=0.85)
        out.append(line(*_zig(p, (-0.96, y0), (0.96, y1)), col, dwell=2))
    return out


def sc_cathedral_beams(c: Ctx):
    """Wide corner-to-ceiling diagonal rays that sweep gently like searchlights."""
    n = c.count(5, 8)
    sweep = 0.45 + 0.28 * smooth_wave(c.bt, 16.0)
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(-0.92, -0.96, sweep + u * 0.65, 2.6)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2, v=0.80), dwell=1))
    return out


def sc_aurora_ribbon(c: Ctx):
    """Slow, silky multi-layered sine waves drifting across the air."""
    x = np.linspace(-0.95, 0.95, 60)
    out = []
    for layer in range(c.count(2, 3)):
        ph = c.bt * TAU / 12.0 + layer * 1.4
        y = 0.35 * smooth_wave(c.bt, 24.0) + 0.22 * np.sin(1.8 * math.pi * x + ph) + (layer - 0.5) * 0.18
        pts = np.stack([x, np.clip(y, -0.94, 0.94)], 1)
        if layer % 2 == 1:
            pts = pts[::-1]
        out.append(Path(pts, c.palette(layer, v=0.82), dwell=0))
    return out


def sc_halo_drift(c: Ctx):
    """Twin off-center sweeping searchlight arcs floating smoothly across the upper canopy."""
    n = c.count(4, 7)
    sweep = math.pi + 0.35 * smooth_wave(c.bt, 16.0)
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(-0.48, 0.96, sweep + u * 0.72, 2.5)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2, v=0.78), dwell=1))
    return out


def sc_velvet_curtain(c: Ctx):
    """Slowly breathing vertical laser sheet drifting horizontally across each projector's half."""
    n = c.count(4, 7)
    x_center = -0.50 + 0.30 * smooth_wave(c.bt, 16.0)
    width = 0.34 + 0.14 * smooth_wave(c.bt, 8.0)
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        x = float(np.clip(x_center + u * width, -0.95, 0.95))
        out.append(line(*_zig(i, (x, -0.95), (x, 0.95)), c.palette(i % 2, v=0.76), dwell=1))
    return out


# ======================================================================== BUILD SECTION SCENES (4)

def sc_build_converge_sweep(c: Ctx):
    """Smoothly accelerating fan sweep that narrows into a razor-bright center beam at the drop."""
    b = c.f.build
    n = c.count(6, 11, max(b, 0.45))
    sweep = (0.50 * (1.0 - 0.65 * (b ** 2))) * math.sin(c.rt * TAU / 4.0)
    bias = 0.34 * (1.0 - 0.7 * b)
    spread = 0.50 * (1.0 - 0.78 * (b ** 2)) + 0.05
    ox = -0.60 * (1.0 - 0.65 * b)

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = bias + sweep + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.5)
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.85 * (b ** 2))
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
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


def sc_build_cross_accel(c: Ctx):
    """Accelerating opposing corner sweeps that cross faster as the drop approaches."""
    b = c.f.build
    n = c.count(4, 7)
    swing = 0.45 * math.sin(c.rt * TAU / 3.5)
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(-0.85, -0.96, 0.42 + swing + u * 0.35, 2.5)
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.75 * (b ** 2))
        out.append(line(*_zig(i, p0, p1), col, dwell=1))
    return out


def sc_build_prism_tighten(c: Ctx):
    """Twin vertical beam walls compressing inward toward the center line as the drop approaches."""
    b = c.f.build
    n = c.count(4, 8)
    x_center = -0.68 * (1.0 - 0.85 * (b ** 1.5))
    width = 0.48 * (1.0 - 0.82 * (b ** 2)) + 0.04
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        x = float(np.clip(x_center + u * width, -0.95, 0.95))
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.85 * (b ** 2))
        out.append(line(*_zig(i, (x, -0.95), (x, 0.95)), col, dwell=2))
    return out


# ======================================================================== SECONDARY OVERLAY LAYER
# Periodically triggers alongside the main scene for beat-synced rotation, stereo shift, or overlays

def _apply_modifier(paths: list[Path], mod: str, c: Ctx) -> list[Path]:
    if not mod or not paths:
        return paths

    if mod == "beat_rotate":
        # Smooth beat-synced angular tilt rock (±22 deg on 4-beat cycle)
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
        # Smoothly shifts X-offset across 8 beats so mirrored twin projectors sweep past each other
        dx = 0.24 * smooth_wave(c.bt, 8.0)
        out = []
        for p in paths:
            pts = p.pts.copy()
            pts[:, 0] = np.clip(pts[:, 0] + dx, -0.98, 0.98)
            out.append(Path(pts, p.rgb, p.dwell))
        return out

    if mod == "kick_zoom_pump":
        # Smooth radial breathing expansion on bass kicks
        pump = beat_pump(c)
        scale = 0.86 + 0.16 * pump
        out = []
        for p in paths:
            pts = np.clip(p.pts * scale, -0.98, 0.98)
            out.append(Path(pts, p.rgb, p.dwell))
        return out

    if mod == "accent_ceiling_ray":
        # Adds a sparse counter-sweeping overhead accent fan in the secondary festival color
        pump = beat_pump(c)
        swing = math.pi - (0.35 + 0.45 * smooth_wave(c.bt, 8.0))
        extra = []
        for i in range(3):
            u = (i - 1) * (0.22 + 0.12 * pump)
            p0, p1 = ray(-0.22, 0.96, swing + u, 2.4)
            extra.append(line(*_zig(i, p0, p1), c.palette(1, v=0.85, white_boost=0.45 * pump), dwell=1))
        return paths + extra

    if mod == "hat_packet_chop":
        # Slices straight 2-point beams into gliding light packets when hi-hat flux is active
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
    """Resamples a polyline to `k` evenly spaced vertices so arbitrary paths can morph smoothly."""
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
    """Smoothly interpolates geometry and color between outgoing and incoming scenes."""
    if not prev_paths:
        return next_paths
    if not next_paths:
        return prev_paths
    # Smoothstep S-curve for fluid galvo transitions
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
    # MAIN (8)
    "sweeping_curtain": sc_sweeping_curtain,
    "liquid_horizon": sc_liquid_horizon,
    "twin_scissor_sweep": sc_twin_scissor_sweep,
    "chevron_wave": sc_chevron_wave,
    "apex_corridor": sc_apex_corridor,
    "rolling_waves": sc_rolling_waves,
    "helix_ribbon": sc_helix_ribbon,
    "prism_pendulum": sc_prism_pendulum,
    # DROP (8)
    "drop_power_sweep": sc_drop_power_sweep,
    "drop_x_blades": sc_drop_x_blades,
    "drop_fan_pump": sc_drop_fan_pump,
    "drop_vertical_sweep": sc_drop_vertical_sweep,
    "drop_arch_pulse": sc_drop_arch_pulse,
    "drop_saw_slash": sc_drop_saw_slash,
    "drop_laser_cage": sc_drop_laser_cage,
    "drop_double_whip": sc_drop_double_whip,
    # BREAKDOWN (6)
    "sky_canopy": sc_sky_canopy,
    "slow_horizon_drift": sc_slow_horizon_drift,
    "cathedral_beams": sc_cathedral_beams,
    "aurora_ribbon": sc_aurora_ribbon,
    "halo_drift": sc_halo_drift,
    "velvet_curtain": sc_velvet_curtain,
    # BUILD (4)
    "build_converge_sweep": sc_build_converge_sweep,
    "build_rising_sheet": sc_build_rising_sheet,
    "build_cross_accel": sc_build_cross_accel,
    "build_prism_tighten": sc_build_prism_tighten,
    # Legacy aliases for CLI / saved sessions
    "tunnel": sc_apex_corridor,
    "tunnel_fast": sc_drop_power_sweep,
    "beam_fan": sc_sweeping_curtain,
    "starburst": sc_drop_fan_pump,
    "starburst_fast": sc_drop_power_sweep,
    "lissajous": sc_helix_ribbon,
    "scope": sc_rolling_waves,
    "spiral": sc_chevron_wave,
    "spiral_slow": sc_aurora_ribbon,
    "grid": sc_drop_laser_cage,
    "drop_blast": sc_drop_double_whip,
    "breath": sc_sky_canopy,
    "riser": sc_build_converge_sweep,
    "ring_pulse": sc_apex_corridor,
    "sky_fan": sc_sky_canopy,
    "sky_cross": sc_cathedral_beams,
    "scan_sweep": sc_slow_horizon_drift,
    "build_fan": sc_build_converge_sweep,
    "build_scan": sc_build_rising_sheet,
    "strobe_poly": sc_drop_x_blades,
    "strobe_beams": sc_drop_vertical_sweep,
    "strobe_burst": sc_drop_arch_pulse,
}

POOLS = {
    MAIN: [
        "sweeping_curtain",
        "liquid_horizon",
        "twin_scissor_sweep",
        "chevron_wave",
        "apex_corridor",
        "rolling_waves",
        "helix_ribbon",
        "prism_pendulum",
    ],
    DROP: [
        "drop_power_sweep",
        "drop_x_blades",
        "drop_fan_pump",
        "drop_vertical_sweep",
        "drop_arch_pulse",
        "drop_saw_slash",
        "drop_laser_cage",
        "drop_double_whip",
    ],
    BREAKDOWN: [
        "sky_canopy",
        "slow_horizon_drift",
        "cathedral_beams",
        "aurora_ribbon",
        "halo_drift",
        "velvet_curtain",
    ],
    BUILD: [
        "build_converge_sweep",
        "build_rising_sheet",
        "build_cross_accel",
        "build_prism_tighten",
    ],
    SILENCE: [],
}

MODIFIER_POOLS = {
    MAIN: ["beat_rotate", "stereo_pan_shift", "kick_zoom_pump", "accent_ceiling_ray", "hat_packet_chop"],
    DROP: ["beat_rotate", "kick_zoom_pump", "accent_ceiling_ray", "hat_packet_chop", "stereo_pan_shift"],
    BREAKDOWN: ["beat_rotate", "stereo_pan_shift", "accent_ceiling_ray"],
    BUILD: ["kick_zoom_pump"],
    SILENCE: [],
}

# 4 beats per bar:
# - MAIN & BREAKDOWN switch every 4 bars = 16 beats
# - DROP & BUILD switch every 2 bars = 8 beats
BARS_PER_SCENE = {MAIN: 4, DROP: 2, BREAKDOWN: 4, BUILD: 2, SILENCE: 99}
MORPH_BEATS = 1.5


class FXEngine:
    """Switches scenes every 8 or 16 beats with smooth morph transitions and a secondary modifier layer."""

    def __init__(self, seed=1, force_scene=None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene, self.prev_scene = None, None, None
        self.modifier: str | None = None
        self._morph_left = 0.0
        self.hue, self.bars, self._last_beat = 0.0, 0, 0
        self.rt, self._last_bt, self.strobe_enabled, self.detail = 0.0, 0.0, True, 0.5
        self._bags: dict[str, list[str]] = {}
        self._mod_bags: dict[str, list[str]] = {}

    def _pick_modifier(self):
        if self.force:
            self.modifier = None
            return
        pool = MODIFIER_POOLS.get(self.section, [])
        # Trigger the secondary modifier ~55% of phrases in MAIN/DROP, ~35% in BREAKDOWN/BUILD
        chance = 0.55 if self.section in (MAIN, DROP) else 0.35
        if not pool or float(self.rng.random()) > chance:
            self.modifier = None
            return
        bag = self._mod_bags.get(self.section, [])
        if not bag:
            bag = list(pool)
            self.rng.shuffle(bag)
        self.modifier = bag.pop(0)
        self._mod_bags[self.section] = bag

    def _pick(self):
        self.prev_scene = self.scene
        if self.force:
            self.scene = self.force
            self._pick_modifier()
            return
        pool = POOLS.get(self.section, [])
        if not pool:
            self.scene = None
            self.modifier = None
            return
        bag = self._bags.get(self.section, [])
        if not bag:
            bag = list(pool)
            self.rng.shuffle(bag)
            if len(bag) > 1 and bag[0] == self.scene:
                bag[0], bag[-1] = bag[-1], bag[0]
        self.scene = bag.pop(0)
        self._bags[self.section] = bag
        self._pick_modifier()
        if self.prev_scene and self.prev_scene != self.scene:
            self._morph_left = MORPH_BEATS

    def update(self, f: Features):
        if f.section != self.section:
            self.section, self.bars = f.section, 0
            self._pick()

        if f.beat_count != self._last_beat:
            if f.bar_pos == 0:
                self.bars += 1
                if self.bars % 4 == 0:
                    self.hue += 0.1
                if self.bars % BARS_PER_SCENE[self.section] == 0:
                    self._pick()
            self._last_beat = f.beat_count

        hue = self.hue
        if f.section == DROP:
            hue = [0.0, 0.1, 0.2, 0.4][(f.beat_count // 16) % 4]
        elif f.section == BREAKDOWN:
            hue = 0.0

        bt = f.beat_count + f.beat_phase
        dbt = bt - self._last_bt
        self._last_bt = bt
        if 0 < dbt < 1:
            self.rt += dbt * (1 + 4 * f.build if f.section == BUILD else 1)
            if self._morph_left > 0.0:
                self._morph_left = max(0.0, self._morph_left - dbt)

        c = Ctx(f, bt, hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
        if not self.scene or f.section == SILENCE:
            return [], None

        paths = SCENES[self.scene](c)
        if self._morph_left > 0.0 and self.prev_scene in SCENES:
            alpha = 1.0 - (self._morph_left / MORPH_BEATS)
            prev_paths = SCENES[self.prev_scene](c)
            paths = _morph_paths(prev_paths, paths, alpha)

        paths = _apply_modifier(paths, self.modifier, c)
        label = f"{self.scene}+{self.modifier}" if self.modifier else self.scene
        return paths, label
