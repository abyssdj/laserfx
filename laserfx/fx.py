"""Laser effects: pure modern aerial beam choreography (no 90s shapes, tunnels, or concentric rings).

Coordinates are in [-1, 1]; colours are RGB in 0..1.
Designed for a single output feeding twin projectors where Projector 2 is X-mirrored (x -> -x):
- Off-axis origins and asymmetric tilts ensure Unit A and X-mirrored Unit B slice across each other
  in mid-air rather than overlapping onto identical beam paths.
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
    dwell: int = 0             # extra points held at each vertex (bright aerial beam endpoints)


def hsv(h, s=1.0, v=1.0):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)))


def line(a, b, rgb, dwell=2):
    return Path(np.array([a, b], float), rgb, dwell)


def dashed_beam(a, b, rgb, segments=4, duty=0.55):
    """Chops a beam into high-contrast aerial pulses (grating / segmented beam look)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = []
    for i in range(segments):
        t0 = i / segments
        t1 = t0 + duty / segments
        p0 = a + (b - a) * t0
        p1 = a + (b - a) * min(t1, 1.0)
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


# ---------------------------------------------------------------- snappy modern motion curves
def whip(bt: float, div: float = 1.0, speed: float = 11.0) -> float:
    """Fast galvo whip: slams to the next beat target in the first ~20% of the beat, then locks."""
    x = bt * div
    idx = math.floor(x)
    frac = x - idx
    return idx + (1.0 - math.exp(-speed * frac))


def saw(bt: float, beats: float = 2.0) -> float:
    """Linear 0->1 sawtooth ramp with instant flyback reset (no slow back-and-forth pendulum)."""
    return (bt / beats) % 1.0


def kick_env(c: Ctx) -> float:
    """Tight transient punch from kick/beat pulse."""
    return min(max(max(c.kick, c.pulse * 0.8) ** 1.3, 0.0), 1.0)


def gate(c: Ctx, div: float = 2.0, duty: float = 0.5) -> bool:
    return (not c.strobe) or ((c.bt * div) % 1.0) < duty


def _zig(i: int, a, b):
    return (a, b) if i % 2 == 0 else (b, a)


# High-contrast modern festival palettes: (primary_hue, accent_hue, accent_sat)
PALETTES = [
    (0.54, 0.83, 0.85),   # Ice Cyan + Electric Magenta
    (0.00, 0.56, 0.12),   # Pure Red + Crisp White
    (0.76, 0.11, 0.90),   # Deep UV + Solar Amber
    (0.36, 0.55, 0.15),   # Acid Green + Ice White
    (0.62, 0.04, 0.95),   # Deep Blue + Laser Orange
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

    def count(self, lo, hi, x=1.0):
        return int(round(lo + (hi - lo) * min(max(self.detail * x, 0.0), 1.0)))

    def palette(self, idx: int = 0, v: float = 1.0, white_boost: float = 0.0):
        p_idx = int(abs(self.hue * 10)) % len(PALETTES)
        h_pri, h_acc, s_acc = PALETTES[p_idx]
        if idx % 2 == 0:
            return hsv(h_pri, max(1.0 - white_boost, 0.0), v)
        return hsv(h_acc, max(s_acc - white_boost, 0.0), v)


# ------------------------------------------------------------------------ MODERN BEAM SCENES

def sc_whip_fan(c: Ctx):
    """Fast on-beat beam whip: slams across the venue on every beat, locking into sharp angles."""
    k = kick_env(c)
    n = c.count(5, 9, max(c.energy, 0.45))
    # Every beat whips rapidly to a contrasting angle
    step = whip(c.bt, div=1.0, speed=12.0)
    beat_idx = int(c.bt)
    targets = [-0.28, 0.62, -0.05, 0.78, -0.35, 0.55, 0.15, 0.85]
    a_prev = targets[(beat_idx - 1) % len(targets)]
    a_curr = targets[beat_idx % len(targets)]
    frac = step - math.floor(step)
    aim = a_prev + (a_curr - a_prev) * frac

    # Fan opens wide on the kick transient, snaps tight between kicks
    spread = 0.14 + 0.44 * k
    ox = -0.62

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.5)
        is_lead = (i == 0 or i == n - 1)
        col = c.palette(1 if is_lead else (i % 2), v=1.0, white_boost=0.8 * k if is_lead else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


def sc_saw_sweep(c: Ctx):
    """Relentless 2-beat sawtooth sweep: slashes rapidly left-to-right, blanks, and snaps back."""
    k = kick_env(c)
    n = c.count(4, 8, max(c.energy, 0.4))
    ramp = saw(c.bt, beats=2.0)
    # Rapid unidirectional sweep from far-left (-0.45 rad) to far-right (+0.85 rad)
    aim = -0.45 + 1.30 * (ramp ** 0.75)
    spread = 0.22 + 0.25 * k
    ox = -0.58

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.5)
        col = c.palette(i % 2, v=1.0, white_boost=0.6 * k if i == n - 1 else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


def sc_beam_chase(c: Ctx):
    """16th-note sequential beam chase: steps rapidly through discrete beam positions across the room."""
    n_slots = c.count(6, 10, max(c.energy, 0.5))
    # 4 steps per beat (16th notes)
    step_16th = int(c.bt * 4)
    # Light up a tight cluster of 2–3 adjacent beams chasing back and forth
    cycle = 2 * n_slots - 2
    pos = step_16th % max(cycle, 1)
    if pos >= n_slots:
        pos = cycle - pos

    ox = -0.60
    out = []
    idx = 0
    for offset in range(-1, 2):
        slot = pos + offset
        if 0 <= slot < n_slots:
            u = (slot / max(n_slots - 1, 1)) - 0.5
            ang = 0.28 + u * 1.15
            p0, p1 = ray(ox, -0.96, ang, 2.5)
            is_head = (offset == 0)
            col = c.palette(0 if is_head else 1, v=1.0 if is_head else 0.55, white_boost=0.85 if is_head else 0.0)
            out.append(line(*_zig(idx, p0, p1), col, dwell=3 if is_head else 1))
            idx += 1
    return out


def sc_scissor_cross(c: Ctx):
    """Fast opposing dual-beam curtains that scissor violently across each other on every beat."""
    k = kick_env(c)
    n = c.count(3, 6, max(c.energy, 0.45))
    w = whip(c.bt, div=1.0, speed=10.0)
    beat_idx = int(c.bt)
    direction = 1.0 if beat_idx % 2 == 0 else -1.0
    frac = w - math.floor(w)
    swing = direction * (-0.35 + 0.70 * frac)

    out = []
    idx = 0
    # Curtain 1: fires from far-left floor aiming right
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = 0.42 + swing + u * (0.28 + 0.22 * k)
        p0, p1 = ray(-0.88, -0.96, ang, 2.6)
        out.append(line(*_zig(idx, p0, p1), c.palette(0, v=1.0, white_boost=0.5 * k if i == 0 else 0.0), dwell=1))
        idx += 1

    # Curtain 2: fires from mid-left ceiling aiming down-right (mirrored unit fires from mid-right ceiling)
    for i in range(max(2, n - 1)):
        u = (i / max(n - 2, 1)) - 0.5
        ang = math.pi - (0.38 - swing * 0.8 + u * 0.25)
        p0, p1 = ray(-0.35, 0.96, ang, 2.6)
        out.append(line(*_zig(idx, p0, p1), c.palette(1, v=0.95), dwell=1))
        idx += 1
    return out


def sc_liquid_sky(c: Ctx):
    """Razor-sharp aerial sheet that whips tilt angle on beats and slices vertically through haze."""
    k = kick_env(c)
    planes = c.count(2, 4, max(c.energy, 0.35))
    beat_idx = int(c.bt)
    w = whip(c.bt, div=0.5, speed=9.0)
    frac = w - math.floor(w)

    # Slanted sheet angle flips sharply every 2 beats (creates giant X-planes with mirrored projector)
    sgn = 1.0 if (beat_idx // 2) % 2 == 0 else -1.0
    tilt = sgn * (-0.35 + 0.82 * frac)
    y_base = 0.32 * math.sin(c.bt * TAU / 16)

    out = []
    idx = 0
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.14 + 0.22 * k)
        y0 = float(np.clip(y_base + gap - tilt, -0.94, 0.94))
        y1 = float(np.clip(y_base + gap + tilt, -0.94, 0.94))
        col = c.palette(p, v=0.9 + 0.1 * k, white_boost=0.55 * k if p == 0 else 0.0)

        # Every 2nd bar, chop the liquid sky into fast-moving dashed light pulses
        if (beat_idx // 4) % 2 == 1:
            for seg in dashed_beam((-0.96, y0), (0.96, y1), col, segments=c.count(3, 6), duty=0.55):
                out.append(seg)
        else:
            out.append(line(*_zig(idx, (-0.96, y0), (0.96, y1)), col, dwell=2))
            idx += 1
    return out


def sc_sky_sweep(c: Ctx):
    """Smooth, wide atmospheric breakdown sweeps: asymmetric floor-to-ceiling beam curtains."""
    v = 0.70 + 0.25 * math.sin(c.bt * TAU / 16)
    n = c.count(5, 9)
    # Slow sweeping fan from left-floor across the center (mirrored unit sweeps from right-floor)
    aim = 0.25 + 0.38 * math.sin(c.bt * TAU / 24)
    spread = 0.45 + 0.18 * math.sin(c.bt * TAU / 16)
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(-0.65, -0.96, ang, 2.5)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2, v=v), dwell=1))
    return out


def sc_vertical_pillars(c: Ctx):
    """High-impact vertical beam columns that whip horizontally across the stage on every beat."""
    k = kick_env(c)
    n = c.count(4, 7, max(c.energy, 0.45))
    w = whip(c.bt, div=1.0, speed=11.0)
    beat_idx = int(c.bt)
    frac = w - math.floor(w)

    # Whip beam cluster between left-edge and center-left on alternating beats
    x_start = -0.88 if beat_idx % 2 == 0 else -0.25
    x_end = -0.25 if beat_idx % 2 == 0 else -0.88
    x_center = x_start + (x_end - x_start) * frac
    width = 0.22 + 0.26 * k

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        x = float(np.clip(x_center + u * width, -0.96, 0.96))
        col = c.palette(i % 2, v=1.0, white_boost=0.75 * k if i == n // 2 else 0.0)
        out.append(line(*_zig(i, (x, -0.95), (x, 0.95)), col, dwell=2))
    return out


def sc_build_accelerate(c: Ctx):
    """Accelerating build-up whip: sweeps faster and faster, compressing into a tight laser blade at the drop."""
    b = c.f.build
    n = c.count(5, 11, max(b, 0.4))
    # Accelerates continuously via c.rt and narrows spread into a hyper-intense beam right before drop
    sweep = (0.55 * (1.0 - 0.65 * (b ** 2))) * math.sin(c.rt * TAU / 2.5)
    bias = 0.35 * (1.0 - 0.7 * b)
    spread = 0.52 * (1.0 - 0.78 * (b ** 2)) + 0.04
    ox = -0.60 * (1.0 - 0.65 * b)

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = bias + sweep + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.5)
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=0.85 * (b ** 2))
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


def sc_drop_strobe_whip(c: Ctx):
    """Hard drop cue: 8th-note gated beam whips alternating between tilted slashes, vertical walls, and fans."""
    if not gate(c, 2, 0.5):
        return []
    k = kick_env(c)
    beat_mod = int(c.bt) % 4
    w = whip(c.bt, div=2.0, speed=13.0)
    frac = w - math.floor(w)
    out = []

    if beat_mod == 0:
        # Rapidly whipping steep diagonal blade sheet
        tilt = -0.55 + 1.15 * frac
        for i, y in enumerate(np.linspace(-0.55, 0.55, c.count(3, 5))):
            out.append(line(*_zig(i, (-0.95, y - tilt), (0.95, y + tilt)), c.palette(i, white_boost=0.7 if i == 1 else 0.0), dwell=2))
    elif beat_mod == 1:
        # Tight whipping fan from left corner
        n = c.count(5, 8)
        aim = 0.15 + 0.65 * frac
        for i in range(n):
            u = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(-0.82, -0.96, aim + u * (0.35 + 0.25 * k), 2.5)
            out.append(line(*_zig(i, p0, p1), c.palette(i % 2, white_boost=0.6 * k), dwell=2))
    elif beat_mod == 2:
        # Whipping vertical beam wall
        x_base = -0.85 + 0.65 * frac
        for i, dx in enumerate(np.linspace(-0.22, 0.22, c.count(4, 6))):
            x = float(np.clip(x_base + dx, -0.96, 0.96))
            out.append(line(*_zig(i, (x, -0.95), (x, 0.95)), c.palette(i % 2, white_boost=0.65 if i == 0 else 0.0), dwell=2))
    else:
        # Overhead ceiling-to-floor scissor slash
        n = c.count(4, 7)
        aim = math.pi - (0.15 + 0.65 * frac)
        for i in range(n):
            u = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(-0.25, 0.96, aim + u * 0.42, 2.5)
            out.append(line(*_zig(i, p0, p1), c.palette(i % 2, white_boost=0.75 * k), dwell=2))
    return out


def sc_drop_machinegun(c: Ctx):
    """16th-note staccato beam stabs firing across alternating angles on drop energy."""
    if not gate(c, 4, 0.55):
        return []
    step = int(c.bt * 4)
    angles = [-0.25, 0.45, 0.05, 0.72, -0.10, 0.58, 0.22, 0.85]
    aim = angles[step % len(angles)]
    n = c.count(3, 5)
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(-0.60, -0.96, aim + u * 0.24, 2.5)
        out.append(line(*_zig(i, p0, p1), c.palette(step % 2, white_boost=0.8 if (step % 4 == 0) else 0.0), dwell=3))
    return out


SCENES = {
    # Pure modern beam scenes
    "whip_fan": sc_whip_fan,
    "saw_sweep": sc_saw_sweep,
    "beam_chase": sc_beam_chase,
    "scissor_cross": sc_scissor_cross,
    "liquid_sky": sc_liquid_sky,
    "sky_sweep": sc_sky_sweep,
    "vertical_pillars": sc_vertical_pillars,
    "build_accelerate": sc_build_accelerate,
    "drop_strobe_whip": sc_drop_strobe_whip,
    "drop_machinegun": sc_drop_machinegun,
    # Legacy aliases mapped strictly to modern beam sweeps (zero shapes)
    "tunnel": sc_saw_sweep,
    "tunnel_fast": sc_drop_strobe_whip,
    "beam_fan": sc_whip_fan,
    "starburst": sc_beam_chase,
    "starburst_fast": sc_drop_machinegun,
    "lissajous": sc_scissor_cross,
    "scope": sc_liquid_sky,
    "spiral": sc_saw_sweep,
    "spiral_slow": sc_sky_sweep,
    "grid": sc_vertical_pillars,
    "drop_blast": sc_drop_strobe_whip,
    "breath": sc_sky_sweep,
    "riser": sc_build_accelerate,
    "ring_pulse": sc_vertical_pillars,
    "sky_fan": sc_sky_sweep,
    "sky_cross": sc_scissor_cross,
    "scan_sweep": sc_liquid_sky,
    "build_fan": sc_build_accelerate,
    "build_scan": sc_build_accelerate,
    "strobe_poly": sc_drop_strobe_whip,
    "strobe_beams": sc_drop_strobe_whip,
    "strobe_burst": sc_drop_machinegun,
}

POOLS = {
    MAIN: ["whip_fan", "saw_sweep", "beam_chase", "scissor_cross", "liquid_sky", "vertical_pillars"],
    DROP: ["drop_strobe_whip", "drop_machinegun", "whip_fan", "scissor_cross", "vertical_pillars"],
    BREAKDOWN: ["liquid_sky", "sky_sweep", "scissor_cross"],
    BUILD: ["build_accelerate", "beam_chase", "liquid_sky"],
    SILENCE: [],
}
BARS_PER_SCENE = {MAIN: 2, DROP: 1, BREAKDOWN: 4, BUILD: 2, SILENCE: 99}


class FXEngine:
    """Picks beam cues by section, switches on bar lines, drives high-contrast festival palettes."""

    def __init__(self, seed=1, force_scene=None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene = None, None
        self.hue, self.bars, self._last_beat = 0.0, 0, 0
        self.rt, self._last_bt, self.strobe_enabled, self.detail = 0.0, 0.0, True, 0.5

    def _pick(self):
        pool = list(POOLS.get(self.section, []))
        if self.force:
            self.scene = self.force
        elif pool:
            choices = [s for s in pool if s != self.scene] or pool
            self.scene = choices[int(self.rng.integers(len(choices)))]
        else:
            self.scene = None

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
            self.rt += dbt * (1 + 5 * f.build if f.section == BUILD else 1)
        c = Ctx(f, bt, hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
        if not self.scene or f.section == SILENCE:
            return [], None
        return SCENES[self.scene](c), self.scene
