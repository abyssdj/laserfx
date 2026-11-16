"""Laser effects: audio-triggered ballistic beam sweeps, transient bursts, and twin-projector chaos.

Coordinates are in [-1, 1]; colours are RGB in 0..1.
Designed for a single output feeding twin projectors where Projector 2 is X-mirrored (x -> -x):
- Off-axis origins and asymmetric tilts ensure Unit A and X-mirrored Unit B slice across each other
  in mid-air rather than overlapping onto identical beam paths.
- Stateful transient detectors extract kick edges, snare hits, and hi-hat energy directly from
  `Features` to trigger one-shot sweeps, polarity flips, and beam fractures.
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


def dashed_beam(a, b, rgb, segments=4, duty=0.55, phase=0.0):
    """Chops a beam into travelling high-contrast aerial packets."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = []
    for i in range(segments):
        t0 = ((i + phase) % segments) / segments
        t1 = min(t0 + duty / segments, 1.0)
        if t1 <= t0 + 0.02:
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
    # Audio-derived event states (populated by FXEngine)
    kick_age: float = 1.0      # beats elapsed since last detected kick transient (0.0 = just fired!)
    kick_count: int = 0        # total kick transients fired (drives deterministic/chaotic state flips)
    snare_age: float = 1.0     # beats elapsed since last mid/high transient spike
    snare_count: int = 0
    hi_energy: float = 0.0     # 0..1 instantaneous high-frequency energy (hats, cymbals, synth fizz)
    sweep_from: float = -0.3   # ballistic whip start angle (rad)
    sweep_to: float = 0.6      # ballistic whip target angle (rad)

    @property
    def energy(self):
        return min(max(self.f.level, 0.0), 1.0)

    @property
    def pulse(self):
        return self.f.beat_pulse

    @property
    def kick(self):
        return self.f.kick

    def whip_progress(self, speed: float = 11.0) -> float:
        """0.0 -> 1.0 ballistic curve driven by actual time since the last kick transient."""
        return 1.0 - math.exp(-speed * max(self.kick_age, 0.0))

    def snare_punch(self, decay: float = 8.0) -> float:
        """1.0 -> 0.0 sharp burst triggered by snare/clap transients."""
        return math.exp(-decay * max(self.snare_age, 0.0))

    def count(self, lo, hi, x=1.0):
        return int(round(lo + (hi - lo) * min(max(self.detail * x, 0.0), 1.0)))

    def palette(self, idx: int = 0, v: float = 1.0, white_boost: float = 0.0):
        p_idx = int(abs(self.hue * 10)) % len(PALETTES)
        h_pri, h_acc, s_acc = PALETTES[p_idx]
        if idx % 2 == 0:
            return hsv(h_pri, max(1.0 - white_boost, 0.0), v)
        return hsv(h_acc, max(s_acc - white_boost, 0.0), v)


# ------------------------------------------------------------------------ AUDIO-DRIVEN CHAOS SCENES

def sc_ballistic_whip(c: Ctx):
    """Every kick fires a real one-shot ballistic whip across the room; hi-hats fracture the beams."""
    prog = c.whip_progress(speed=10.5)
    punch = 1.0 - prog
    snare = c.snare_punch(9.0)

    n = c.count(5, 9, max(c.energy, 0.45))
    aim = c.sweep_from + (c.sweep_to - c.sweep_from) * prog
    # Fan explodes wide right as the kick hits, then tightens into a razor blade
    spread = 0.12 + 0.52 * punch + 0.25 * snare
    # Flip origin between left-floor and left-ceiling on snares!
    oy = 0.96 if (c.snare_count % 2 == 1 and snare > 0.25) else -0.96
    ox = -0.62 if oy < 0 else -0.42
    base_dir = math.pi - aim if oy > 0 else aim

    out = []
    idx = 0
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = base_dir + u * 2.0 * spread
        p0, p1 = ray(ox, oy, ang, 2.5)
        is_edge = (i == 0 or i == n - 1)
        col = c.palette(1 if is_edge else (i % 2), v=1.0, white_boost=0.85 * punch if is_edge else 0.35 * snare)

        # High-frequency energy (hats/cymbals) shatters outer beams into flying light packets
        if c.hi_energy > 0.45 and (i % 2 == 1):
            for seg in dashed_beam(p0, p1, col, segments=4, duty=0.5, phase=c.bt * 6.0):
                out.append(seg)
                idx += 1
        else:
            out.append(line(*_zig(idx, p0, p1), col, dwell=2))
            idx += 1
    return out


def sc_chaos_storm(c: Ctx):
    """Full-tilt drop chaos: kick-triggered angle slams, snare ceiling-drops, and hi-hat strobe bursts."""
    if not gate(c, 4, 0.65):
        return []
    prog = c.whip_progress(speed=13.0)
    punch = 1.0 - prog
    snare = c.snare_punch(10.0)

    # Choose chaotic mode dynamically from kick_count & snare hits
    mode = (c.kick_count + c.snare_count) % 3
    out = []
    idx = 0

    if mode == 0:
        # Twin intersecting floor + ceiling whips firing simultaneously
        n = c.count(4, 7)
        aim1 = c.sweep_from + (c.sweep_to - c.sweep_from) * prog
        aim2 = math.pi - (0.75 - 0.95 * prog)
        for i in range(n):
            u = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(-0.75, -0.96, aim1 + u * (0.35 + 0.3 * punch), 2.5)
            out.append(line(*_zig(idx, p0, p1), c.palette(0, white_boost=0.8 * punch), dwell=2))
            idx += 1
        for i in range(max(3, n - 1)):
            u = (i / max(n - 2, 1)) - 0.5
            p0, p1 = ray(-0.28, 0.96, aim2 + u * (0.30 + 0.25 * snare), 2.5)
            out.append(line(*_zig(idx, p0, p1), c.palette(1, white_boost=0.8 * snare), dwell=2))
            idx += 1

    elif mode == 1:
        # Violent horizontal + diagonal blade slashes driven by kick & snare positions
        planes = c.count(3, 6)
        tilt = (0.75 * punch - 0.35) * (1.0 if c.kick_count % 2 == 0 else -1.0)
        y_sweep = -0.65 + 1.30 * prog
        for p in range(planes):
            off = (p - (planes - 1) / 2.0) * (0.16 + 0.18 * snare)
            y0 = float(np.clip(y_sweep + off - tilt, -0.95, 0.95))
            y1 = float(np.clip(y_sweep + off + tilt, -0.95, 0.95))
            col = c.palette(p % 2, white_boost=0.9 if (p == 0 and punch > 0.4) else 0.0)
            if c.hi_energy > 0.4:
                for seg in dashed_beam((-0.96, y0), (0.96, y1), col, segments=5, duty=0.55, phase=c.bt * 8.0):
                    out.append(seg)
            else:
                out.append(line(*_zig(idx, (-0.96, y0), (0.96, y1)), col, dwell=2))
                idx += 1

    else:
        # Rapid-fire 16th-note staccato angle bursts + vertical laser wall on snares
        step = int(c.bt * 4)
        angles = [-0.32, 0.68, 0.05, 0.85, -0.18, 0.52, 0.25, 0.92]
        aim = angles[(step + c.kick_count) % len(angles)]
        n = c.count(4, 7)
        for i in range(n):
            u = (i / max(n - 1, 1)) - 0.5
            p0, p1 = ray(-0.60, -0.96, aim + u * (0.22 + 0.38 * punch), 2.5)
            out.append(line(*_zig(idx, p0, p1), c.palette(step % 2, white_boost=0.85 if punch > 0.5 else 0.0), dwell=2))
            idx += 1
        if snare > 0.2:
            x_wall = -0.85 + 0.65 * (1.0 - snare)
            out.append(line((x_wall, -0.95), (x_wall, 0.95), hsv(0.0, 0.0, 1.0), dwell=3))
    return out


def sc_hat_shatter(c: Ctx):
    """High-frequency reactive beam array: solid beams explode into travelling light bullets on hats/synths."""
    prog = c.whip_progress(speed=9.0)
    punch = 1.0 - prog
    n = c.count(5, 9, max(c.energy, 0.4))
    aim = c.sweep_from + (c.sweep_to - c.sweep_from) * prog
    spread = 0.32 + 0.35 * c.hi_energy
    segs = max(2, int(2 + 5 * c.hi_energy))

    out = []
    idx = 0
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(-0.58, -0.96, ang, 2.5)
        col = c.palette(i % 2, v=1.0, white_boost=0.7 * c.hi_energy if i % 2 == 0 else 0.0)
        if c.hi_energy > 0.25:
            for seg in dashed_beam(p0, p1, col, segments=segs, duty=0.48, phase=c.bt * 7.0 + i * 0.3):
                out.append(seg)
                idx += 1
        else:
            out.append(line(*_zig(idx, p0, p1), col, dwell=2))
            idx += 1
    return out


def sc_snare_scissors(c: Ctx):
    """Kick sweeps floor curtain; Snare slams an opposing ceiling curtain violently downward."""
    k_prog = c.whip_progress(speed=11.0)
    s_prog = 1.0 - c.snare_punch(9.0)
    n = c.count(4, 7, max(c.energy, 0.45))

    out = []
    idx = 0
    # Floor curtain driven by kick transient
    aim_floor = c.sweep_from + (c.sweep_to - c.sweep_from) * k_prog
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        p0, p1 = ray(-0.82, -0.96, aim_floor + u * 0.36, 2.6)
        out.append(line(*_zig(idx, p0, p1), c.palette(0, white_boost=0.6 * (1.0 - k_prog)), dwell=1))
        idx += 1

    # Ceiling curtain driven independently by snare transient
    aim_ceil = math.pi - (-0.25 + 1.05 * s_prog)
    for i in range(max(3, n - 1)):
        u = (i / max(n - 2, 1)) - 0.5
        p0, p1 = ray(-0.22, 0.96, aim_ceil + u * 0.32, 2.6)
        out.append(line(*_zig(idx, p0, p1), c.palette(1, white_boost=0.85 * (1.0 - s_prog)), dwell=1))
        idx += 1
    return out


def sc_vertical_slam(c: Ctx):
    """Vertical laser columns that slam horizontally across the room on every kick & snare hit."""
    prog = c.whip_progress(speed=12.0)
    punch = 1.0 - prog
    snare = c.snare_punch(8.0)
    n = c.count(4, 8, max(c.energy, 0.45))

    # Map sweep_from/to into X coordinates [-0.90, -0.12] (mirrored unit covers [+0.12, +0.90])
    x0 = -0.50 + 0.38 * math.sin(c.sweep_from * 2.0)
    x1 = -0.50 + 0.38 * math.sin(c.sweep_to * 2.0)
    x_center = x0 + (x1 - x0) * prog
    width = 0.14 + 0.34 * punch + 0.22 * snare

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        x = float(np.clip(x_center + u * width, -0.96, 0.96))
        col = c.palette(i % 2, v=1.0, white_boost=0.8 * punch if i == n // 2 else 0.0)
        out.append(line(*_zig(i, (x, -0.95), (x, 0.95)), col, dwell=2))
    return out


def sc_liquid_sky(c: Ctx):
    """Dynamic aerial sheet: tilts sharply on kicks, ripples with audio energy, and fractures on hi-hats."""
    prog = c.whip_progress(speed=9.5)
    punch = 1.0 - prog
    snare = c.snare_punch(8.0)
    planes = c.count(2, 4, max(c.energy, 0.35))

    sgn = 1.0 if c.kick_count % 2 == 0 else -1.0
    tilt = sgn * (-0.35 + 0.85 * prog)
    y_base = 0.35 * math.sin(c.bt * TAU / 12) + 0.22 * snare * (-sgn)

    out = []
    idx = 0
    for p in range(planes):
        gap = (p - (planes - 1) / 2.0) * (0.12 + 0.26 * punch)
        y0 = float(np.clip(y_base + gap - tilt, -0.94, 0.94))
        y1 = float(np.clip(y_base + gap + tilt, -0.94, 0.94))
        col = c.palette(p, v=1.0, white_boost=0.7 * punch if p == 0 else 0.0)
        if c.hi_energy > 0.38:
            for seg in dashed_beam((-0.96, y0), (0.96, y1), col, segments=c.count(3, 6), duty=0.55, phase=c.bt * 5.0):
                out.append(seg)
        else:
            out.append(line(*_zig(idx, (-0.96, y0), (0.96, y1)), col, dwell=2))
            idx += 1
    return out


def sc_sky_sweep(c: Ctx):
    """Atmospheric breakdown canopy that still reacts dynamically to subtle percussion & synth swells."""
    v = 0.65 + 0.35 * c.energy
    n = c.count(5, 9)
    aim = 0.25 + 0.38 * math.sin(c.bt * TAU / 20) + 0.15 * c.snare_punch(5.0)
    spread = 0.42 + 0.24 * c.energy
    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = aim + u * 2.0 * spread
        p0, p1 = ray(-0.65, -0.96, ang, 2.5)
        col = c.palette(i % 2, v=v, white_boost=0.45 * c.hi_energy if i == n // 2 else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=1))
    return out


def sc_build_accelerate(c: Ctx):
    """Accelerating build-up whip: sweeps faster with build tension and flickers wildly on snare rolls."""
    b = c.f.build
    snare = c.snare_punch(12.0)
    n = c.count(5, 11, max(b, 0.4))
    sweep = (0.58 * (1.0 - 0.65 * (b ** 2))) * math.sin(c.rt * TAU / 2.2)
    bias = 0.35 * (1.0 - 0.7 * b)
    spread = 0.52 * (1.0 - 0.78 * (b ** 2)) + 0.04 + 0.18 * snare
    ox = -0.60 * (1.0 - 0.65 * b)

    out = []
    for i in range(n):
        u = (i / max(n - 1, 1)) - 0.5
        ang = bias + sweep + u * 2.0 * spread
        p0, p1 = ray(ox, -0.96, ang, 2.5)
        col = c.palette(i % 2, v=0.75 + 0.25 * b, white_boost=min(1.0, 0.85 * (b ** 2) + 0.5 * snare))
        out.append(line(*_zig(i, p0, p1), col, dwell=2))
    return out


SCENES = {
    # Audio-triggered modern chaos & beam scenes
    "ballistic_whip": sc_ballistic_whip,
    "chaos_storm": sc_chaos_storm,
    "hat_shatter": sc_hat_shatter,
    "snare_scissors": sc_snare_scissors,
    "vertical_slam": sc_vertical_slam,
    "liquid_sky": sc_liquid_sky,
    "sky_sweep": sc_sky_sweep,
    "build_accelerate": sc_build_accelerate,
    # Aliases so CLI / saved sessions remain compatible
    "whip_fan": sc_ballistic_whip,
    "saw_sweep": sc_ballistic_whip,
    "beam_chase": sc_hat_shatter,
    "scissor_cross": sc_snare_scissors,
    "vertical_pillars": sc_vertical_slam,
    "drop_strobe_whip": sc_chaos_storm,
    "drop_machinegun": sc_chaos_storm,
    "tunnel": sc_ballistic_whip,
    "tunnel_fast": sc_chaos_storm,
    "beam_fan": sc_ballistic_whip,
    "starburst": sc_hat_shatter,
    "starburst_fast": sc_chaos_storm,
    "lissajous": sc_snare_scissors,
    "scope": sc_liquid_sky,
    "spiral": sc_ballistic_whip,
    "spiral_slow": sc_sky_sweep,
    "grid": sc_vertical_slam,
    "drop_blast": sc_chaos_storm,
    "breath": sc_sky_sweep,
    "riser": sc_build_accelerate,
    "ring_pulse": sc_vertical_slam,
    "sky_fan": sc_sky_sweep,
    "sky_cross": sc_snare_scissors,
    "scan_sweep": sc_liquid_sky,
    "build_fan": sc_build_accelerate,
    "build_scan": sc_build_accelerate,
    "strobe_poly": sc_chaos_storm,
    "strobe_beams": sc_chaos_storm,
    "strobe_burst": sc_chaos_storm,
}

POOLS = {
    MAIN: ["ballistic_whip", "snare_scissors", "hat_shatter", "vertical_slam", "liquid_sky"],
    DROP: ["chaos_storm", "ballistic_whip", "snare_scissors", "hat_shatter", "vertical_slam"],
    BREAKDOWN: ["liquid_sky", "sky_sweep", "hat_shatter"],
    BUILD: ["build_accelerate", "hat_shatter", "liquid_sky"],
    SILENCE: [],
}
BARS_PER_SCENE = {MAIN: 2, DROP: 1, BREAKDOWN: 4, BUILD: 2, SILENCE: 99}


class FXEngine:
    """Tracks real-time audio transients (kicks, snares, hi-hats) to fire ballistic beam sweeps & chaos."""

    def __init__(self, seed=1, force_scene=None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene = None, None
        self.hue, self.bars, self._last_beat = 0.0, 0, 0
        self.rt, self._last_bt, self.strobe_enabled, self.detail = 0.0, 0.0, True, 0.5

        # Stateful transient & ballistic sweep trackers
        self._prev_kick = 0.0
        self._prev_hi = 0.0
        self._hi_avg = 0.05
        self.kick_age = 1.0
        self.kick_count = 0
        self.snare_age = 1.0
        self.snare_count = 0
        self.hi_energy = 0.0
        self.sweep_from = -0.35
        self.sweep_to = 0.65

    def _pick(self):
        pool = list(POOLS.get(self.section, []))
        if self.force:
            self.scene = self.force
        elif pool:
            choices = [s for s in pool if s != self.scene] or pool
            self.scene = choices[int(self.rng.integers(len(choices)))]
        else:
            self.scene = None

    def _trigger_kick_sweep(self):
        self.kick_age = 0.0
        self.kick_count += 1
        # Launch a new ballistic whip from the current target to a contrasting wild angle
        self.sweep_from = self.sweep_to
        if self.kick_count % 2 == 0:
            self.sweep_to = float(self.rng.uniform(0.35, 0.92))
        else:
            self.sweep_to = float(self.rng.uniform(-0.42, 0.15))

    def update(self, f: Features):
        if f.section != self.section:
            self.section, self.bars = f.section, 0
            self._pick()

        bt = f.beat_count + f.beat_phase
        dbt = bt - self._last_bt
        self._last_bt = bt
        if 0 < dbt < 1:
            self.rt += dbt * (1 + 5 * f.build if f.section == BUILD else 1)
            self.kick_age += dbt
            self.snare_age += dbt

        # 1. Detect rising-edge kick transients (or hard beat downbeats) to trigger ballistic sweeps
        kick_edge = (f.kick > 0.45 and self._prev_kick <= 0.45) or (f.beat_count != self._last_beat and f.kick > 0.3)
        if kick_edge and self.kick_age > 0.18:
            self._trigger_kick_sweep()
        self._prev_kick = f.kick

        # 2. Extract high-frequency transient energy directly from `f.wave` for snares & hi-hats
        w = f.wave
        if len(w) > 8:
            # First-order difference isolates high frequencies (hats, claps, snare snap)
            hf = float(np.sqrt(np.mean(np.diff(w) ** 2)))
            self._hi_avg = 0.92 * self._hi_avg + 0.08 * hf
            rel_hi = hf / max(self._hi_avg, 1e-4)
            self.hi_energy = float(np.clip((rel_hi - 0.85) * 0.9, 0.0, 1.0))

            # Sharp spike in HF energy triggers a snare/clap ceiling slam or polarity inversion
            if rel_hi > 1.55 and self._prev_hi <= 1.55 and self.snare_age > 0.22:
                self.snare_age = 0.0
                self.snare_count += 1
            self._prev_hi = rel_hi

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
            # Shift festival palette aggressively every 8 beats on drops
            hue = [0.0, 0.1, 0.2, 0.4][(f.beat_count // 8) % 4]
        elif f.section == BREAKDOWN:
            hue = 0.0

        c = Ctx(
            f, bt, hue, self.rng,
            rt=self.rt,
            strobe=self.strobe_enabled,
            detail=self.detail,
            kick_age=self.kick_age,
            kick_count=self.kick_count,
            snare_age=self.snare_age,
            snare_count=self.snare_count,
            hi_energy=self.hi_energy,
            sweep_from=self.sweep_from,
            sweep_to=self.sweep_to,
        )
        if not self.scene or f.section == SILENCE:
            return [], None
        return SCENES[self.scene](c), self.scene
