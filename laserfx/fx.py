"""Laser effects: scenes are pure functions of a Ctx (beat time, energy, section) -> list[Path].

Coordinates are in [-1, 1]; colours are RGB in 0..1.
Designed for a single output feeding twin projectors where Projector 2 is X-mirrored (x -> -x):
- Scenes use deliberate X-asymmetry, off-axis origins, and slanted planes so the two units
  interlock, criss-cross, and fill wide stereo space without overlapping onto the same beams.
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
    dwell: int = 0             # extra points held at each vertex (sharp corners)


def hsv(h, s=1.0, v=1.0):
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0.0), 1.0), min(max(v, 0.0), 1.0)))


def poly(cx, cy, r, sides, rot, rgb, dwell=2):
    a = rot + np.arange(sides + 1) * TAU / sides
    return Path(np.stack([cx + r * np.sin(a), cy + r * np.cos(a)], 1), rgb, dwell)


def circle(cx, cy, r, rgb, n=48, rot=0.0):
    return poly(cx, cy, r, n, rot, rgb, dwell=0)


def line(a, b, rgb, dwell=1):
    return Path(np.array([a, b], float), rgb, dwell)


def dashed_line(a, b, rgb, segments=4, duty=0.6):
    """Splits a beam into crisp aerial segments (modern multi-beam / grating look)."""
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


def ray(ox, oy, ang, length):
    """Ray from (ox, oy) at `ang` (0 = up), clipped to the unit box."""
    dx, dy, t = math.sin(ang), math.cos(ang), length
    if dx > 1e-9:
        t = min(t, (1 - ox) / dx)
    elif dx < -1e-9:
        t = min(t, (-1 - ox) / dx)
    if dy > 1e-9:
        t = min(t, (1 - oy) / dy)
    elif dy < -1e-9:
        t = min(t, (-1 - oy) / dy)
    return (ox, oy), (ox + dx * t, oy + dy * t)


# ---------------------------------------------------------------- motion & palette helpers
def snap_ease(bt: float, div: float = 1.0, sharpness: float = 5.0) -> float:
    """Beat-quantized step + exponential whip: snaps quickly on each beat, then glides."""
    x = bt * div
    idx = math.floor(x)
    frac = x - idx
    eased = 1.0 - math.exp(-sharpness * frac)
    return idx + eased


def kick_env(c: Ctx) -> float:
    """Snappy transient response combining kick detector and beat pulse."""
    return min(max(max(c.kick, c.pulse * 0.75) ** 1.35, 0.0), 1.0)


def gate(c: Ctx, div: float = 2.0, duty: float = 0.5) -> bool:
    """Strobe gate: lit for `duty` of each 1/div beat. Always lit when strobing is disabled."""
    return (not c.strobe) or ((c.bt * div) % 1.0) < duty


def _zig(i: int, a, b):
    return (a, b) if i % 2 == 0 else (b, a)


# Curated high-contrast festival palettes: (primary_hue, accent_hue, accent_sat)
PALETTES = [
    (0.54, 0.83, 0.85),   # Icy Cyan + Electric Magenta
    (0.00, 0.56, 0.15),   # Pure Crimson + Crisp Ice-White
    (0.76, 0.11, 0.90),   # Deep UV Violet + Amber Gold
    (0.36, 0.55, 0.20),   # Acid Emerald + Cool White
    (0.62, 0.04, 0.95),   # Cobalt Blue + Laser Red/Orange
]


@dataclass
class Ctx:
    f: Features
    bt: float                  # continuous beat time (beats since start)
    hue: float
    rng: np.random.Generator
    rt: float = 0.0            # "rise time": beats integrated with a speed-up that grows during BUILD
    strobe: bool = True        # False -> strobing scenes stay lit (photosensitivity-safe)
    detail: float = 0.5        # 0.2 sparse .. 1.0 busy: scales how many beams/rings a scene draws

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
        """Element count between lo and hi, scaled by the Detail setting and a 0..1 musical drive `x`."""
        return int(round(lo + (hi - lo) * min(max(self.detail * x, 0.0), 1.0)))

    def palette(self, idx: int = 0, v: float = 1.0, white_boost: float = 0.0):
        """Returns high-contrast two-tone festival colors instead of rainbow gradients."""
        p_idx = int(abs(self.hue * 10)) % len(PALETTES)
        h_pri, h_acc, s_acc = PALETTES[p_idx]
        if idx % 2 == 0:
            return hsv(h_pri, max(1.0 - white_boost, 0.0), v)
        return hsv(h_acc, max(s_acc - white_boost, 0.0), v)

    def col(self, off=0.0, v=1.0, s=1.0):
        return hsv(self.hue + off, s, v)


# ------------------------------------------------------------------------ MODERN SCENES
# All scenes are crafted for single-output + X-mirrored second projector:
# - Off-axis origins (ox != 0) produce huge crossing V-lattices in the room
# - Tilted planes (y_left != y_right) produce giant aerial X-scissors
# - Off-center tunnels/chevrons form twin side-by-side 3D corridors instead of overlapping

def sc_dual_sweep(c: Ctx):
    """Off-axis sweeping fan: Unit A fires from left-bottom, mirrored Unit B fires from right-bottom."""
    k = kick_env(c)
    n = c.count(5, 9, max(c.energy, 0.4))
    step = snap_ease(c.bt, div=0.5, sharpness=6.0)
    # Origin offset ensures the two units never overlap onto identical beam paths
    ox = -0.62 + 0.14 * math.sin(c.bt * TAU / 16)
    oy = -0.96
    # Sweep inward across the centerline so both fans intersect mid-air
    aim_angle = 0.36 + 0.34 * math.sin(step * math.pi * 0.5)
    spread = 0.34 + 0.28 * k

    out = []
    for i in range(n):
        frac = (i / max(n - 1, 1)) - 0.5
        ang = aim_angle + frac * 2.0 * spread
        p0, p1 = ray(ox, oy, ang, 2.4)
        is_edge = (i == 0 or i == n - 1)
        col = c.palette(1 if is_edge else (i % 2), v=1.0, white_boost=0.65 * k if is_edge else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=1))
    return out


def sc_horizon_slice(c: Ctx):
    """Slanted 'Liquid Sky' planes: because y_left != y_right, Unit B's X-flip creates a giant X-plane."""
    k = kick_env(c)
    planes = c.count(2, 4, max(c.energy, 0.35))
    beat_idx = int(c.bt)
    snap = snap_ease(c.bt, div=1.0, sharpness=7.0)

    # Deliberate non-zero tilt so mirrored projector creates intersecting diagonal sheets
    tilt_dir = 1.0 if (beat_idx // 2) % 2 == 0 else -1.0
    tilt = tilt_dir * (0.34 + 0.18 * math.sin(snap * math.pi * 0.5))
    y_center = 0.25 * math.cos(snap * math.pi * 0.25)

    out = []
    idx = 0
    for p in range(planes):
        offset = (p - (planes - 1) / 2.0) * (0.20 + 0.16 * k)
        y0 = float(np.clip(y_center + offset - tilt, -0.92, 0.92))
        y1 = float(np.clip(y_center + offset + tilt, -0.92, 0.92))
        col = c.palette(p, v=0.85 + 0.15 * k, white_boost=0.45 if p == 0 else 0.0)

        if (beat_idx // 4) % 2 == 1:
            segs = c.count(3, 6)
            for seg in dashed_line((-0.95, y0), (0.95, y1), col, segments=segs, duty=0.62):
                out.append(seg)
        else:
            out.append(line(*_zig(idx, (-0.95, y0), (0.95, y1)), col, dwell=2))
            idx += 1
    return out


def sc_chevron_matrix(c: Ctx):
    """Asymmetric aerial wedges shifted off-center so twin units form interlocking W/M architecture."""
    k = kick_env(c)
    layers = c.count(3, 6, max(c.energy, 0.4))
    half_beat = int(c.bt * 2)
    flip = 1.0 if (half_beat % 4) < 2 else -1.0
    # Shift X center off zero so Unit A sits left-center and mirrored Unit B sits right-center
    cx = -0.34 if (int(c.bt / 4) % 2 == 0) else 0.34
    wing_spread = 0.52 + 0.18 * k
    y_shift = 0.16 * math.sin(snap_ease(c.bt, 1.0, 6.0) * math.pi)

    out = []
    for i in range(layers):
        u = i / max(layers - 1, 1)
        apex_y = flip * (-0.52 + 0.90 * u) + y_shift * 0.3
        wing_y = float(np.clip(apex_y + flip * (0.36 + 0.22 * k), -0.95, 0.95))
        apex_y = float(np.clip(apex_y, -0.95, 0.95))
        w = wing_spread * (0.45 + 0.55 * u)

        x_left = max(cx - w, -0.96)
        x_right = min(cx + w, 0.96)
        pts = np.array([[x_left, wing_y], [cx, apex_y], [x_right, wing_y]], dtype=float)
        if i % 2 == 1:
            pts = pts[::-1]
        col = c.palette(i, v=0.55 + 0.45 * (1.0 - u), white_boost=0.5 * k if i == 0 else 0.0)
        out.append(Path(pts, col, dwell=3))
    return out


def sc_apex_tunnel(c: Ctx, speed: float = 0.75):
    """Off-axis perspective corridor: creates twin side-by-side 3D tunnels in the room."""
    k = kick_env(c)
    n = c.count(4, 7)
    bar_idx = int(c.bt / 4)
    sides = 4
    base_rot = (math.pi / 4) if (bar_idx % 2 == 0) else 0.0
    twist = 0.14 * math.sin(snap_ease(c.bt, 1.0, 5.0) * math.pi * 0.5)
    shift = (c.bt * speed + 0.25 * k) % 1.0

    # Vanishing point sweeps smoothly off-center so mirrored unit opens a second parallel corridor
    vp_x = -0.38 * math.cos(c.bt * TAU / 16)
    vp_y = 0.15 * math.sin(c.bt * TAU / 16)

    out = []
    for idx in range(n):
        u = (idx + shift) / n
        r = min(0.05 + 0.58 * (u ** 1.6), 0.65) * (1.0 + 0.08 * k)
        cx = vp_x * (1.0 - u * 0.45)
        cy = vp_y * (1.0 - u * 0.45)
        v = min(1.0, u * 2.2)
        col = c.palette(idx, v=v, white_boost=0.45 * k if idx == n - 1 else 0.0)
        out.append(poly(cx, cy, r, sides, base_rot + twist * (1.0 - u), col, dwell=3))
    return out


def sc_prism_burst(c: Ctx, white: bool = False, fast: bool = False):
    """Off-axis beat-snapped star cluster: projects twin counter-rotating aerial hubs."""
    k = kick_env(c)
    arms = 2 * c.count(3, 6, max(c.energy, 0.4))
    div = 2.0 if fast else 1.0
    snap_angle = int(c.bt * div) * (TAU / arms / 2.0)
    r_inner = 0.08 + 0.08 * k
    r_outer = 0.46 + 0.22 * k

    # Shift hub off-center so Unit A and Unit B form twin wide-stereo burst hubs
    cx = -0.42 if (int(c.bt / 2) % 2 == 0) else 0.36
    cy = 0.12 * math.sin(c.bt * TAU / 8)

    out = []
    for i in range(arms):
        a = snap_angle + i * TAU / arms
        p0 = (cx + r_inner * math.sin(a), cy + r_inner * math.cos(a))
        p1 = (np.clip(cx + r_outer * math.sin(a), -0.98, 0.98), np.clip(cy + r_outer * math.cos(a), -0.98, 0.98))
        is_accent = (i % 2 == 1)
        col = c.palette(
            1 if is_accent else 0,
            v=1.0,
            white_boost=0.75 if (white and i % 4 == 0) else 0.0,
        )
        out.append(line(*_zig(i, p0, p1), col, dwell=1))
    return out


def sc_beam_fan(c: Ctx):
    """Sweeping diagonal curtain: sweeps from left-to-right so mirrored unit sweeps right-to-left."""
    k = kick_env(c)
    n = 2 * c.count(3, 6, max(c.energy, 0.4)) + 1
    step = snap_ease(c.bt, div=1.0, sharpness=6.5)
    spread = 0.36 + 0.34 * k
    # Biased sweep angle avoids overlapping the mirrored unit on the dead-center vertical
    sway = -0.25 + 0.55 * math.sin(step * math.pi * 0.5)
    ox = -0.35

    out = []
    for i in range(n):
        a = (i / max(n - 1, 1) - 0.5) * 2.0 * spread + sway
        p0, p1 = ray(ox, -0.96, a, 2.2)
        col = c.palette(i, v=1.0, white_boost=0.5 * k if i == n // 2 else 0.0)
        out.append(line(*_zig(i, p0, p1), col, dwell=1))
    return out


def sc_grid(c: Ctx):
    """Tilted cyber horizon + off-axis ray fan: forms an intersecting double-perspective floor."""
    k = kick_env(c)
    yh = 0.12 + 0.06 * math.sin(snap_ease(c.bt, 0.5, 5.0) * math.pi)
    slant = 0.14  # Asymmetric horizon tilt creates a crisp X-horizon with mirrored unit
    n_horiz = c.count(3, 6)
    shift = (c.bt * 0.8 + 0.2 * k) % 1.0
    out = []
    for idx in range(n_horiz):
        d = (idx + shift) / n_horiz
        y = yh - (yh + 0.92) * (d ** 2)
        w = 0.25 + 0.72 * d
        col = c.palette(0, v=min(1.0, d * 2.0 + 0.25))
        out.append(line(*_zig(idx, (-w, y - slant * d), (w, y + slant * d)), col, dwell=1))

    n_rays = 2 * c.count(2, 4) + 1
    vp_x = -0.32
    for i, xb in enumerate(np.linspace(-1.1, 1.3, n_rays)):
        p0, p1 = (vp_x, yh), (np.clip(xb * 0.75, -0.96, 0.96), -0.92)
        col = c.palette(1, v=0.85 + 0.15 * k)
        out.append(line(*_zig(i, p0, p1), col, dwell=1))
    return out


def sc_scope(c: Ctx):
    """Audio waveform ribbon: mirrored unit automatically produces an inverted X-axis stereo twin."""
    w = c.f.wave
    n = 48
    w = w[: len(w) // n * n].reshape(n, -1).mean(1)
    w = np.convolve(np.pad(w, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    y = np.clip(w / max(np.abs(w).max(), 0.05) * 0.36 * (0.45 + 0.55 * c.energy), -0.55, 0.55)
    # Shift slightly upward so inverted X-mirror doesn't cancel out symmetrical waveforms
    y_shifted = y + 0.18
    x = np.linspace(-0.92, 0.92, n)
    return [Path(np.stack([x, y_shifted], 1), c.palette(0, v=1.0))]


def sc_ring_pulse(c: Ctx):
    """Twin off-axis expanding octagons (Unit A on left, mirrored Unit B on right)."""
    k = kick_env(c)
    n = c.count(2, 4, max(c.energy, 0.4))
    rot = (int(c.bt) % 2) * (math.pi / 8)
    cx = -0.38
    out = []
    for i in range(n):
        r = min(0.14 + 0.15 * i + 0.12 * k, 0.58)
        out.append(poly(cx, 0.0, r, 8, rot, c.palette(i, v=1.0, white_boost=0.45 * k if i == 0 else 0.0), dwell=2))
    return out


# ------------------------------------------------- BREAKDOWN: wide atmospheric skies + slow sweeps
def sc_sky_fan(c: Ctx):
    """Asymmetric aerial canopy: Unit A sweeps from left-floor, mirrored Unit B sweeps from right-floor."""
    v = 0.65 + 0.2 * math.sin(c.bt * TAU / 16)
    spread = 0.55 + 0.18 * math.sin(c.bt * TAU / 32)
    sway = 0.20 * math.sin(c.bt * TAU / 48)
    out, idx = [], 0
    # Lower canopy biased left (-0.48), upper sparse canopy biased right (+0.48)
    configs = (
        (-0.48, -0.97, c.count(5, 8), 1, v, 0.25),
        (0.48, 0.97, c.count(3, 5), -1, v * 0.6, -0.25),
    )
    for ox, oy, n, sgn, vv, bias in configs:
        for i in range(n):
            a = (i / max(n - 1, 1) - 0.5) * 2.0 * spread * (1.0 if sgn > 0 else 0.85) + bias + sway * sgn
            p0, p1 = ray(ox, oy, a if sgn > 0 else math.pi - a, 2.3)
            out.append(line(*_zig(idx, p0, p1), c.palette(i, v=vv)))
            idx += 1
    return out


def sc_sky_cross(c: Ctx):
    """Asymmetric corner fan: Unit A fires diagonally up-right, Unit B mirrors up-left for a cathedral arch."""
    sw = 0.24 * math.sin(c.bt * TAU / 32)
    v = 0.65 + 0.2 * math.sin(c.bt * TAU / 16 + 1)
    n = c.count(5, 8)
    out = []
    for i in range(n):
        a = 0.22 + 0.11 * i + sw
        p0, p1 = ray(-0.92, -0.97, a, 2.6)
        out.append(line(*_zig(i, p0, p1), c.palette(i % 2, v=v)))
    return out


def sc_scan_sweep(c: Ctx):
    """Crisp tilted bars + vertical curtains that criss-cross cleanly across both projectors."""
    out = []
    echoes = c.count(2, 4)
    for k, v in enumerate((1.0, 0.5, 0.25, 0.12)[:echoes]):
        lag = k * 0.16
        y = 0.72 * math.sin(c.bt * TAU / 16 - lag)
        # Keep x-curtain off exact 0.0 so twin beams form dual parallel vertical sheets
        x = -0.45 + 0.42 * math.sin(c.bt * TAU / 24 + 1.0 - lag)
        out.append(line(*_zig(k, (-0.92, y - 0.15), (0.92, y + 0.15)), c.palette(0, v=0.8 * v)))
        out.append(line(*_zig(k + 1, (x, -0.92), (x, 0.92)), c.palette(1, v=0.8 * v)))
    return out


# ---------------------------------------------------- BUILD: accelerating sweeps & compression
def sc_riser(c: Ctx):
    """Twin converging diamonds that pull inward toward the center as the drop approaches."""
    b = c.f.build
    u = (c.rt / 3.5) % 1.0
    shake = 0.022 * (b ** 1.5)
    col_main = c.palette(0, v=0.65 + 0.35 * b, white_boost=0.5 * (b ** 2))
    col_acc = c.palette(1, v=0.75 + 0.25 * b, white_boost=0.7 * (b ** 2))

    # Hubs slide from (-0.45) toward center (0.0) right as build hits 1.0
    cx = -0.45 * (1.0 - 0.85 * b)
    r_poly = max(1.0 - u, 0.04) * 0.62
    out = [poly(cx, 0.0, r_poly, 4, math.pi / 4 + c.rt * 0.15, col_acc, dwell=3)]
    n = c.count(4, 10, max(b, 0.3))
    rot = c.rt * 0.25
    for i in range(n):
        a = rot + i * TAU / n
        r1, r2 = 0.08, 0.28 + 0.38 * b
        j = c.rng.normal(0, shake, 2)
        p0 = (cx + r1 * math.sin(a), r1 * math.cos(a))
        p1 = (np.clip(cx + r2 * math.sin(a) + j[0], -0.98, 0.98), np.clip(r2 * math.cos(a) + j[1], -0.98, 0.98))
        out.append(line(*_zig(i, p0, p1), col_main if i % 2 == 0 else col_acc))
    return out


def sc_build_fan(c: Ctx):
    """Accelerating off-axis fan that sweeps inward and locks into a tight V-beam at the drop."""
    b = c.f.build
    n = c.count(6, 12, max(b, 0.35))
    ox = -0.58 * (1.0 - 0.5 * b)
    spread = (0.65 * (1.0 - 0.68 * (b ** 2))) + 0.10
    bias = 0.32 * (1.0 - 0.6 * b)
    sweep = (0.38 * (1.0 - 0.5 * b)) * math.sin(c.rt * TAU / 3.0)
    out = []
    for i in range(n):
        a = (i / max(n - 1, 1) - 0.5) * 2.0 * spread + bias + sweep
        p0, p1 = ray(ox, -0.97, a, 2.3)
        col = c.palette(i % 2, v=0.7 + 0.3 * b, white_boost=0.65 * (b ** 2))
        out.append(line(*_zig(i, p0, p1), col))
    return out


def sc_build_scan(c: Ctx):
    """Accelerating criss-crossing diagonal laser planes."""
    b = c.f.build
    k = c.count(1, 3, max(b, 0.3))
    out = []
    for j in range(k):
        ph = j * TAU / k
        y = 0.78 * math.sin(c.rt * TAU / 3.5 + ph)
        x = -0.45 + 0.42 * math.sin(c.rt * TAU / 5.0 + ph + 1.0)
        col = c.palette(j, v=0.95, white_boost=0.6 * (b ** 2))
        out.append(line(*_zig(j, (-0.92, y - 0.18), (0.92, y + 0.18)), col))
        out.append(line(*_zig(j + 1, (x, -0.92), (x, 0.92)), col))
    return out


# --------------------------------------------- DROP: beat-snapped stereo & criss-cross strobes
def sc_strobe_poly(c: Ctx):
    """Ping-ponging stereo polygons: snaps between left-center and right-center on every half-beat."""
    if not gate(c, 2, 0.5):
        return []
    sides = [3, 4, 6, 4][int(c.bt) % 4]
    step = int(c.bt * 2)
    k = kick_env(c)
    n = c.count(2, 3, max(c.energy, 0.5))
    # Off-axis X center ensures twin units sit side-by-side in stereo rather than stacking
    cx = -0.40 if (step % 2 == 0) else 0.40
    out = []
    for idx in range(n):
        r = (0.24 + 0.32 * idx / max(n - 1, 1)) * (0.92 + 0.08 * k)
        rot = step * (TAU / sides) / 2.0 * (1 if idx % 2 == 0 else -1)
        is_downbeat = (step % 4 == 0)
        col = c.palette(idx, v=1.0, white_boost=0.85 if is_downbeat else 0.0)
        out.append(poly(cx, 0.0, r, sides, rot, col, dwell=3))
    return out


def sc_strobe_beams(c: Ctx):
    """4-beat stereo strobe sequence: Slanted X-Planes -> Stereo Verticals -> Diagonal Slash -> Twin Diamonds."""
    if not gate(c, 2, 0.5):
        return []
    beat_mod = int(c.bt) % 4
    L = 0.92
    out = []
    if beat_mod == 0:
        # Slanted horizontal bars -> forms giant intersecting X-planes across both projectors
        for i, y in enumerate(np.linspace(-0.72, 0.72, c.count(3, 5))):
            out.append(line(*_zig(i, (-L, y - 0.22), (L, y + 0.22)), c.palette(i, white_boost=0.5 if i == 2 else 0.0)))
    elif beat_mod == 1:
        # Biased vertical columns -> forms 6–10 distinct vertical beams across the full room width
        for i, x in enumerate(np.linspace(-0.88, -0.15, c.count(3, 5))):
            out.append(line(*_zig(i, (x, -L), (x, L)), c.palette(i, white_boost=0.5 if i == 0 else 0.0)))
    elif beat_mod == 2:
        # Asymmetric diagonal slash -> mirrored projector forms the opposing slash
        for i, o in enumerate((-0.32, 0.0, 0.32)):
            out.append(line(*_zig(i, (-0.82, -0.68 + o), (0.82, 0.68 + o)), c.palette(i)))
    else:
        # Twin side-by-side diamonds with crosshairs
        cx = -0.42
        out.append(poly(cx, 0.0, 0.48, 4, math.pi / 4, c.palette(0, white_boost=0.7), dwell=3))
        out.append(line((cx - 0.48, 0.0), (cx + 0.48, 0.0), c.palette(1)))
        out.append(line((cx, 0.48), (cx, -0.48), c.palette(1)))
    return out


def sc_strobe_burst(c: Ctx):
    """Twin off-axis beam hubs + expanding stereo diamond shockwaves."""
    if not gate(c, 2, 0.5):
        return []
    n = 2 * c.count(3, 5, max(c.energy, 0.5))
    snap = int(c.bt * 2) * (TAU / n / 2.0)
    cx = -0.40
    out = []
    for i in range(n):
        a = snap + i * TAU / n
        p0 = (cx + 0.08 * math.sin(a), 0.08 * math.cos(a))
        p1 = (np.clip(cx + 0.54 * math.sin(a), -0.96, 0.96), 0.72 * math.cos(a))
        col = c.palette(i % 2, white_boost=0.75 if i % 4 == 0 else 0.0)
        out.append(line(*_zig(i, p0, p1), col))
    u = (c.bt * 2) % 1.0
    out.append(poly(cx, 0.0, 0.08 + 0.46 * u, 4, math.pi / 4 + snap, c.palette(1, white_boost=0.5), dwell=3))
    return out


def sc_drop_blast(c: Ctx):
    """Full-arena stereo drop climax: twin prism hubs + dual expanding diamond shockwaves."""
    out = sc_prism_burst(c, white=True, fast=True)
    nr = c.count(1, 3)
    cx = -0.42 if (int(c.bt / 2) % 2 == 0) else 0.36
    for j in range(nr):
        u = (c.bt + j / nr) % 1.0
        r = min(0.06 + u * 0.52, 0.60)
        col = c.palette(j, v=1.0 - 0.5 * u, white_boost=0.8 * (1.0 - u))
        out.append(poly(cx, 0.0, r, 4, math.pi / 4, col, dwell=3))
    return out


SCENES = {
    # Modern core scenes (single-output + X-mirrored twin projector optimized)
    "dual_sweep": sc_dual_sweep,
    "horizon_slice": sc_horizon_slice,
    "chevron_matrix": sc_chevron_matrix,
    "apex_tunnel": sc_apex_tunnel,
    "prism_burst": sc_prism_burst,
    "beam_fan": sc_beam_fan,
    "grid": sc_grid,
    "scope": sc_scope,
    "ring_pulse": sc_ring_pulse,
    "drop_blast": sc_drop_blast,
    "riser": sc_riser,
    "sky_fan": sc_sky_fan,
    "sky_cross": sc_sky_cross,
    "scan_sweep": sc_scan_sweep,
    "build_fan": sc_build_fan,
    "build_scan": sc_build_scan,
    "strobe_poly": sc_strobe_poly,
    "strobe_beams": sc_strobe_beams,
    "strobe_burst": sc_strobe_burst,
    "prism_fast": lambda c: sc_prism_burst(c, white=True, fast=True),
    "tunnel_fast": lambda c: sc_apex_tunnel(c, speed=1.25),
    # Legacy aliases so CLI / saved sessions never break
    "tunnel": sc_apex_tunnel,
    "starburst": sc_prism_burst,
    "starburst_fast": lambda c: sc_prism_burst(c, white=True, fast=True),
    "spiral": sc_chevron_matrix,
    "spiral_slow": sc_horizon_slice,
    "lissajous": sc_dual_sweep,
    "breath": sc_sky_fan,
}

POOLS = {
    MAIN: ["dual_sweep", "chevron_matrix", "apex_tunnel", "beam_fan", "prism_burst", "grid", "scope", "ring_pulse"],
    DROP: ["strobe_beams", "strobe_burst", "strobe_poly", "drop_blast", "prism_fast", "chevron_matrix"],
    BREAKDOWN: ["horizon_slice", "sky_fan", "sky_cross", "scan_sweep", "dual_sweep"],
    BUILD: ["build_fan", "riser", "build_scan", "horizon_slice"],
    SILENCE: [],
}
BARS_PER_SCENE = {MAIN: 2, DROP: 1, BREAKDOWN: 4, BUILD: 2, SILENCE: 99}


class FXEngine:
    """Picks scenes by section, switches on bar lines, drives palette from the music."""

    def __init__(self, seed=1, force_scene=None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene = None, None
        self.hue, self.bars, self._last_beat = 0.5, 0, 0
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
        if 0 < dbt < 1:   # speed-up only while building; integrated so acceleration is smooth
            self.rt += dbt * (1 + 5 * f.build if f.section == BUILD else 1)
        c = Ctx(f, bt, hue, self.rng, rt=self.rt, strobe=self.strobe_enabled, detail=self.detail)
        if not self.scene or f.section == SILENCE:
            return [], None
        return SCENES[self.scene](c), self.scene
