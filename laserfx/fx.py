"""Laser effects: scenes are pure functions of a Ctx (beat time, energy, section) -> list[Path].

Coordinates are in [-1, 1]; colours are RGB in 0..1.
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
    return np.array(colorsys.hsv_to_rgb(h % 1.0, min(max(s, 0), 1), min(max(v, 0), 1)))


def poly(cx, cy, r, sides, rot, rgb, dwell=2):
    a = rot + np.arange(sides + 1) * TAU / sides
    return Path(np.stack([cx + r * np.sin(a), cy + r * np.cos(a)], 1), rgb, dwell)


def circle(cx, cy, r, rgb, n=48, rot=0.0):
    return poly(cx, cy, r, n, rot, rgb, dwell=0)


def line(a, b, rgb):
    return Path(np.array([a, b], float), rgb, 1)


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


@dataclass
class Ctx:
    f: Features
    bt: float                  # continuous beat time (beats since start)
    hue: float
    rng: np.random.Generator

    @property
    def energy(self):
        return min(max(self.f.level, 0.0), 1.0)

    @property
    def pulse(self):
        return self.f.beat_pulse

    @property
    def kick(self):
        return self.f.kick

    def col(self, off=0.0, v=1.0, s=1.0):
        return hsv(self.hue + off, s, v)


def _gradient(c, n, span=0.4, v=1.0):
    return np.array([hsv(c.hue + span * i / max(n - 1, 1), 1, v) for i in range(n)])


# ------------------------------------------------------------------------ scenes
def sc_ring_pulse(c):
    n = 2 + int(c.energy * 3)
    return [circle(0, 0, min(0.2 + 0.2 * i + 0.15 * c.pulse, 0.95), c.col(i * 0.1)) for i in range(n)]


def sc_tunnel(c, speed=0.5):
    N, sides = 7, (4 if int(c.bt / 16) % 2 == 0 else 6)
    shift = (c.bt * speed) % 1
    out = []
    for k in range(N):
        u = (k + shift) / N
        r = min(0.04 + 0.9 * u ** 1.5, 0.92) * (1 + 0.06 * c.pulse)
        out.append(poly(0, 0, r, sides, c.bt * 0.12 + u * 0.8, c.col(u * 0.3, v=min(1, u * 2.5))))
    return out


def sc_beam_fan(c):
    n = 5 + 2 * int(c.energy * 4)
    spread = 0.55 + 0.35 * math.sin(c.bt * TAU / 8)
    sway = 0.35 * math.sin(c.bt * TAU / 16)
    out = []
    for i in range(n):
        a = (i / (n - 1) - 0.5) * 2 * spread + sway
        p0, p1 = ray(0, -0.95, a, 1.9)
        if i % 2:
            p0, p1 = p1, p0
        out.append(line(p0, p1, c.col(0.5 * (i % 2) + 0.05 * i)))
    return out


def sc_starburst(c, white=False, fast=False):
    n = 8 + int(c.energy * 16)
    rot = c.bt * (0.5 if fast else 0.15)
    r1, r2 = 0.12 + 0.1 * c.pulse, 0.55 + 0.4 * max(c.kick, c.pulse * 0.6)
    out = []
    for i in range(n):
        a = rot + i * TAU / n
        p0 = (r1 * math.sin(a), r1 * math.cos(a))
        p1 = (r2 * math.sin(a), r2 * math.cos(a))
        if i % 2:
            p0, p1 = p1, p0
        col = hsv(c.hue + 0.5 * (i % 2), 0.15 if white and i % 4 == 0 else 1.0)
        out.append(line(p0, p1, col))
    return out


def sc_lissajous(c):
    a, b = [(3, 2), (5, 4), (3, 4), (1, 2)][int(c.bt / 8) % 4]
    t = np.linspace(0, TAU, 300)
    A = 0.55 + 0.3 * c.energy
    ph = c.bt * TAU / 16
    pts = np.stack([A * np.sin(a * t + ph), A * np.sin(b * t)], 1)
    return [Path(pts, _gradient(c, len(t), 0.5))]


def sc_scope(c):
    w = c.f.wave
    n = 128
    w = w[: len(w) // n * n].reshape(n, -1).mean(1)
    y = np.clip(w / max(np.abs(w).max(), 0.05) * 0.5 * (0.4 + 0.6 * c.energy), -0.9, 0.9)
    x = np.linspace(-0.9, 0.9, n)
    g = _gradient(c, n, 0.4)
    return [Path(np.stack([x, y + 0.25], 1), g), Path(np.stack([x[::-1], -y[::-1] - 0.25], 1), g[::-1])]


def sc_spiral(c, slow=False):
    th = np.linspace(0, TAU * 3, 240)
    r = 0.05 + 0.85 * th / th[-1]
    rot = c.bt * (0.08 if slow else 0.25)
    pts = np.stack([r * np.sin(th + rot), r * np.cos(th + rot)], 1)
    return [Path(pts, _gradient(c, len(th), 0.4, 0.7 if slow else 1.0))]


def sc_grid(c):
    yh, N = 0.15, 6
    shift = (c.bt * 0.5) % 1
    out = []
    for k in range(N):
        d = (k + shift) / N
        y, w = yh - (yh + 0.9) * d ** 2, 0.2 + 0.8 * d
        a, b = (-w, y), (w, y)
        out.append(line(*((a, b) if k % 2 else (b, a)), c.col(0.0, v=min(1, d * 2 + 0.2))))
    for i, xb in enumerate(np.linspace(-1.3, 1.3, 7)):
        p0, p1 = (0, yh), (xb * 0.75, -0.9)
        out.append(line(*((p0, p1) if i % 2 else (p1, p0)), c.col(0.5, v=0.8)))
    return out


def sc_drop_blast(c):
    out = sc_starburst(c, white=True, fast=True)
    for j in range(3):
        u = (c.bt + j / 3) % 1
        out.append(circle(0, 0, min(0.05 + u * 1.0, 0.98), hsv(c.hue + j * 0.1, 0.3 * (1 - u), 1 - 0.6 * u)))
    return out


def sc_breath(c):
    r = 0.3 + 0.12 * math.sin(c.bt * TAU / 8)
    col = c.col(0, v=0.6)
    out = [poly(0, 0, r, 6, c.bt * 0.03, col), circle(0, 0, r * 0.5, c.col(0.08, v=0.5))]
    x = np.linspace(-0.9, 0.9, 80)
    for s in (1, -1):
        y = s * 0.45 + 0.18 * np.sin(2.5 * math.pi * x + s * c.bt * TAU / 8)
        out.append(Path(np.stack([x, y], 1), c.col(0.05 * s, v=0.5)))
    return out


def sc_riser(c):
    b = c.f.build
    P = 4.0 - 3.5 * b
    u = (c.bt / P) % 1
    shake = 0.02 * b
    col = hsv(0.62 - 0.62 * b, 1 - 0.5 * b, 0.6 + 0.4 * b)
    out = [circle(0, 0, max(1 - u, 0.03) * 0.9, col)]
    n = 4 + int(b * 12)
    rot = c.bt * 0.1 * (1 + 4 * b)
    for i in range(n):
        a = rot + i * TAU / n
        r1, r2 = 0.1, 0.35 + 0.5 * b
        j = c.rng.normal(0, shake, 2)
        out.append(line((r1 * math.sin(a), r1 * math.cos(a)), (r2 * math.sin(a) + j[0], r2 * math.cos(a) + j[1]), col))
    return out


SCENES = {
    "ring_pulse": sc_ring_pulse, "tunnel": sc_tunnel, "beam_fan": sc_beam_fan,
    "starburst": sc_starburst, "lissajous": sc_lissajous, "scope": sc_scope,
    "spiral": sc_spiral, "grid": sc_grid, "drop_blast": sc_drop_blast,
    "breath": sc_breath, "riser": sc_riser,
    "starburst_fast": lambda c: sc_starburst(c, white=True, fast=True),
    "tunnel_fast": lambda c: sc_tunnel(c, speed=1.0),
    "spiral_slow": lambda c: sc_spiral(c, slow=True),
}

POOLS = {
    MAIN: ["tunnel", "beam_fan", "starburst", "lissajous", "grid", "spiral", "scope", "ring_pulse"],
    DROP: ["drop_blast", "starburst_fast", "tunnel_fast"],
    BREAKDOWN: ["breath", "scope", "spiral_slow"],
    BUILD: ["riser"],
    SILENCE: [],
}
BARS_PER_SCENE = {MAIN: 2, DROP: 1, BREAKDOWN: 4, BUILD: 99, SILENCE: 99}


class FXEngine:
    """Picks scenes by section, switches on bar lines, drives palette from the music."""

    def __init__(self, seed=1, force_scene=None):
        self.rng = np.random.default_rng(seed)
        self.force = force_scene
        self.section, self.scene = None, None
        self.hue, self.bars, self._last_beat = 0.5, 0, 0

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
                self.hue += 0.06
                if self.bars % BARS_PER_SCENE[self.section] == 0:
                    self._pick()
            self._last_beat = f.beat_count

        hue = self.hue
        if f.section == DROP:
            hue = [0.0, 0.88, 0.55, 0.12][f.beat_count % 4]
        elif f.section == BREAKDOWN:
            hue = 0.62 + 0.04 * math.sin(f.time * 0.3)
        c = Ctx(f, f.beat_count + f.beat_phase, hue, self.rng)
        if not self.scene or f.section == SILENCE:
            return [], None
        return SCENES[self.scene](c), self.scene
