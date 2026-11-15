"""Turn Paths into a laser point frame (N x 5 float: x, y, r, g, b) with blanking + safety."""
from __future__ import annotations

import numpy as np

from .fx import Path


def blank_frame(n=64):
    return np.zeros((n, 5), np.float32)


def _blank(a, b, n):
    t = np.linspace(0, 1, n)[:, None]
    xy = np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t
    return np.hstack([xy, np.zeros((n, 3))])


def compose(paths, budget=900, max_points=4000, size=0.8, settle=3, jump_scale=20):
    items = []
    for p in paths:
        if len(p.pts) < 2:
            continue
        xy = np.clip(p.pts * size, -1, 1)
        V = np.hstack([xy, np.broadcast_to(p.rgb, (len(xy), 3))])
        L = np.hypot(*np.diff(xy, axis=0).T)
        items.append((V, L, p.dwell))
    if not items:
        return blank_frame()
    total_len = sum(L.sum() for _, L, _ in items)
    if total_len < 1e-6:
        return blank_frame()

    ends = [(V[-1, :2], items[(i + 1) % len(items)][0][0, :2]) for i, (V, _, _) in enumerate(items)]
    overhead = sum(2 * settle + 4 + int(np.hypot(*(b - a)) * jump_scale) for a, b in ends)
    overhead += sum(d * len(V) for V, _, d in items)
    spacing = total_len / max(budget - overhead, 40)

    out = []
    for i, (V, L, dwell) in enumerate(items):
        counts = np.maximum(1, np.round(L / spacing)).astype(int)
        seg = np.repeat(np.arange(len(L)), counts)
        starts = np.cumsum(counts) - counts
        t = (np.arange(len(seg)) - np.repeat(starts, counts)) / np.repeat(counts, counts)
        P = V[seg] + (V[seg + 1] - V[seg]) * t[:, None]
        P = np.vstack([P, V[-1]])
        if dwell:
            at = np.repeat(np.append(starts, len(P) - 1), dwell)
            P = np.insert(P, at, P[at], axis=0)
        out.append(P)
        a, b = ends[i]
        n = 4 + int(np.hypot(*(b - a)) * jump_scale)
        out += [_blank(a, a, settle), _blank(a, b, n), _blank(b, b, settle)]

    f = np.vstack(out).astype(np.float32)
    if len(f) > max_points:
        f = f[np.linspace(0, len(f) - 1, max_points).astype(int)]
    if len(f) < 64:
        f = np.vstack([f, np.repeat(f[-1:], 64 - len(f), 0) * [1, 1, 0, 0, 0]])
    return f


class SafetyLimiter:
    """Global brightness plus a static-beam guard (a lit, near-stationary point is blanked)."""

    def __init__(self, brightness=0.5, min_extent=0.04):
        self.brightness, self.min_extent = brightness, min_extent
        self.tripped = False

    def apply(self, f):
        f = f.copy()
        lit = f[:, 2:5].max(1) > 0.02
        self.tripped = False
        if lit.any():
            xy = f[lit, :2]
            if np.ptp(xy, axis=0).max() < self.min_extent:
                f[:, 2:5] = 0
                self.tripped = True
        f[:, 2:5] = np.clip(f[:, 2:5] * self.brightness, 0, 1)
        return f
