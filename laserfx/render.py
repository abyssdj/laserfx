"""Turn Paths into a laser point frame (N x 5 float: x, y, r, g, b) with blanking + safety."""
from __future__ import annotations

import numpy as np

from .fx import Path


def blank_frame(n=64):
    return np.zeros((n, 5), np.float32)


def _blank(a, b, n, ease=True):
    t = np.linspace(0, 1, n)
    if ease:                                   # smoothstep: gentle accel/decel for the galvos
        t = t * t * (3 - 2 * t)
    a, b = np.asarray(a), np.asarray(b)
    xy = a + (b - a) * t[:, None]
    return np.hstack([xy, np.zeros((n, 3))])


def _order(items):
    """Greedy nearest-neighbour ordering (paths may be reversed) to minimise blanked travel.
    The first path (the scene's most important one) stays first."""
    rest, out = list(items[1:]), [items[0]]
    while rest:
        end = out[-1][0][-1, :2]
        best, flip, bd = 0, False, 1e9
        for i, (V, _, _) in enumerate(rest):
            d0, d1 = np.hypot(*(V[0, :2] - end)), np.hypot(*(V[-1, :2] - end))
            if d0 < bd:
                best, flip, bd = i, False, d0
            if d1 < bd:
                best, flip, bd = i, True, d1
        V, L, d = rest.pop(best)
        out.append((V[::-1], L[::-1], d) if flip else (V, L, d))
    return out


def _corner_dwell(xy, dwell):
    """Points to hold at vertices: holds `dwell` on sharp turns AND gives 2-point rays bright aerial endpoints."""
    n = len(xy)
    d = np.zeros(n, int)
    if dwell <= 0 or n < 2:
        return d
    if n == 2:
        # Hold laser briefly at both ends of straight beams for crisp modern 'hot-beam' aerial dots
        end_hold = min(dwell, 3)
        d[0] = end_hold
        d[-1] = end_hold
        return d
    v = np.diff(xy, axis=0)
    ok = np.hypot(v[:, 0], v[:, 1]) > 1e-9
    a = np.arctan2(v[:, 1], v[:, 0])
    turn = np.abs((np.diff(a) + np.pi) % (2 * np.pi) - np.pi)
    turn = np.where(ok[:-1] & ok[1:], turn, 0.0)
    d[1:-1] = np.round(dwell * turn / (np.pi / 2)).astype(int)
    d[0] = 1
    d[-1] = 1
    return d


def _jump_points(dist, jump_scale=18, max_step=0.045):
    """Blanked points for a jump: fast enough for crisp dashed beams without overshooting galvos."""
    return max(3 + int(dist * jump_scale), int(np.ceil(dist * 1.5 / max_step)))


def compose(paths, budget=800, max_points=4000, size=0.8, settle=2, jump_scale=18, min_spacing=0.015,
            max_blank_frac=0.58):
    items = []
    for p in paths:
        if len(p.pts) < 2:
            continue
        xy = np.clip(p.pts * size, -1, 1)
        V = np.hstack([xy, np.broadcast_to(p.rgb, (len(xy), 3))])
        L = np.hypot(*np.diff(xy, axis=0).T)
        if L.sum() < 1e-4:
            continue
        items.append((V, L, p.dwell))
    if not items:
        return blank_frame(budget)
    items = _order(items)

    def overhead_of(its):
        o = 0
        for i, (V, _, _) in enumerate(its):
            b = its[(i + 1) % len(its)][0][0, :2]
            o += 2 * settle + _jump_points(np.hypot(*(b - V[-1, :2])), jump_scale)
        return o

    # Allow higher blanking fraction so dashed 'bullet' beams and multi-ray fans stay intact
    while len(items) > 4 and overhead_of(items) > max_blank_frac * budget:
        items.pop()

    dw = [_corner_dwell(V[:, :2], d) for V, _, d in items]
    overhead = overhead_of(items) + sum(int(x.sum()) for x in dw)
    total_len = sum(L.sum() for _, L, _ in items)
    spacing = max(total_len / max(budget - overhead, 40), min_spacing)

    out = []
    for i, (V, L, _) in enumerate(items):
        counts = np.maximum(1, np.round(L / spacing)).astype(int)
        seg = np.repeat(np.arange(len(L)), counts)
        starts = np.cumsum(counts) - counts
        t = (np.arange(len(seg)) - np.repeat(starts, counts)) / np.repeat(counts, counts)
        P = V[seg] + (V[seg + 1] - V[seg]) * t[:, None]
        P = np.vstack([P, V[-1]])
        if dw[i].any():
            at = np.repeat(np.append(starts, len(P) - 1), dw[i])
            P = np.insert(P, at, P[at], axis=0)
        out.append(P)
        a = V[-1, :2]
        b = items[(i + 1) % len(items)][0][0, :2]
        n = _jump_points(np.hypot(*(b - a)), jump_scale)
        out += [_blank(a, a, settle, False), _blank(a, b, n), _blank(b, b, settle, False)]

    f = np.vstack(out).astype(np.float32)
    if len(f) > max_points:
        f = f[np.linspace(0, len(f) - 1, max_points).astype(int)]
    if len(f) < 64:
        f = np.vstack([f, np.repeat(f[-1:], 64 - len(f), 0) * [1, 1, 0, 0, 0]])
    return f


class SafetyLimiter:
    """Global brightness plus a static-beam guard (a lit, near-stationary point is blanked)."""

    def __init__(self, brightness=0.5, min_extent=0.03):
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
