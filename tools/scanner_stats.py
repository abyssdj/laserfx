"""Per-scene scanner-load report: python tools/scanner_stats.py [points] [pps]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from laserfx.audio import Features
from laserfx.fx import SCENES, Ctx, POOLS
from laserfx.render import compose

points = int(sys.argv[1]) if len(sys.argv) > 1 else 900
pps = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
rng = np.random.default_rng(1)
rows = []
for name in sorted(SCENES):
    stat = []
    for bt in (3.1, 9.15, 13.4, 21.7):          # several moments in the beat cycle
        f = Features(bpm=128, locked=True, beat_phase=bt % 1, beat_pulse=np.exp(-3 * (bt % 1)), beat_count=int(bt),
                     level=0.9, kick=0.6, build=0.7,
                     wave=(np.sin(np.arange(1024) * 0.15) * 0.6 + rng.normal(0, 0.05, 1024)).astype(np.float32))
        c = Ctx(f, bt, 0.5, rng, rt=bt)
        paths = SCENES[name](c)
        fr = compose(paths, points, size=0.6)
        lit = fr[:, 2:5].max(1) > 0
        step = np.hypot(*np.diff(fr[:, :2], axis=0).T)
        # sharp corners among lit points: direction change > 60deg
        d = np.diff(fr[:, :2], axis=0); ang = np.arctan2(d[:, 1], d[:, 0]); da = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
        corners = int(((da > 1.05) & lit[1:-1] & (np.hypot(*d[1:].T) > 1e-4) & (np.hypot(*d[:-1].T) > 1e-4)).sum())
        stat.append((len(paths), len(fr), 100 * (1 - lit.mean()), step.max() * pps / 1000, corners))
    a = np.array(stat).mean(0)
    rows.append((name, *a))
print(f"{'scene':<15}{'paths':>6}{'pts':>6}{'blank%':>8}{'maxstep(kU/s)':>15}{'corners':>9}")
for r in rows:
    print(f"{r[0]:<15}{r[1]:>6.0f}{r[2]:>6.0f}{r[3]:>8.0f}{r[4]:>15.1f}{r[5]:>9.0f}")
print(f"\nmean paths {np.mean([r[1] for r in rows]):.1f}   mean blank% {np.mean([r[3] for r in rows]):.0f}   "
      f"mean maxstep {np.mean([r[4] for r in rows]):.1f}   (frame rate at {points} pts/{pps} pps = {pps/points:.0f} fps)")
