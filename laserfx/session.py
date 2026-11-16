"""Background engine used by the GUI: audio -> analysis -> FX -> galvo physics -> laser output."""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass

import numpy as np

from . import audio, output
from .fx import FXEngine, Path
from .render import SafetyLimiter, blank_frame, compose


@dataclass
class Settings:
    brightness: float = 0.5
    size: float = 0.6
    points: int = 260
    pps: int = 30000
    flip_x: bool = False
    flip_y: bool = False
    swap_xy: bool = False
    scene: str | None = None        # None = automatic
    section: str | None = None      # None = automatic
    blackout: bool = False
    strobe: bool = True
    detail: float = 0.5
    min_fps: int = 80
    galvo_lag: int = 2              # color-to-XY point shift (-12..+12 pts)
    corner_dwell: int = 3           # repeated points at sharp vertices & line ends (0..12 pts)
    blank_delay: int = 2            # blanked settle points before/after jumps (0..12 pts)
    corner_brake: float = 0.55      # symmetric deceleration/acceleration into vertices (0.0..1.0)


@dataclass
class Snapshot:
    feats: audio.Features
    scene: str | None
    frame: np.ndarray
    tripped: bool
    log: list


def prepare_paths_for_galvos(
    paths: list[Path],
    corner_dwell: int = 0,
    corner_brake: float = 0.0,
) -> list[Path]:
    """Applies physical mirror deceleration and vertex dwell directly on raw Path geometry
    BEFORE compose() resamples points—ensuring razor-sharp corners instead of rounded turns.
    """
    if not paths or (corner_dwell <= 0 and corner_brake <= 0.01):
        return paths

    out: list[Path] = []
    brake = min(max(float(corner_brake), 0.0), 1.0)

    for p in paths:
        pts = np.asarray(p.pts, dtype=float)
        n = len(pts)
        if n < 2:
            out.append(p)
            continue

        # Short aerial pips (< 0.02 length) already represent stationary points
        span = float(np.hypot(*(pts[-1] - pts[0]))) if n == 2 else 1.0
        if n == 2 and span < 0.02:
            out.append(Path(pts, p.rgb, max(p.dwell, 1)))
            continue

        # 1. Subdivide segments with S-curve velocity easing so galvos decelerate smoothly into vertices
        if brake > 0.05:
            eased_pts = []
            eased_rgb = []
            is_per_v = (np.asarray(p.rgb).ndim == 2)
            rgb_arr = np.asarray(p.rgb, dtype=float)
            steps = 5
            u_lin = np.linspace(0.0, 1.0, steps + 1)
            # Blend linear spacing with smoothstep (0-velocity at endpoints)
            u_smooth = u_lin * u_lin * (3.0 - 2.0 * u_lin)
            u = (1.0 - brake) * u_lin + brake * u_smooth

            for seg in range(n - 1):
                p0, p1 = pts[seg], pts[seg + 1]
                sl = slice(0, steps) if seg < n - 2 else slice(0, steps + 1)
                uu = u[sl, None]
                eased_pts.append(p0 * (1.0 - uu) + p1 * uu)
                if is_per_v:
                    c0, c1 = rgb_arr[seg], rgb_arr[seg + 1]
                    eased_rgb.append(c0 * (1.0 - uu) + c1 * uu)
            new_pts = np.vstack(eased_pts)
            new_rgb = np.vstack(eased_rgb) if is_per_v else p.rgb
        else:
            new_pts = pts.copy()
            new_rgb = np.asarray(p.rgb, dtype=float).copy()

        # 2. Insert explicit vertex dwell at start, end, and any corner > ~35 degrees
        if corner_dwell > 0:
            v = np.diff(new_pts, axis=0)
            seg_len = np.hypot(v[:, 0], v[:, 1])
            valid = seg_len > 1e-5
            u_vec = np.zeros_like(v)
            u_vec[valid] = v[valid] / seg_len[valid, None]
            turn_cos = np.sum(u_vec[:-1] * u_vec[1:], axis=1) if len(u_vec) > 1 else np.array([])

            is_per_v = (new_rgb.ndim == 2)
            dw_pts = []
            dw_rgb = []
            for idx in range(len(new_pts)):
                is_end = (idx == 0) or (idx == len(new_pts) - 1)
                is_corner = (0 < idx < len(new_pts) - 1) and (turn_cos[idx - 1] < 0.80)
                rep = (1 + corner_dwell) if (is_end or is_corner) else 1
                dw_pts.append(np.repeat(new_pts[idx : idx + 1], rep, axis=0))
                if is_per_v:
                    dw_rgb.append(np.repeat(new_rgb[idx : idx + 1], rep, axis=0))
            new_pts = np.vstack(dw_pts)
            if is_per_v:
                new_rgb = np.vstack(dw_rgb)

        out.append(Path(new_pts, new_rgb, dwell=0))

    return out


def apply_galvo_timing(
    frame: np.ndarray,
    galvo_lag: int = 0,
    blank_delay: int = 0,
) -> np.ndarray:
    """Applies blanking settle hold and diode-vs-galvo color shift on an (N, 5) [x, y, r, g, b] frame."""
    if len(frame) < 3 or (galvo_lag == 0 and blank_delay <= 0):
        return frame

    pts = frame.copy()

    # 1. Blanking delay: blank the first `blank_delay` lit points after every dark jump
    # and hold the final vertex dark for `blank_delay` points before jumping away
    if blank_delay > 0:
        lit = pts[:, 2:5].max(axis=1) > 0.01
        d = int(blank_delay)
        for i in range(1, len(pts)):
            if lit[i] and not lit[i - 1]:
                end = min(len(pts), i + d)
                pts[i:end, 2:5] = 0.0

    # 2. Color-to-XY galvo lag shift: shifts RGB modulation relative to XY scanner waveform
    if galvo_lag != 0 and len(pts) > abs(galvo_lag):
        lag = int(galvo_lag)
        shifted = np.roll(pts[:, 2:5], lag, axis=0)
        if lag > 0:
            shifted[:lag] = 0.0
        else:
            shifted[lag:] = 0.0
        pts[:, 2:5] = shifted

    return pts


class Session:
    def __init__(self, settings: Settings | None = None, sr=44100):
        self.settings = settings or Settings()
        self.sr = sr
        self.snapshot: Snapshot | None = None
        self.error: str | None = None
        self.running = False
        self.finished = False
        self._stop = threading.Event()
        self._thread = None

    def start(self, source, kind="preview"):
        s = self.settings
        if kind == "helios":
            out = output.HeliosUSB(0, s.pps, s.flip_x, s.flip_y, s.swap_xy)
            out.open()
        else:
            out = output.NullOutput(s.pps)
            out.open()
        try:
            source.start()
        except Exception:
            out.close()
            raise
        self.error, self.finished, self.running = None, False, True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(source, out), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self, src, out):
        an = audio.Analyzer(self.sr)
        eng = FXEngine()
        safety = SafetyLimiter()
        feats = an.features()
        last_force = object()
        cached_log: list = []
        last_log_len = -1
        try:
            while not self._stop.is_set() and not getattr(src, "done", False):
                s = self.settings
                chunk = src.read()
                if len(chunk):
                    feats = an.process(chunk)
                if s.section:
                    feats = dataclasses.replace(feats, section=s.section)
                if s.scene != last_force:
                    eng.force, last_force = s.scene, s.scene
                    eng.section = feats.section
                    eng._pick()
                eng.strobe_enabled, eng.detail = s.strobe, s.detail
                budget = max(90, min(int(s.points), s.pps // max(s.min_fps, 1)))
                paths, scene = eng.update(feats)
                safety.brightness = s.brightness
                if s.blackout:
                    frame = blank_frame(budget)
                else:
                    shaped_paths = prepare_paths_for_galvos(
                        paths,
                        corner_dwell=int(s.corner_dwell),
                        corner_brake=float(s.corner_brake),
                    )
                    raw_frame = compose(shaped_paths, budget, size=s.size)
                    timed_frame = apply_galvo_timing(
                        raw_frame,
                        galvo_lag=int(s.galvo_lag),
                        blank_delay=int(s.blank_delay),
                    )
                    frame = safety.apply(timed_frame)
                out.pps = s.pps
                if hasattr(out, "flip"):
                    out.flip = dict(flip_x=s.flip_x, flip_y=s.flip_y, swap_xy=s.swap_xy)

                if len(an.log) != last_log_len:
                    cached_log = list(an.log)
                    last_log_len = len(an.log)
                self.snapshot = Snapshot(feats, scene, frame, safety.tripped, cached_log)

                out.send(frame)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            for c in (src, out):
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            self.running, self.finished = False, True