"""Background engine used by the GUI: audio -> analysis -> FX -> galvo compensation -> laser output."""
from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import audio, output
from .fx import FXEngine
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
    galvo_lag: int = 1              # color-to-XY point shift (-10..+10 pts)
    corner_dwell: int = 1           # extra dwell points at sharp corners (0..6 pts)
    blank_delay: int = 1            # extra blanked settling points on jumps (0..6 pts)


@dataclass
class Snapshot:
    feats: audio.Features
    scene: str | None
    frame: np.ndarray
    tripped: bool
    log: list


def apply_galvo_compensation(
    frame: np.ndarray,
    galvo_lag: int = 0,
    corner_dwell: int = 0,
    blank_delay: int = 0,
    max_points: int | None = None,
) -> np.ndarray:
    """Compensates physical scanner inertia and diode-vs-galvo latency on an (N, 5) [x, y, r, g, b] frame."""
    if len(frame) < 3 or (galvo_lag == 0 and corner_dwell <= 0 and blank_delay <= 0):
        return frame

    pts = frame

    # 1. Corner dwell & blanking settle insertion (respecting max_points budget)
    if corner_dwell > 0 or blank_delay > 0:
        xy = pts[:, :2]
        rgb = pts[:, 2:5]
        lit = rgb.max(axis=1) > 0.01

        v = np.diff(xy, axis=0)
        seg_len = np.hypot(v[:, 0], v[:, 1])
        valid = seg_len > 1e-4
        u = np.zeros_like(v)
        u[valid] = v[valid] / seg_len[valid, None]
        turn_cos = np.sum(u[:-1] * u[1:], axis=1)

        cap = max_points if max_points is not None else (len(pts) * 2)
        extra_budget = max(0, cap - len(pts))

        out_list = [pts[0:1]]
        for i in range(1, len(pts) - 1):
            row = pts[i : i + 1]
            out_list.append(row)
            if extra_budget <= 0:
                continue
            prev_lit = bool(lit[i - 1])
            curr_lit = bool(lit[i])
            next_lit = bool(lit[i + 1])

            if blank_delay > 0 and (not curr_lit) and (prev_lit or next_lit):
                rep = min(blank_delay, extra_budget)
                blank_row = row.copy()
                blank_row[:, 2:5] = 0.0
                out_list.append(np.repeat(blank_row, rep, axis=0))
                extra_budget -= rep
            elif corner_dwell > 0 and curr_lit and (not prev_lit or not next_lit or turn_cos[i - 1] < 0.65):
                rep = min(corner_dwell, extra_budget)
                out_list.append(np.repeat(row, rep, axis=0))
                extra_budget -= rep

        out_list.append(pts[-1:])
        pts = np.concatenate(out_list, axis=0)

    # 2. Color-to-XY galvo lag shift (rolls RGB channels relative to XY coordinates)
    if galvo_lag != 0 and len(pts) > abs(galvo_lag):
        comp = pts.copy()
        comp[:, 2:5] = np.roll(pts[:, 2:5], int(galvo_lag), axis=0)
        if galvo_lag > 0:
            comp[:galvo_lag, 2:5] = 0.0
        else:
            comp[galvo_lag:, 2:5] = 0.0
        pts = comp

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
                    dwell_reserve = min(budget // 4, (int(s.corner_dwell) + int(s.blank_delay)) * 10)
                    compose_budget = max(60, budget - dwell_reserve)
                    raw_frame = compose(paths, compose_budget, size=s.size)
                    comp_frame = apply_galvo_compensation(
                        raw_frame,
                        galvo_lag=int(s.galvo_lag),
                        corner_dwell=int(s.corner_dwell),
                        blank_delay=int(s.blank_delay),
                        max_points=budget,
                    )
                    frame = safety.apply(comp_frame)
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