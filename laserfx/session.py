"""Background engine used by the GUI: audio -> analysis -> FX -> laser output, in one worker thread."""
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
    points: int = 250             # 250 pts @ 30k pps = 120 FPS (was 700 pts = 42 FPS)
    pps: int = 30000
    flip_x: bool = False
    flip_y: bool = False
    swap_xy: bool = False
    scene: str | None = None        # None = automatic
    section: str | None = None      # None = automatic
    blackout: bool = False
    strobe: bool = True
    detail: float = 0.5
    min_fps: int = 90               # frame budget is capped so the picture refreshes at least 90x/sec


@dataclass
class Snapshot:
    feats: audio.Features
    scene: str | None
    frame: np.ndarray
    tripped: bool
    log: list


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

    # source: audio.FilePlayer or audio.LiveInput; kind: "preview" | "helios"
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
                budget = max(80, min(int(s.points), s.pps // max(s.min_fps, 1)))
                paths, scene = eng.update(feats)
                safety.brightness = s.brightness
                if s.blackout:
                    frame = blank_frame(budget)
                else:
                    frame = safety.apply(compose(paths, budget, size=s.size))
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
