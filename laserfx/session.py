"""Background engine used by the GUI: audio -> analysis -> FX (+Art-Net RGB) -> zone -> galvo physics -> output."""
from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import audio, output
from .artnet import ArtNetReceiver, ArtNetState
from .fx import FXEngine, Path, render_test_pattern
from .render import SafetyLimiter, blank_frame, compose


@dataclass
class Settings:
    brightness: float = 0.55
    size: float = 0.70
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    keystone: float = 0.0
    points: int = 260
    pps: int = 30000
    flip_x: bool = False
    flip_y: bool = False
    swap_xy: bool = False
    scene: str | None = None
    section: str | None = None
    harmony_mode: str = "auto"      # "auto" | "complementary" | "split_complementary" | "rectangular" | "triadic" | "analogous"
    test_pattern: str = "off"
    blackout: bool = False
    strobe: bool = True
    detail: float = 0.55
    min_fps: int = 80
    galvo_lag: int = 2
    corner_dwell: int = 4
    blank_delay: int = 2
    corner_brake: float = 0.60
    # Art-Net DMX512 RGB Override settings
    artnet_enabled: bool = False
    artnet_net: int = 0
    artnet_subnet: int = 0
    artnet_universe: int = 0
    artnet_channel: int = 1
    artnet_dual_rgb: bool = False


@dataclass
class Snapshot:
    feats: audio.Features
    scene: str | None
    frame: np.ndarray
    tripped: bool
    log: list
    artnet: ArtNetState | None = None


def apply_zone_geometry(
    paths: list[Path],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    keystone: float = 0.0,
) -> list[Path]:
    if not paths:
        return paths
    sx = min(max(float(scale_x), 0.1), 1.0)
    sy = min(max(float(scale_y), 0.1), 1.0)
    ox = min(max(float(offset_x), -0.85), 0.85)
    oy = min(max(float(offset_y), -0.85), 0.85)
    ks = min(max(float(keystone), -0.5), 0.5)

    if abs(sx - 1.0) < 1e-4 and abs(sy - 1.0) < 1e-4 and abs(ox) < 1e-4 and abs(oy) < 1e-4 and abs(ks) < 1e-4:
        return paths

    out: list[Path] = []
    for p in paths:
        pts = np.asarray(p.pts, dtype=float)
        x = pts[:, 0]
        y = pts[:, 1]
        if abs(ks) > 1e-4:
            x = x * (1.0 + ks * y)
        nx = np.clip(x * sx + ox, -1.0, 1.0)
        ny = np.clip(y * sy + oy, -1.0, 1.0)
        out.append(Path(np.stack([nx, ny], axis=1), p.rgb, p.dwell))
    return out


def prepare_paths_for_galvos(
    paths: list[Path],
    corner_dwell: int = 0,
    corner_brake: float = 0.0,
) -> list[Path]:
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

        span = float(np.hypot(*(pts[-1] - pts[0]))) if n == 2 else 1.0
        if n == 2 and span < 0.02:
            out.append(Path(pts, p.rgb, max(p.dwell, 1)))
            continue

        if brake > 0.05:
            eased_pts = []
            eased_rgb = []
            is_per_v = (np.asarray(p.rgb).ndim == 2)
            rgb_arr = np.asarray(p.rgb, dtype=float)
            steps = 5
            u_lin = np.linspace(0.0, 1.0, steps + 1)
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
                is_corner = (0 < idx < len(new_pts) - 1) and (turn_cos[idx - 1] < 0.82)
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
    if len(frame) < 3 or (galvo_lag == 0 and blank_delay <= 0):
        return frame

    pts = frame.copy()

    if blank_delay > 0:
        lit = pts[:, 2:5].max(axis=1) > 0.01
        d = int(blank_delay)
        for i in range(1, len(pts)):
            if lit[i] and not lit[i - 1]:
                end = min(len(pts), i + d)
                pts[i:end, 2:5] = 0.0

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
    def __init__(self, settings: Settings | None = None, sr: int = 44100):
        self.settings = settings or Settings()
        self.sr = sr
        self.snapshot: Snapshot | None = None
        self.error: str | None = None
        self.running = False
        self.finished = False
        self._stop = threading.Event()
        self._thread = None

        self.artnet = ArtNetReceiver()
        self._tap_times: list[float] = []
        self._manual_bpm: float | None = None
        self._tap_anchor: float = 0.0

    def sync_artnet_config(self):
        s = self.settings
        self.artnet.enabled = bool(s.artnet_enabled)
        self.artnet.net = int(s.artnet_net)
        self.artnet.subnet = int(s.artnet_subnet)
        self.artnet.universe = int(s.artnet_universe)
        self.artnet.start_channel = int(s.artnet_channel)
        self.artnet.dual_rgb_mode = bool(s.artnet_dual_rgb)
        if self.artnet.enabled:
            self.artnet.start()
        else:
            self.artnet.stop()

    def tap_tempo(self):
        now = time.perf_counter()
        if self._tap_times and (now - self._tap_times[-1]) > 2.2:
            self._tap_times.clear()
        self._tap_times.append(now)
        if len(self._tap_times) > 8:
            self._tap_times.pop(0)
        self._tap_anchor = now
        if len(self._tap_times) >= 2:
            intervals = np.diff(self._tap_times)
            avg_dt = float(np.median(intervals))
            if 0.25 <= avg_dt <= 1.5:
                self._manual_bpm = 60.0 / avg_dt

    def clear_manual_bpm(self):
        self._manual_bpm = None
        self._tap_times.clear()

    @property
    def manual_bpm(self) -> float | None:
        return self._manual_bpm

    def start(self, source, kind: str = "preview"):
        s = self.settings
        self.sync_artnet_config()
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
                self.sync_artnet_config()
                artnet_state = self.artnet.get_active_override()

                chunk = src.read()
                if len(chunk):
                    feats = an.process(chunk)

                if self._manual_bpm is not None:
                    elapsed = time.perf_counter() - self._tap_anchor
                    total_beats = elapsed * (self._manual_bpm / 60.0)
                    b_cnt = int(total_beats)
                    b_ph = total_beats - b_cnt
                    feats = dataclasses.replace(
                        feats,
                        bpm=self._manual_bpm,
                        locked=True,
                        beat_count=b_cnt,
                        beat_phase=b_ph,
                        bar_pos=b_cnt % 4,
                        beat_pulse=max(0.0, 1.0 - b_ph * 2.5),
                    )

                if s.section:
                    feats = dataclasses.replace(feats, section=s.section)
                if s.scene != last_force:
                    eng.force, last_force = s.scene, s.scene
                    eng.section = feats.section
                    eng._pick()
                eng.strobe_enabled = s.strobe
                eng.detail = s.detail
                eng.harmony_mode = s.harmony_mode
                budget = max(90, min(int(s.points), s.pps // max(s.min_fps, 1)))

                if s.test_pattern and s.test_pattern != "off":
                    paths = render_test_pattern(s.test_pattern, feats.beat_count + feats.beat_phase)
                    scene = f"TEST · {s.test_pattern.upper()}"
                else:
                    paths, scene = eng.update(feats, artnet=artnet_state)

                safety.brightness = s.brightness
                if s.blackout:
                    frame = blank_frame(budget)
                else:
                    zoned_paths = apply_zone_geometry(
                        paths,
                        scale_x=float(s.scale_x),
                        scale_y=float(s.scale_y),
                        offset_x=float(s.offset_x),
                        offset_y=float(s.offset_y),
                        keystone=float(s.keystone),
                    )
                    shaped_paths = prepare_paths_for_galvos(
                        zoned_paths,
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
                self.snapshot = Snapshot(feats, scene, frame, safety.tripped, cached_log, artnet=artnet_state)

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