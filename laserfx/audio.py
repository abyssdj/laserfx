"""Audio input + music analysis.

Analyzer is a streaming (real-time capable) analyser that tracks:
  * BPM and beat phase (onset-strength autocorrelation + comb phase fit)
  * bar position (downbeat guessed from bass energy on each beat of the bar)
  * kick pulses and smoothed bass / mid / high levels
  * song sections: MAIN, BREAKDOWN, BUILD, DROP (and SILENCE)

The section logic is heuristic (it works on relative bass / kick energy against a
rolling reference), so thresholds live in SectionConfig for tuning.
"""
from __future__ import annotations

import collections
import math
import queue
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass, field

import numpy as np

try:  # PortAudio may be missing; file analysis still works without it
    import sounddevice as sd
except Exception:  # noqa: BLE001
    sd = None

SILENCE, MAIN, BREAKDOWN, BUILD, DROP = "SILENCE", "MAIN", "BREAKDOWN", "BUILD", "DROP"


# --------------------------------------------------------------------------- features
@dataclass
class Features:
    time: float = 0.0
    bpm: float = 120.0
    locked: bool = False          # True once a tempo estimate exists
    beat_phase: float = 0.0       # 0 just after a beat -> 1 just before the next
    beat_pulse: float = 0.0       # exp decay from each beat, 1 -> 0
    beat_count: int = 0
    bar_pos: int = 0              # 0..3, 0 = guessed downbeat
    level: float = 0.0            # loudness relative to recent peak, ~0..1
    bass: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    kick: float = 0.0             # 1 on a kick onset, decays
    section: str = MAIN
    section_time: float = 0.0
    section_beats: int = 0
    build: float = 0.0            # 0..1 build-up progress (only in BREAKDOWN/BUILD)
    wave: np.ndarray = field(default_factory=lambda: np.zeros(512, np.float32))


@dataclass
class SectionConfig:
    breakdown_bass: float = 0.40     # bass below this fraction of reference ...
    breakdown_kick: float = 0.20     # ... or kick-flux below this => "low"
    breakdown_hold: float = 1.5      # seconds "low" must persist
    drop_bass: float = 0.60          # fast bass ratio needed to call a drop
    drop_kick: float = 0.35
    drop_jump: float = 2.2           # or bass jump vs 4 s ago (intro -> drop)
    drop_beats: int = 16             # DROP lasts this long before becoming MAIN
    build_hf: float = 1.8            # high-band growth over breakdown minimum
    min_section: float = 2.0
    silence_rms: float = 0.002


class _AutoGain:
    def __init__(self, dt, tau=15.0, floor=1e-6):
        self.k = math.exp(-dt / tau)
        self.ref = floor
        self.floor = floor

    def __call__(self, x):
        self.ref = max(self.ref * self.k, x, self.floor)
        return x / self.ref


class Analyzer:
    def __init__(self, sr=44100, hop=512, n_fft=1024, cfg: SectionConfig | None = None):
        self.sr, self.hop, self.n_fft = sr, hop, n_fft
        self.dt = hop / sr
        self.cfg = cfg or SectionConfig()
        self.win = np.hanning(n_fft).astype(np.float32)
        fr = np.fft.rfftfreq(n_fft, 1 / sr)
        self.i_bass = (fr >= 30) & (fr < 150)
        self.i_mid = (fr >= 150) & (fr < 2000)
        self.i_high = (fr >= 2000) & (fr < 12000)
        self.i_flux = fr < 10000
        self.i_low = fr < 200

        self.buf = np.zeros(0, np.float32)
        self.prev = None
        self.frame = 0
        self.env = collections.deque(maxlen=int(8 / self.dt))
        self.last_frame = np.zeros(n_fft, np.float32)

        # tempo / beat tracking
        self.bpm = 120.0
        self.locked = False
        self.period = 60 / self.bpm / self.dt          # frames per beat
        self.next_beat = self.period
        self.beat_count = 0
        self._cand, self._cand_n = 0.0, 0
        self._bar_e = np.zeros(4)
        self._bflux = collections.deque(maxlen=3)
        self.bpm_history: list[float] = []

        # smoothers
        ak = lambda tau: 1 - math.exp(-self.dt / tau)  # noqa: E731
        self.a_fast, self.a_slow = ak(0.15), ak(0.6)
        self.loud_s = self.bass_s = self.bass_f = self.high_s = self.mid_s = 0.0
        self.lowflux_s = self.lowflux_f = 0.0
        self.kick = 0.0
        self._kick_decay = math.exp(-self.dt / 0.12)
        self._kick_hist = collections.deque(maxlen=int(2 / self.dt))
        self._last_kick = -1.0
        self.g_bass, self.g_mid, self.g_high, self.g_loud = (_AutoGain(self.dt) for _ in range(4))

        # sections
        self.tick_n = max(1, int(0.1 / self.dt))
        self.h_bass = collections.deque(maxlen=900)   # 90 s @ 10 Hz
        self.h_loud = collections.deque(maxlen=900)
        self.h_flux = collections.deque(maxlen=900)
        self.ref_b = self.ref_l = self.ref_f = 1e-9
        self.section = MAIN
        self._sec_t0, self._sec_beat0 = 0.0, 0
        self._low_t = self._sil_t = 0.0
        self._hf_min = math.inf
        self.build = 0.0
        self.log: list[tuple[float, str]] = [(0.0, MAIN)]

    # ------------------------------------------------------------------ public
    def process(self, samples) -> Features:
        x = np.asarray(samples, np.float32).ravel()
        if x.size:
            self.buf = np.concatenate([self.buf, x]) if self.buf.size else x
        pos = 0
        while len(self.buf) - pos >= self.n_fft:
            self._step(self.buf[pos:pos + self.n_fft])
            pos += self.hop
        self.buf = self.buf[pos:]
        return self.features()

    def features(self) -> Features:
        now = self.frame * self.dt
        ph = 1 - (self.next_beat - self.frame) / self.period
        ph = min(max(ph, 0.0), 0.999)
        down = int(np.argmax(self._bar_e))
        return Features(
            time=now, bpm=self.bpm, locked=self.locked, beat_phase=ph,
            beat_pulse=math.exp(-3.0 * ph), beat_count=self.beat_count,
            bar_pos=(self.beat_count - down) % 4,
            level=min(self.loud_s / self.g_loud.ref, 1.5),
            bass=min(self.bass_f / self.g_bass.ref, 1.5),
            mid=min(self.mid_s / self.g_mid.ref, 1.5),
            high=min(self.high_s / self.g_high.ref, 1.5),
            kick=self.kick, section=self.section, section_time=now - self._sec_t0,
            section_beats=self.beat_count - self._sec_beat0, build=self.build,
            wave=self.last_frame[-512:].copy(),
        )

    # ----------------------------------------------------------------- per hop
    def _step(self, x):
        spec = np.abs(np.fft.rfft(x * self.win)) / self.n_fft
        rms = float(np.sqrt(np.mean(x * x)))
        band = lambda m: float(np.sqrt(np.mean(spec[m] ** 2)))  # noqa: E731
        bass, mid, high = band(self.i_bass), band(self.i_mid), band(self.i_high)
        lm = np.log1p(spec * 1000.0)
        if self.prev is None:
            f_all = f_low = 0.0
        else:
            d = np.maximum(lm - self.prev, 0.0)
            f_all, f_low = float(d[self.i_flux].sum()), float(d[self.i_low].sum())
        self.prev = lm
        self.last_frame = x
        self.env.append(f_all + 2.0 * f_low)
        self._bflux.append(f_low)

        fa, sa = self.a_fast, self.a_slow
        self.loud_s += sa * (rms - self.loud_s)
        self.bass_s += sa * (bass - self.bass_s)
        self.bass_f += fa * (bass - self.bass_f)
        self.mid_s += sa * (mid - self.mid_s)
        self.high_s += sa * (high - self.high_s)
        self.lowflux_s += sa * (f_low - self.lowflux_s)
        self.lowflux_f += fa * (f_low - self.lowflux_f)
        self.g_bass(self.bass_f); self.g_mid(self.mid_s); self.g_high(self.high_s); self.g_loud(self.loud_s)

        # kick detection (adaptive threshold on low-band flux)
        now = self.frame * self.dt
        self.kick *= self._kick_decay
        self._kick_hist.append(f_low)
        if len(self._kick_hist) > 20:
            h = np.fromiter(self._kick_hist, float)
            thr = h.mean() + 1.2 * h.std()
            if f_low > thr and f_low > 0.5 and now - self._last_kick > 0.12:
                self.kick, self._last_kick = 1.0, now

        # beat clock
        if self.frame >= self.next_beat:
            self.beat_count += 1
            self.next_beat += self.period
            self._bar_e[self.beat_count % 4] += max(self._bflux)
            self._bar_e *= 0.995

        self.frame += 1
        if self.frame % self.tick_n == 0:
            self._tick()
        if self.frame % int(0.5 / self.dt) == 0 and len(self.env) > 4 / self.dt:
            self._update_tempo()
        self._update_section(self.frame * self.dt)

    def _tick(self):
        self.h_bass.append(self.bass_s)
        self.h_loud.append(self.loud_s)
        self.h_flux.append(self.lowflux_s)
        if len(self.h_bass) >= 10:
            self.ref_b = max(float(np.percentile(self.h_bass, 90)), 1e-9)
            self.ref_l = max(float(np.percentile(self.h_loud, 90)), 1e-9)
            self.ref_f = max(float(np.percentile(self.h_flux, 90)), 1e-9)

    # ------------------------------------------------------------------ tempo
    def _update_tempo(self):
        dt = self.dt
        e = np.fromiter(self.env, float)
        n = len(e)
        k = max(3, int(0.4 / dt))
        e = np.maximum(e - np.convolve(e, np.ones(k) / k, mode="same"), 0.0)
        if e.max() <= 1e-6:
            return
        F = np.fft.rfft(e, 2 * n)
        ac = np.fft.irfft(F * np.conj(F))[:n] / (n - np.arange(n))
        ac /= ac[0] + 1e-12
        lo, hi = int(60 / 200 / dt), int(60 / 65 / dt)
        lags = np.arange(lo, hi + 1)
        score = ac[lags].copy()
        for m, w in ((2, 0.5), (3, 0.25), (4, 0.35)):
            idx = lags * m
            ok = idx < n
            score[ok] += w * ac[idx[ok]]
        bpm_l = 60 / (lags * dt)
        score *= np.exp(-0.5 * (np.log2(bpm_l / 125.0) / 0.9) ** 2)
        i = int(np.argmax(score))
        lag = float(lags[i])
        if 0 < i < len(score) - 1:                      # parabolic refine
            a, b, c = score[i - 1], score[i], score[i + 1]
            den = a - 2 * b + c
            if den != 0:
                lag += 0.5 * (a - c) / den
        cand = 60 / (lag * dt)

        # accept / smooth tempo
        if not self.locked:
            self._set_tempo(cand, hard=True)
            self.locked = True
        elif abs(cand - self.bpm) / self.bpm < 0.025:
            self._set_tempo(self.bpm + 0.3 * (cand - self.bpm))
            self._cand_n = 0
        else:
            if self._cand and abs(cand - self._cand) / self._cand < 0.025:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = cand, 1
            octave = abs(math.log2(cand / self.bpm)) % 1.0 < 0.05 or abs(math.log2(cand / self.bpm)) > 0.95
            if self._cand_n >= (6 if octave else 3):
                self._set_tempo(cand, hard=True)
                self._cand_n = 0
        self.bpm_history.append(self.bpm)

        # beat phase: comb fit of the onset envelope at the current period
        p = self.period
        K = int(n / p)
        if K < 4:
            return
        offs = np.arange(0, p, 0.5)
        idx = np.clip(np.round(offs[:, None] + p * np.arange(K)[None, :]).astype(int), 0, n - 1)
        w = np.linspace(0.5, 1.0, K)[None, :]
        o = float(offs[int(np.argmax((e[idx] * w).sum(1)))])
        base = self.frame - n + o
        meas = base + math.ceil((self.frame - base) / p) * p
        if self._phase_reset:
            self.next_beat, self._phase_reset = meas, False
        else:
            err = (meas - self.next_beat + p / 2) % p - p / 2
            self.next_beat += 0.3 * err

    _phase_reset = True

    def _set_tempo(self, bpm, hard=False):
        self.bpm = float(bpm)
        self.period = 60 / self.bpm / self.dt
        if hard:
            self._phase_reset = True
            self._bar_e[:] = 0

    # --------------------------------------------------------------- sections
    def _update_section(self, now):
        c, dt = self.cfg, self.dt
        warm = len(self.h_bass) >= 80
        rb = self.bass_s / self.ref_b
        rbf = self.bass_f / self.ref_b
        rf = self.lowflux_s / self.ref_f
        rff = self.lowflux_f / self.ref_f

        silent = self.loud_s < c.silence_rms
        self._sil_t = self._sil_t + dt if silent else 0.0
        low = warm and (rb < c.breakdown_bass or rf < c.breakdown_kick)
        self._low_t = self._low_t + dt if low else 0.0
        st, age = self.section, now - self._sec_t0
        new = None

        if self._sil_t > 1.0 and st != SILENCE:
            new = SILENCE
        elif st == SILENCE:
            if not silent:
                new = MAIN
        elif st in (MAIN, DROP):
            ago = self.h_bass[-40] if len(self.h_bass) > 40 else None
            if self._low_t > c.breakdown_hold and age > c.min_section:
                new = BREAKDOWN
            elif (st == MAIN and warm and age > 3 and ago and self.bass_s > c.drop_jump * ago and rb > 0.7):
                new = DROP
            elif st == DROP and (self.beat_count - self._sec_beat0 >= c.drop_beats or age > 12):
                new = MAIN
        else:  # BREAKDOWN / BUILD
            self._hf_min = min(self._hf_min, self.high_s)
            grow = self.high_s / max(self._hf_min, 1e-9)
            self.build = float(min(max((grow - 1.0) / 4.0, 0.0), 1.0)) if age > 2 else 0.0
            if age > 1.0 and rbf > c.drop_bass and rff > c.drop_kick:
                new = DROP
            elif st == BREAKDOWN and age > 3 and grow > c.build_hf:
                new = BUILD
        if new and new != st:
            self.section, self._sec_t0, self._sec_beat0 = new, now, self.beat_count
            self._hf_min = math.inf
            if new not in (BREAKDOWN, BUILD):
                self.build = 0.0
            self.log.append((now, new))


# --------------------------------------------------------------------- audio I/O
def load_audio(path, sr=44100) -> np.ndarray:
    """Mono float32 at `sr`. Plain PCM wavs need nothing; other formats use ffmpeg."""
    try:
        with wave.open(path, "rb") as w:
            ch, sw, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
        if sw == 2:
            x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
        elif sw == 4:
            x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
        elif sw == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
            v = (b[:, 0].astype(np.int32) | b[:, 1].astype(np.int32) << 8 | b[:, 2].astype(np.int32) << 16)
            x = np.where(v >= 2**23, v - 2**24, v).astype(np.float32) / 2**23
        elif sw == 1:
            x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
        else:
            raise wave.Error("unsupported width")
        x = x.reshape(-1, ch).mean(1)
        if rate != sr:
            t = np.arange(int(len(x) * sr / rate)) * rate / sr
            x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
        return x
    except (wave.Error, EOFError):
        if not shutil.which("ffmpeg"):
            raise RuntimeError("Not a plain PCM .wav and ffmpeg is not installed; convert to wav or install ffmpeg")
        out = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, np.float32).copy()


class FilePlayer:
    """Plays a loaded file against the wall clock and hands the analyser the audio up to 'now'."""

    def __init__(self, x, sr, play=True, lead=0.0):
        self.x, self.sr, self.play, self.lead = x, sr, play and sd is not None, lead
        self.pos, self.done, self.t0 = 0, False, 0.0

    def start(self):
        if self.play:
            sd.play(self.x, self.sr)
        self.t0 = time.perf_counter()

    def read(self):
        t = max(0.0, time.perf_counter() - self.t0 + self.lead)
        end = min(int(t * self.sr), len(self.x))
        chunk, self.pos = self.x[self.pos:end], max(self.pos, end)
        self.done = self.pos >= len(self.x)
        return chunk

    def close(self):
        if self.play:
            sd.stop()


class LiveInput:
    """Line-in / microphone / loopback device via sounddevice."""

    def __init__(self, device=None, sr=44100):
        if sd is None:
            raise RuntimeError("sounddevice/PortAudio not available: pip install sounddevice (and install PortAudio)")
        self.sr, self.device, self.q, self.stream = sr, device, queue.Queue(), None
        self.done = False

    def start(self):
        def cb(indata, frames, t, status):  # noqa: ARG001
            self.q.put(indata[:, :2].mean(1).copy() if indata.shape[1] > 1 else indata[:, 0].copy())
        ch = 2 if sd.query_devices(self.device, "input")["max_input_channels"] >= 2 else 1
        self.stream = sd.InputStream(samplerate=self.sr, device=self.device, channels=ch, blocksize=512, callback=cb)
        self.stream.start()

    def read(self):
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def close(self):
        if self.stream:
            self.stream.stop(); self.stream.close()


def analyze_file(x, sr=44100, cfg=None):
    """Run the streaming analyser over a whole signal faster than real time."""
    a = Analyzer(sr, cfg=cfg)
    step = sr // 4
    for i in range(0, len(x), step):
        a.process(x[i:i + step])
    return a
