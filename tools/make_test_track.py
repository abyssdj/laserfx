"""Synthesize a 128 BPM test track with a known structure.

 bars 0-8   MAIN       kick + hats + bass
 bars 8-14  BREAKDOWN  pad only
 bars 14-16 BUILD      pad + rising noise + snare roll
 bars 16-24 DROP/MAIN  kick + bass + lead (louder)
 bars 24-32 MAIN       kick + hats
"""
import sys, wave
import numpy as np

sr, bpm = 44100, 128
beat = 60 / bpm
n = int(32 * 4 * beat * sr)
t = np.arange(n) / sr
rng = np.random.default_rng(0)
x = np.zeros(n)

def add(sig, start):
    i = int(start * sr); j = min(n, i + len(sig))
    x[i:j] += sig[: j - i]

kt = np.arange(int(0.25 * sr)) / sr
kick = np.sin(2 * np.pi * (45 * kt + 90 * (1 - np.exp(-kt * 30)) / 30)) * np.exp(-kt * 12)
ht = np.arange(int(0.05 * sr)) / sr
hat = rng.normal(0, 1, len(ht)) * np.exp(-ht * 90) * 0.25
hat = np.diff(hat, prepend=0)
bar = 4 * beat
def bars(a, b): return range(int(a * 4), int(b * 4))

for b in bars(0, 8): add(kick, b * beat); add(hat, b * beat + beat / 2)
for b in bars(8, 14): pass
for b in bars(16, 32): add(kick * 1.2, b * beat)
for b in bars(16, 24): add(hat, b * beat + beat / 2)
for b in bars(24, 32): add(hat, b * beat + beat / 2)
bass = np.sin(2 * np.pi * 55 * t) * 0.3
for a, bnd in ((0, 8), (16, 24)):
    x[int(a * bar * sr):int(bnd * bar * sr)] += bass[int(a * bar * sr):int(bnd * bar * sr)]
pad = sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330)) * 0.05
s, e = int(8 * bar * sr), int(16 * bar * sr)
x[s:e] += pad[s:e]
# riser
rs, re_ = int(14 * bar * sr), int(16 * bar * sr)
env = np.linspace(0, 1, re_ - rs) ** 2
x[rs:re_] += np.diff(rng.normal(0, 1, re_ - rs), prepend=0) * env * 0.25
for k in range(int(14 * 4 * 2), int(16 * 4 * 2)):
    add(hat * 2, k * beat / 2)
# lead on drop
lead = np.sign(np.sin(2 * np.pi * 440 * t)) * 0.06
ls, le = int(16 * bar * sr), int(24 * bar * sr)
x[ls:le] += lead[ls:le]
x = x / np.abs(x).max() * 0.8
out = sys.argv[1] if len(sys.argv) > 1 else "test_track.wav"
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((x * 32767).astype("<i2").tobytes())
print("wrote", out, f"{n / sr:.1f}s")
