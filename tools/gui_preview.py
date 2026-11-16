"""Render the GUI to PNG without a display (uses the strict fake tkinter + Pillow).
    python tools/gui_preview.py track.wav outdir
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "fake_tk"))          # fake tkinter first
sys.path.insert(0, os.path.join(here, ".."))
import numpy as np
import tkinter as tk
from laserfx import audio, gui
from laserfx.fx import FXEngine
from laserfx.render import SafetyLimiter, compose
from laserfx.session import Snapshot

track, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
x = audio.load_audio(track)
root = tk.Tk()
app = gui.App(root, 1.0)
W, H = 1100, 700

def shot(name):
    app.canvas.render(W, H).save(os.path.join(out, name + ".png"))
    print("saved", name)

def run_to(t, scene=None):
    an, eng, sf = audio.Analyzer(), FXEngine(seed=2), SafetyLimiter(brightness=0.9)
    step = 44100 // 30
    f = None
    for i in range(0, int(t * 44100), step):
        f = an.process(x[i:i + step]); paths, sc = eng.update(f)
    if scene:
        eng.force = scene; eng._pick(); paths, sc = eng.update(f)
    fr = sf.apply(compose(paths, 700, size=0.8))
    app.session.snapshot = Snapshot(f, sc, fr, False, list(an.log))
    app.session.running = True
    app.btn_start.set("Stop", "danger"); app.status.set("Running · preview", gui.C["ok"])
    app._tick()

shot("1_idle")
run_to(34.0, "tunnel"); shot("2_main_setup")
run_to(24.0, "sky_fan"); shot("3_breakdown")
run_to(29.5, "build_fan"); shot("4_build")
run_to(31.5, "drop_blast"); app.tab = "look"; app._apply_visibility(); shot("5_drop_look")
app.tab = "scanner"; app._apply_visibility(); shot("6_scanner")
app.tab = "look"; app._apply_visibility()
dd = app.look[4]; dd.press(dd.x2 - 50, dd.y1 + 10); shot("7_dropdown")
app.ui.close_popup(); app.tab = "setup"; app.src_mode = "live"; app._apply_visibility(); app._toggle_blackout(); run_to(36.0, "starburst"); shot("8_live_blackout")
