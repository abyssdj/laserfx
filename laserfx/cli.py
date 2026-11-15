"""Command line entry point:  python -m laserfx ..."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import audio
from .audio import BREAKDOWN, BUILD, DROP, MAIN, Features
from .fx import POOLS, SCENES, Ctx, FXEngine
from .render import SafetyLimiter, compose


def build_parser():
    p = argparse.ArgumentParser(prog="laserfx", description="Sound-to-light laser FX generator for the Helios DAC")
    p.add_argument("input", nargs="?", help="audio file (wav; mp3/flac/etc. need ffmpeg). Omit with --live")
    p.add_argument("--live", action="store_true", help="analyse a live input device instead of a file")
    p.add_argument("--device", help="audio input device (index or name)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--analyze", action="store_true", help="print BPM + section timeline for the file and exit")
    p.add_argument("--gallery", metavar="PNG", help="render a contact sheet of all scenes and exit")
    p.add_argument("--output", choices=["sim", "helios", "null"], default="sim")
    p.add_argument("--arm", action="store_true", help="required for --output helios: confirms beams are physically masked")
    p.add_argument("--helios-lib", help="path to the HeliosDacAPI shared library")
    p.add_argument("--helios-index", type=int, default=0)
    p.add_argument("--pps", type=int, default=30000, help="points per second (scanner rating)")
    p.add_argument("--points", type=int, default=700, help="target points per frame (capped to pps/30 so refresh stays >= 30 fps)")
    p.add_argument("--size", type=float, default=0.6, help="projection size 0..1 (keep small until tuned)")
    p.add_argument("--brightness", type=float, default=0.5)
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--swap-xy", action="store_true")
    p.add_argument("--no-play", action="store_true", help="do not play the file through speakers")
    p.add_argument("--lead", type=float, default=0.0, help="ms: >0 lasers earlier, <0 later (audio output latency)")
    p.add_argument("--scene", choices=sorted(SCENES), help="force one scene")
    p.add_argument("--detail", type=float, default=0.5, help="0.2 sparse .. 1.0 busy")
    p.add_argument("--no-strobe", action="store_true", help="keep strobing drop scenes continuously lit")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sr", type=int, default=44100)
    return p


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def cmd_analyze(path, sr):
    x = audio.load_audio(path, sr)
    a = audio.analyze_file(x, sr)
    bpms = a.bpm_history[len(a.bpm_history) // 4:] or [a.bpm]
    bpm = float(np.median(bpms))
    print(f"{path}: {len(x) / sr:.1f}s   BPM ~ {bpm:.1f}")
    bar = 4 * 60 / bpm
    ends = [t for t, _ in a.log[1:]] + [len(x) / sr]
    for (t, s), e in zip(a.log, ends):
        print(f"  {mmss(t)}  {s:<10} {e - t:6.1f}s  ({(e - t) / bar:4.1f} bars)")


def cmd_gallery(path, a):
    from .output import save_png
    frames, titles = [], []
    rng = np.random.default_rng(1)
    names = sorted(SCENES)
    for name in names:
        sec = next((s for s, pool in POOLS.items() if name in pool), MAIN)
        f = Features(bpm=128, locked=True, beat_phase=0.15, beat_pulse=0.65, beat_count=9, level=0.8,
                     kick=0.6, section=sec, build=0.6,
                     wave=(np.sin(np.arange(1024) * 0.15) * 0.6 + rng.normal(0, 0.05, 1024)).astype(np.float32))
        c = Ctx(f, 9.15, 0.5, rng, rt=9.15)
        frames.append(compose(SCENES[name](c), a.points, size=1.0))
        titles.append(f"{name} ({len(frames[-1])} pts)")
    save_png(frames, titles, path)
    print("wrote", path)


def make_output(a):
    from . import output
    if a.output == "helios":
        if not a.arm:
            sys.exit("Refusing to start Helios output without --arm. Confirm the laser is physically masked, "
                     "aimed away from audiences, and that you have an e-stop, then re-run with --arm.")
        if a.helios_lib:   # optional: use the vendor shared library instead of the built-in USB driver
            d = output.HeliosDAC(a.helios_lib, a.helios_index, a.pps, a.flip_x, a.flip_y, a.swap_xy)
        else:
            d = output.HeliosUSB(a.helios_index, a.pps, a.flip_x, a.flip_y, a.swap_xy)
        n = d.open()
        print(f"Helios: {n} device(s) found, using #{a.helios_index}")
        return d
    return (output.SimDisplay if a.output == "sim" else output.NullOutput)(a.pps)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:                       # double-click / no arguments -> GUI
        from .gui import main as gui_main
        return gui_main()
    a = build_parser().parse_args(argv)
    if a.list_devices:
        if audio.sd is None:
            sys.exit("sounddevice / PortAudio not available")
        print(audio.sd.query_devices())
        return
    if a.gallery:
        return cmd_gallery(a.gallery, a)
    if a.analyze:
        if not a.input:
            sys.exit("--analyze needs an input file")
        return cmd_analyze(a.input, a.sr)
    if not a.live and not a.input:
        sys.exit("Give an audio file, or use --live")

    if a.live:
        dev = int(a.device) if a.device and a.device.isdigit() else a.device
        src = audio.LiveInput(dev, a.sr)
    else:
        x = audio.load_audio(a.input, a.sr)
        if audio.sd is None and not a.no_play:
            print("(no audio playback available: PortAudio missing; running visuals only)")
        src = audio.FilePlayer(x, a.sr, play=not a.no_play, lead=a.lead / 1000)

    an = audio.Analyzer(a.sr)
    eng = FXEngine(a.seed, a.scene)
    eng.strobe_enabled, eng.detail = not a.no_strobe, a.detail
    budget = max(100, min(a.points, a.pps // 30))
    safety = SafetyLimiter(a.brightness)
    out = make_output(a)
    if a.output != "helios":
        out.open()
    src.start()
    last_print, feats = 0.0, an.features()
    try:
        while not getattr(src, "done", False):
            chunk = src.read()
            if len(chunk):
                feats = an.process(chunk)
            paths, scene = eng.update(feats)
            frame = safety.apply(compose(paths, budget, size=a.size))
            out.send(frame)
            now = time.perf_counter()
            if now - last_print > 0.25:
                last_print = now
                print(f"\r{mmss(feats.time)}  {feats.bpm:6.1f} BPM  bar {feats.bar_pos + 1}  "
                      f"{feats.section:<9} scene={scene or '-':<14} pts={len(frame):<5}"
                      f"{' [static-beam guard]' if safety.tripped else ''}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nshutting down (blanking output)")
        src.close()
        out.close()
