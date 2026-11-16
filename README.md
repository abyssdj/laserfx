# laserfx – sound-to-light laser FX for the Helios DAC

Listens to music (file or live input), tracks **BPM / beats / bars**, detects **MAIN, BREAKDOWN, BUILD and DROP**
sections, generates beat-synced laser scenes and streams them to a **Helios USB DAC**.

## Safety first
Lasers can blind people and start fires. Physically mask the beam path (aperture cover / foil tape), keep
the beam out of audience areas, and have a hardware stop button. The software masks and limiters here are not a substitute.
* `--output helios` refuses to start without `--arm`.
* Default size 0.6 and brightness 0.5; raise only after tuning scanners.
* A static-beam guard blanks any lit, near-stationary pattern. Output blanks on silence and on exit/Ctrl+C.
* Check `--pps` against your scanners' rating; large scan angles at 30 kpps can overdrive cheap galvos.

## Scenes by section
* **BREAKDOWN** - slow and smooth: `sky_fan` (swaying canopy of beams from floor and ceiling), `sky_cross` (two fans sweeping
  through each other), `scan_sweep` (rippling scan lines with fading echoes), plus `breath` and `spiral_slow`.
* **BUILD** - accelerating: `riser` (contracting ring + spinning rays), `build_fan`, `build_scan`. Speed ramps with the detected
  build amount and fades from blue to white/red.
* **DROP** - strobing geometric beams, snapped to the beat: `strobe_poly` (nested triangle/square/hexagon/pentagon),
  `strobe_beams` (horizontal, vertical, diagonal, box+cross bar sets), `strobe_burst`, plus `drop_blast`.
* **MAIN** - tunnel, beam fan, starburst, lissajous, grid, spiral, scope, rings.
Strobing flashes at up to 2 flashes per beat (about 4 Hz at 128 BPM). Flashing light can trigger seizures in photosensitive people:
untick **Strobe effects** in the GUI (or use `--no-strobe`) for such audiences and warn your audience as required at your venue.
Force any scene or section from the GUI to rehearse.

## Scanner-friendly rendering
* **Detail** slider (GUI) / `--detail` (0.2 sparse .. 1.0 busy, default 0.5) scales how many beams, rings and rays each scene draws.
  Typical scene now uses ~6 paths instead of ~9; busy scenes like starburst went from 22 paths to 10.
* Frame size is capped at `pps / 30` points so the picture always refreshes at 30 fps or better (default 700 points at 30 kpps = 43 fps);
  lower your `--pps` for slower scanners and the budget shrinks automatically.
* Paths are re-ordered to minimise blanked travel and dropped lowest-priority-first if blanking would exceed 40% of the frame.
* Jumps between shapes are eased (smooth start/stop) and spread over enough points that peak speed per point stays under a cap;
  corners hold points in proportion to how sharp they are, curves hold none.
* `python tools/scanner_stats.py 700 30000` prints paths, blanking % and peak step per scene so you can tune for your scanners.

## Standalone app (GUI)
Run with no arguments (`python -m laserfx`, or the packaged **LaserFX** app) to open the GUI: a dark, flat, rounded interface
drawn on one tkinter canvas (no extra packages; scales on high-DPI screens). Left: live laser preview with glow, big BPM,
section badge, beat 1-2-3-4 indicator, bass/mid/high meters and a section timeline. Right: **Setup** (audio file or live input,
preview or Helios output, ARM switch), **Look** (brightness, size, detail, strobe, scene and section overrides) and
**Scanner** (points per second / per frame with live refresh rate, flip / swap). Start/Stop and BLACKOUT (space bar or Esc) are always visible.
`python tools/gui_preview.py track.wav outdir` renders screenshots of the GUI without a display (uses a strict fake tkinter + Pillow).
**Build the single-file app:** `pip install numpy sounddevice pyusb libusb-package pyinstaller` then `python build.py`
(PyInstaller builds only for the OS it runs on). Or push this folder to GitHub: the included Action builds Windows, macOS and
Linux executables as downloadable artifacts. The result bundles Python, numpy, libusb and PortAudio, so end users install nothing.
The Helios is driven directly over USB (no vendor DLL). Linux needs the Helios udev rule; Windows/macOS are plug-and-play.
File loading: WAV always; mp3/flac need ffmpeg on PATH. For music playing on the computer use Live input with a loopback / "Stereo Mix" device.

## Install (running from source)
    pip install numpy matplotlib sounddevice      # sounddevice optional (live input + playback; needs PortAudio)
    pip install pyusb libusb-package              # Helios USB driver (built in; no vendor DLL). Optionally --helios-lib for the vendor library
On Linux add the udev rule from the Helios docs
(or run as root); Windows/macOS are plug-and-play. mp3/flac etc. need `ffmpeg` on PATH (wav works without it).

## Use
    python -m laserfx song.wav --analyze                 # BPM + section timeline, no output
    python -m laserfx --gallery scenes.png               # contact sheet of every scene
    python -m laserfx song.wav                           # play + matplotlib laser simulator
    python -m laserfx song.wav --output helios --arm --size 0.5 --brightness 0.4
    python -m laserfx --live --device 2 --output helios --arm   # line-in / loopback
    python -m laserfx --list-devices
Useful flags: `--pps`, `--points`, `--flip-x/--flip-y/--swap-xy`, `--scene NAME`, `--lead MS` (compensate audio output latency).
`python tools/make_test_track.py out.wav` makes a 128 BPM track with a known structure for testing.

## How it works
* `audio.py` – streaming analyser: onset-strength autocorrelation for BPM (with octave-flip protection), comb-fit beat phase,
  downbeat guess from bass-on-beat, kick detector. Sections come from bass level and kick activity relative to a rolling
  90th-percentile reference: low for 1.5 s ⇒ BREAKDOWN; rising highs inside it ⇒ BUILD; kick/bass returning ⇒ DROP (16 beats) ⇒ MAIN.
  Thresholds are in `SectionConfig`.
* `fx.py` – scenes are functions of beat time, energy and section; the engine picks scenes per section and changes on bar lines.
  Add a scene by writing `def sc_x(ctx) -> [Path]` and registering it in `SCENES` / `POOLS`.
* `render.py` – resamples paths to a point budget, adds corner dwell and blanked transits, applies the safety limiter.
* `output.py` – Helios via ctypes (12-bit XY, 8-bit RGB, waits for ready then `WriteFrame`), simulator, null output.

## Ideas borrowed / next steps (from Liberation + Libera)
Liberation is built on BPM-timed looping clips, per-laser zones and masks, and chainable effects; Libera is its open-source
C++ layer for Helios, Ether Dream, IDN and LaserCube. Natural extensions: more outputs behind the same `send(frame)`
interface (Ether Dream/IDN), zones/masks and per-laser geometry correction, MIDI control, (a manual section override now exists in the GUI).

## Status
Tested here: analysis on a synthetic track, scene rendering, loop speed, the Helios packet encoding against a mock USB device,
and the GUI's engine thread. **Not tested:** the tkinter window itself and the PyInstaller build (neither tool was available
in my workspace), real Helios hardware, live audio input, and real-world music; section thresholds will probably need tuning on your tracks.
