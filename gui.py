"""LaserFX desktop GUI: a custom-drawn, dark, modern interface on a single tkinter Canvas (stdlib only)."""
from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from . import audio
from .audio import BREAKDOWN, BUILD, DROP, MAIN, SILENCE
from .fx import SCENES
from .session import Session, Settings
from .ui import (C, UI, W, BeatDots, Button, Card, Chip, Dropdown, Label, Meter, Pill, Preview, Segmented, Slider,
                 Stepper, Timeline, Toggle, mix)

SECTION_COLORS = {MAIN: "#3b82f6", BREAKDOWN: "#8b5cf6", BUILD: "#f59e0b", DROP: "#ef4444", SILENCE: "#475569"}
RW = 392            # right column width
M = 20              # outer margin


class Bar(W):
    """Top bar: brand mark, name, and a hairline underneath."""
    interactive = False

    def draw(self):
        d = self.ui.d
        d.rect(self.x1, self.y1, self.x2, self.y2, C["bg"], "", 0, self.tag)
        d.line([self.x1, self.y2, self.x2, self.y2], C["line"], 1, self.tag)
        x, y = self.x1 + M, self.y1 + 15
        d.rrect(x, y, x + 32, y + 32, 9, C["accent"], C["accent"], 1, self.tag)
        for dx in (-9, 0, 9):
            d.line([x + 16, y + 25, x + 16 + dx, y + 7], C["white"], 2, self.tag)
        d.text(x + 44, y + 1, "LaserFX", C["text"], self.ui.f["title"], "nw", self.tag)
        d.text(x + 44, y + 22, "sound to light", C["muted"], self.ui.f["small"], "nw", self.tag)


class App:
    def __init__(self, root, k=1.0):
        self.root = root
        root.title("LaserFX")
        root.configure(bg=C["bg"])
        root.minsize(1040, 680)
        self.canvas = tk.Canvas(root, bg=C["bg"], highlightthickness=0, bd=0, width=1100, height=700)
        self.canvas.pack(fill="both", expand=True)
        self.ui = UI(root, self.canvas, k)
        self.settings = Settings()
        self.session = Session(self.settings)
        self.samples, self.file_name = None, ""
        self.src_mode, self.out_mode, self.armed, self.tab = "file", "preview", False, "setup"
        self.devs = self._input_devices()
        self.dev_idx = self.devs[0][0] if self.devs else None
        self.play = True
        self._last_snap = None
        self._last_hud_t = 0.0
        self._build()
        self._layout(1100, 700)
        self.canvas.bind("<Configure>", lambda e: self._layout(e.width / k, e.height / k))
        root.bind("<space>", lambda e: self._toggle_blackout())
        root.bind("<Escape>", lambda e: self._toggle_blackout())
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()

    # ----------------------------------------------------------------- build
    def _build(self):
        u, s = self.ui, self.settings
        self.bar = Bar(u)
        self.status = Pill(u, "Idle", C["dim"])

        # left: preview + "now" card
        self.preview = Preview(u)
        self.now_card = Card(u, "Now")
        self.bpm_cap = Label(u, "BPM", "tiny", C["dim"])
        self.bpm = Label(u, "---", "big", C["text"])
        self.section = Pill(u, "IDLE", C["dim"])
        self.beats = BeatDots(u)
        self.meters = {n: Meter(u, n, c) for n, c in (("bass", C["accent"]), ("mid", C["accent2"]), ("high", "#f472b6"))}
        self.tl_cap = Label(u, "SECTIONS", "tiny", C["dim"])
        self.timeline = Timeline(u, SECTION_COLORS)

        # right: tabs + body card + actions
        self.tabs = Segmented(u, [("setup", "Setup"), ("look", "Look"), ("scanner", "Scanner")],
                             lambda: self.tab, lambda v: setattr(self, "tab", v), self._tab_changed)
        self.body = Card(u)
        self.setup = []
        reg = lambda lst, w: (lst.append(w), w)[1]                        # noqa: E731
        T = self.setup
        self.cap_src = reg(T, Label(u, "SOURCE", "tiny", C["dim"]))
        self.seg_src = reg(T, Segmented(u, [("file", "Audio file"), ("live", "Live input")], lambda: self.src_mode,
                                        lambda v: setattr(self, "src_mode", v), self._mode_changed))
        self.btn_file = reg(T, Button(u, "Choose file…", self._open_file))
        self.lbl_file = reg(T, Label(u, "No file selected", "body", C["muted"], "w"))
        self.dd_dev = reg(T, Dropdown(u, "Input device", [(i, n) for i, n in self.devs] or [(None, "None found")],
                                      lambda: self.dev_idx, lambda v: setattr(self, "dev_idx", v)))
        self.tg_play = reg(T, Toggle(u, "Play through speakers", lambda: self.play, lambda v: setattr(self, "play", v)))
        self.cap_out = reg(T, Label(u, "LASER OUTPUT", "tiny", C["dim"]))
        self.seg_out = reg(T, Segmented(u, [("preview", "Preview only"), ("helios", "Helios USB")], lambda: self.out_mode,
                                        lambda v: setattr(self, "out_mode", v), self._mode_changed))
        self.tg_arm = reg(T, Toggle(u, "Armed", lambda: self.armed, lambda v: setattr(self, "armed", v),
                                    "Beam masked, aimed safely, e-stop in reach"))

        L = self.look = []
        reg(L, Slider(u, "Brightness", 0, 1, lambda: s.brightness, lambda v: setattr(s, "brightness", v), "{:.0%}"))
        reg(L, Slider(u, "Size", 0.1, 1, lambda: s.size, lambda v: setattr(s, "size", v), "{:.0%}"))
        reg(L, Slider(u, "Detail", 0.2, 1, lambda: s.detail, lambda v: setattr(s, "detail", v), "{:.0%}"))
        reg(L, Toggle(u, "Strobe effects", lambda: s.strobe, lambda v: setattr(s, "strobe", v),
                      "Flashing: turn off for photosensitive audiences"))
        scenes = [("auto", "Auto")] + [(n, n.replace("_", " ").title()) for n in sorted(SCENES)]
        reg(L, Dropdown(u, "Scene", scenes, lambda: s.scene or "auto",
                        lambda v: setattr(s, "scene", None if v == "auto" else v)))
        secs = [("auto", "Auto"), (MAIN, "Main"), (BREAKDOWN, "Breakdown"), (BUILD, "Build"), (DROP, "Drop")]
        reg(L, Dropdown(u, "Section", secs, lambda: s.section or "auto",
                        lambda v: setattr(s, "section", None if v == "auto" else v)))

        S = self.scan = []
        reg(S, Stepper(u, "Points per second", lambda: s.pps, lambda v: (setattr(s, "pps", int(v)), self._fps()), 1000,
                       5000, 60000, "{:,}"))
        reg(S, Stepper(u, "Points per frame", lambda: s.points, lambda v: (setattr(s, "points", int(v)), self._fps()),
                       50, 100, 3000, "{:,}"))
        self.fps_lbl = reg(S, Label(u, "", "mid", C["ok"], "w"))
        self.cap_galvo = reg(S, Label(u, "GALVO CALIBRATION", "tiny", C["dim"]))
        reg(S, Stepper(u, "Color shift (galvo lag)", lambda: s.galvo_lag, lambda v: setattr(s, "galvo_lag", int(v)),
                       1, -10, 10, "{:+d} pts"))
        reg(S, Stepper(u, "Corner dwell", lambda: s.corner_dwell, lambda v: setattr(s, "corner_dwell", int(v)),
                       1, 0, 8, "{:d} pts"))
        reg(S, Stepper(u, "Blanking delay", lambda: s.blank_delay, lambda v: setattr(s, "blank_delay", int(v)),
                       1, 0, 8, "{:d} pts"))
        self.cap_flip = reg(S, Label(u, "ORIENTATION", "tiny", C["dim"]))
        for t, n in (("Flip X", "flip_x"), ("Flip Y", "flip_y"), ("Swap X/Y", "swap_xy")):
            reg(S, Chip(u, t, lambda n=n: getattr(s, n), lambda v, n=n: setattr(s, n, v)))

        self.btn_start = Button(u, "Start", self._start_stop, "success", "h")
        self.btn_black = Button(u, "Blackout  ·  Space", self._toggle_blackout, "outline_danger", "h")

    # ---------------------------------------------------------------- layout
    def _layout(self, W_, H_):
        u = self.ui
        u.logical_h = H_
        self.bar.place(0, 0, W_, 62)
        self.status.place(W_ - M - 190, 16, 190, 30)
        top = 62 + M
        lw = W_ - RW - 3 * M
        now_h = 208
        self.preview.place(M, top, lw, H_ - top - M - now_h - 14)
        ny = top + self.preview.h + 14
        self.now_card.place(M, ny, lw, now_h)
        self.bpm_cap.place(M + 20, ny + 38, 60, 14)
        self.bpm.place(M + 20, ny + 50, 170, 56)
        self.section.place(M + 196, ny + 54, 150, 30)
        self.beats.place(M + lw - 20 - 190, ny + 54, 190, 30)
        mw = (lw - 40 - 2 * 18) / 3
        for i, m in enumerate(self.meters.values()):
            m.place(M + 20 + i * (mw + 18), ny + 116, mw, 30)
        self.tl_cap.place(M + 20, ny + 158, 90, 12)
        self.timeline.place(M + 20, ny + 174, lw - 40, 16)

        rx = W_ - M - RW
        self.tabs.place(rx, top, RW, 40)
        by = top + 52
        bh = H_ - by - M - 2 * 48 - 2 * 12
        self.body.place(rx, by, RW, bh)
        ix, iw = rx + 20, RW - 40
        self.btn_start.place(rx, H_ - M - 48 - 12 - 48, RW, 48)
        self.btn_black.place(rx, H_ - M - 48, RW, 48)

        # setup tab
        y = by + 18
        self.cap_src.place(ix, y, iw, 12)
        self.seg_src.place(ix, y + 20, iw, 38)
        self.btn_file.place(ix, y + 72, 140, 36)
        self.lbl_file.place(ix + 154, y + 72, iw - 154, 36)
        self.dd_dev.place(ix, y + 72, iw, 36)
        self.tg_play.place(ix, y + 122, iw, 36)
        self.cap_out.place(ix, y + 184, iw, 12)
        self.seg_out.place(ix, y + 204, iw, 38)
        self.tg_arm.place(ix, y + 256, iw, 48)

        # look tab
        y = by + 14
        for i, w in enumerate(self.look[:3]):
            w.place(ix, y + i * 54, iw, 46)
        self.look[3].place(ix, y + 170, iw, 48)
        self.look[4].place(ix, y + 232, iw, 40)
        self.look[5].place(ix, y + 280, iw, 40)

        # scanner tab
        y = by + 14
        self.scan[0].place(ix, y, iw, 38)
        self.scan[1].place(ix, y + 46, iw, 38)
        self.fps_lbl.place(ix, y + 90, iw, 22)
        self.cap_galvo.place(ix, y + 122, iw, 12)
        self.scan[4].place(ix, y + 140, iw, 38)
        self.scan[5].place(ix, y + 186, iw, 38)
        self.scan[6].place(ix, y + 232, iw, 38)
        self.cap_flip.place(ix, y + 284, iw, 12)
        for i, c in enumerate(self.scan[8:11]):
            c.place(ix + i * 112, y + 302, 100, 32)
        self._apply_visibility()

    def _apply_visibility(self):
        for w in self.setup:
            w.visible = self.tab == "setup"
        for w in self.look:
            w.visible = self.tab == "look"
        for w in self.scan:
            w.visible = self.tab == "scanner"
        file_mode = self.src_mode == "file"
        self.btn_file.visible = self.lbl_file.visible = self.tab == "setup" and file_mode
        self.dd_dev.visible = self.tab == "setup" and not file_mode
        self.tg_play.visible = self.tab == "setup" and file_mode
        self.tg_arm.sub = ("Beam masked, aimed safely, e-stop in reach" if self.out_mode == "helios"
                           else "Only needed for Helios output")
        self._fps()
        self.ui.redraw_all()

    def _tab_changed(self):
        self._apply_visibility()

    def _mode_changed(self):
        self._apply_visibility()

    def _fps(self):
        s = self.settings
        n = max(80, min(s.points, s.pps // max(s.min_fps, 1)))
        fps = s.pps / n
        self.fps_lbl.text = f"{fps:.0f} fps refresh" + ("   (frame capped)" if n < s.points else "")
        self.fps_lbl.color = C["ok"] if fps >= 60 else C["warn"]
        if self.fps_lbl.visible:
            self.fps_lbl.redraw()

    # --------------------------------------------------------------- actions
    @staticmethod
    def _input_devices():
        if audio.sd is None:
            return []
        try:
            return [(i, d["name"]) for i, d in enumerate(audio.sd.query_devices()) if d["max_input_channels"] > 0]
        except Exception:  # noqa: BLE001
            return []

    def _open_file(self):
        p = filedialog.askopenfilename(title="Choose audio",
                                       filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")])
        if not p:
            return
        try:
            self.status.set("Loading…", C["warn"])
            self.root.update_idletasks()
            self.samples = audio.load_audio(p, 44100)
            self.file_name = os.path.basename(p)
            self.src_mode = "file"
            self.lbl_file.text = f"{self.file_name}  ·  {len(self.samples) / 44100:.0f}s"
            self.lbl_file.color = C["text"]
            self.status.set("Ready", C["ok"])
            self._apply_visibility()
        except Exception as e:  # noqa: BLE001
            self.samples = None
            self.status.set("Idle", C["dim"])
            messagebox.showerror("Could not load audio", f"{e}\n\nWAV files always work. Other formats need ffmpeg installed.")

    def _toggle_blackout(self):
        self.settings.blackout = not self.settings.blackout
        self.btn_black.set("Blackout active  ·  click to resume" if self.settings.blackout else "Blackout  ·  Space",
                           "danger" if self.settings.blackout else "outline_danger")

    def _start_stop(self):
        if self.session.running:
            self.session.stop()
            self._set_idle()
            return
        kind = self.out_mode
        if kind == "helios" and not self.armed:
            messagebox.showwarning("Not armed", "Turn on Armed (Setup tab) to confirm the laser is masked and safe "
                                   "before sending output to the Helios.")
            return
        try:
            if self.src_mode == "file":
                if self.samples is None:
                    messagebox.showinfo("No file", "Choose an audio file first, or switch to Live input.")
                    return
                src = audio.FilePlayer(self.samples, 44100, play=self.play)
            else:
                if self.dev_idx is None:
                    messagebox.showerror("No input", "No audio input devices were found.")
                    return
                src = audio.LiveInput(self.dev_idx, 44100)
            self.session.start(src, kind)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not start", str(e))
            return
        self._last_snap = None
        self.btn_start.set("Stop", "danger")
        self.status.set("Live · Helios" if kind == "helios" else "Running · preview", C["ok"] if kind != "helios" else C["danger"])

    def _set_idle(self):
        self._last_snap = None
        self.btn_start.set("Start", "success")
        self.status.set("Idle", C["dim"])
        self.section.set("IDLE", C["dim"])
        self.bpm.set("---")
        self.preview.update(None, "", False)

    def _close(self):
        try:
            self.session.stop()
        finally:
            self.root.destroy()

    # ------------------------------------------------------------------ tick
    def _tick(self):
        ses, snap = self.session, self.session.snapshot
        if ses.finished and self.btn_start.text == "Stop":
            self._set_idle()
            if ses.error:
                self.status.set("Error", C["danger"])
                messagebox.showerror("Output stopped", ses.error)
        if snap is not None and ses.running and snap is not self._last_snap:
            self._last_snap = snap
            f, s = snap.feats, self.settings
            n = max(1, len(snap.frame))
            hdr = f"{snap.scene or '–'}   ·   {n} pts   ·   {s.pps / n:.0f} fps"
            if snap.tripped:
                hdr += "   ·   static-beam guard"
            self.preview.update(snap.frame, hdr, s.blackout)

            now = time.perf_counter()
            if now - self._last_hud_t >= 0.05:
                self._last_hud_t = now
                self.bpm.set(f"{f.bpm:.1f}" if f.locked else "…")
                self.section.set(f.section, SECTION_COLORS.get(f.section, C["dim"]))
                self.beats.set(f.bar_pos if f.locked else -1, f.beat_pulse)
                for name, m in self.meters.items():
                    m.set(getattr(f, name))
                self.timeline.set(snap.log, f.time)
        self.root.after(8, self._tick)


def main():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass
    root = tk.Tk()
    try:
        k = min(max(root.winfo_fpixels("1i") / 96.0, 1.0), 2.5)
    except Exception:  # noqa: BLE001
        k = 1.0
    root.geometry(f"{int(1100 * k)}x{int(700 * k)}")
    App(root, k)
    root.mainloop()


if __name__ == "__main__":
    main()