"""LaserFX desktop GUI: QuickShow-inspired live busking cue grid, Color-Theory Harmony selector,
Projection Zones, Scanner & Galvo calibration, and Art-Net DMX512 RGB Override.
"""
from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from . import audio
from .audio import BREAKDOWN, BUILD, DROP, MAIN, SILENCE
from .fx import HARMONY_MODES, QUICK_CUES, SCENES, TEST_PATTERNS
from .session import Session, Settings
from .ui import (C, UI, W, BeatDots, Button, Card, Chip, Dropdown, Label, Meter, Pill, Preview, Segmented, Slider,
                 Stepper, Timeline, Toggle)

SECTION_COLORS = {MAIN: "#3b82f6", BREAKDOWN: "#8b5cf6", BUILD: "#f59e0b", DROP: "#ef4444", SILENCE: "#475569"}
RW = 408
M = 18


class Bar(W):
    interactive = False

    def draw(self):
        d = self.ui.d
        d.rect(self.x1, self.y1, self.x2, self.y2, C["bg"], "", 0, self.tag)
        d.line([self.x1, self.y2, self.x2, self.y2], C["line"], 1, self.tag)
        x, y = self.x1 + M, self.y1 + 14
        d.rrect(x, y, x + 32, y + 32, 9, C["accent"], C["accent"], 1, self.tag)
        for dx in (-9, 0, 9):
            d.line([x + 16, y + 25, x + 16 + dx, y + 7], C["white"], 2, self.tag)
        d.text(x + 44, y + 1, "LaserFX", C["text"], self.ui.f["title"], "nw", self.tag)
        d.text(x + 44, y + 21, "pro aerial laser controller", C["muted"], self.ui.f["small"], "nw", self.tag)


class App:
    def __init__(self, root, k=1.0):
        self.root = root
        root.title("LaserFX")
        root.configure(bg=C["bg"])
        root.minsize(1080, 780)
        self.canvas = tk.Canvas(root, bg=C["bg"], highlightthickness=0, bd=0, width=1140, height=820)
        self.canvas.pack(fill="both", expand=True)
        self.ui = UI(root, self.canvas, k)
        self.settings = Settings()
        self.session = Session(self.settings)
        self.samples, self.file_name = None, ""
        self.src_mode, self.out_mode, self.armed, self.tab = "file", "preview", False, "live"
        self.devs = self._input_devices()
        self.dev_idx = self.devs[0][0] if self.devs else None
        self.play = True
        self._last_snap = None
        self._last_hud_t = 0.0
        self._build()
        self._layout(1140, 820)
        self.canvas.bind("<Configure>", lambda e: self._layout(e.width / k, e.height / k))
        root.bind("<space>", lambda e: self._toggle_blackout())
        root.bind("<Escape>", lambda e: self._on_escape())
        root.bind("<t>", lambda e: self._tap_tempo())
        root.bind("<T>", lambda e: self._tap_tempo())
        for idx in range(len(QUICK_CUES)):
            root.bind(str(idx + 1), lambda e, i=idx: self._trigger_cue(QUICK_CUES[i][0]))
        root.bind("0", lambda e: self._trigger_cue(None))
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()

    def _on_escape(self):
        if self.ui.active_dropdown is not None:
            self.ui.active_dropdown.close()
        else:
            self._toggle_blackout()

    def _tap_tempo(self):
        self.session.tap_tempo()
        mbpm = self.session.manual_bpm
        if mbpm is not None:
            self.btn_tap.set(f"Tap · {mbpm:.0f}", "primary")

    def _reset_bpm(self):
        self.session.clear_manual_bpm()
        self.btn_tap.set("Tap BPM · T", "secondary")

    def _trigger_cue(self, scene_key: str | None):
        self.settings.test_pattern = "off"
        self.settings.scene = scene_key
        self._sync_cue_chips()
        self.ui.redraw_all()

    def _sync_cue_chips(self):
        if hasattr(self, "cue_auto"):
            self.cue_auto.redraw()
            for c in self.cue_chips:
                c.redraw()

    def _toggle_artnet(self, enabled: bool):
        self.settings.artnet_enabled = enabled
        self.session.sync_artnet_config()
        self._update_artnet_status()

    def _update_artnet_status(self):
        if not hasattr(self, "artnet_pill"):
            return
        if not self.settings.artnet_enabled:
            self.artnet_pill.set("Art-Net Off", C["dim"])
            self.artnet_info.set("UDP 6454 idle · using internal harmony engine", C["dim"])
            return
        if self.session.artnet.bind_error:
            self.artnet_pill.set("Port Error", C["danger"])
            self.artnet_info.set(self.session.artnet.bind_error, C["danger"])
            return
        st = self.session.artnet.get_active_override()
        uni_str = f"Net {self.settings.artnet_net} · Sub {self.settings.artnet_subnet} · Uni {self.settings.artnet_universe} · Ch {self.settings.artnet_channel}"
        if st is None:
            self.artnet_pill.set("Listening · 6454", C["warn"])
            self.artnet_info.set(f"Waiting for DMX ({uni_str})", C["muted"])
        else:
            r, g, b = int(st.rgb1[0] * 255), int(st.rgb1[1] * 255), int(st.rgb1[2] * 255)
            hex_col = f"#{r:02x}{g:02x}{b:02x}" if (r + g + b) > 20 else C["ok"]
            self.artnet_pill.set(f"RX · {st.source_ip}", hex_col)
            self.artnet_info.set(
                f"Dim {int(st.dimmer * 100)}% · RGB({r},{g},{b}) · {uni_str}",
                C["text"],
            )

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
        self.btn_tap = Button(u, "Tap BPM · T", self._tap_tempo, "secondary", "small")
        self.btn_auto_bpm = Button(u, "Auto", self._reset_bpm, "secondary", "small")
        self.beats = BeatDots(u)
        self.meters = {n: Meter(u, n, c) for n, c in (("bass", C["accent"]), ("mid", C["accent2"]), ("high", "#f472b6"))}
        self.tl_cap = Label(u, "SECTIONS", "tiny", C["dim"])
        self.timeline = Timeline(u, SECTION_COLORS)

        # right: 4 tabs + body card + actions
        self.tabs = Segmented(
            u,
            [("live", "Live"), ("zone", "Zone"), ("scanner", "Scanner"), ("setup", "Setup")],
            lambda: self.tab,
            lambda v: setattr(self, "tab", v),
            self._tab_changed,
        )
        self.body = Card(u)
        reg = lambda lst, w: (lst.append(w), w)[1]                        # noqa: E731

        # --- TAB 1: LIVE (Quick-Cue Grid + Color Harmony + Look Controls)
        L = self.live = []
        self.cap_cues = reg(L, Label(u, "QUICK CUES  (KEYS 1–8 · 0 AUTO)", "tiny", C["dim"]))
        self.cue_auto = reg(L, Chip(u, "Auto Show (0)", lambda: s.scene is None, lambda _: self._trigger_cue(None)))
        self.cue_chips = [
            reg(
                L,
                Chip(
                    u,
                    f"{i + 1} · {lbl}",
                    lambda k=key: s.scene == k,
                    lambda _, k=key: self._trigger_cue(k),
                ),
            )
            for i, (key, lbl) in enumerate(QUICK_CUES)
        ]
        self.sl_bright = reg(L, Slider(u, "Master brightness", 0, 1, lambda: s.brightness, lambda v: setattr(s, "brightness", v), "{:.0%}"))
        self.sl_detail = reg(L, Slider(u, "Beam density / detail", 0.2, 1, lambda: s.detail, lambda v: setattr(s, "detail", v), "{:.0%}"))
        self.dd_harmony = reg(L, Dropdown(u, "Split color", HARMONY_MODES, lambda: s.harmony_mode,
                                          lambda v: setattr(s, "harmony_mode", v)))
        scenes = [("auto", "Auto (Full Library)")] + [(n, n.replace("_", " ").title()) for n in sorted(SCENES)]
        self.dd_scene = reg(L, Dropdown(u, "Scene lock", scenes, lambda: s.scene or "auto",
                                        lambda v: (setattr(s, "scene", None if v == "auto" else v), self._sync_cue_chips())))
        secs = [("auto", "Auto"), (MAIN, "Main"), (BREAKDOWN, "Breakdown"), (BUILD, "Build"), (DROP, "Drop")]
        self.dd_sec = reg(L, Dropdown(u, "Section lock", secs, lambda: s.section or "auto",
                                      lambda v: setattr(s, "section", None if v == "auto" else v)))
        self.tg_strobe = reg(L, Toggle(u, "Strobe effects", lambda: s.strobe, lambda v: setattr(s, "strobe", v),
                                       "Turn off for photosensitive audiences"))

        # --- TAB 2: ZONE
        Z = self.zone = []
        self.cap_zone = reg(Z, Label(u, "PROJECTION ZONE GEOMETRY", "tiny", C["dim"]))
        self.sl_size = reg(Z, Slider(u, "Master scale", 0.1, 1.0, lambda: s.size, lambda v: setattr(s, "size", v), "{:.0%}"))
        self.sl_sx = reg(Z, Slider(u, "Width (X scale)", 0.2, 1.0, lambda: s.scale_x, lambda v: setattr(s, "scale_x", v), "{:.0%}"))
        self.sl_sy = reg(Z, Slider(u, "Height (Y scale)", 0.2, 1.0, lambda: s.scale_y, lambda v: setattr(s, "scale_y", v), "{:.0%}"))
        self.sl_ox = reg(Z, Slider(u, "Horizontal pan (X offset)", -0.6, 0.6, lambda: s.offset_x, lambda v: setattr(s, "offset_x", v), "{:+.0%}"))
        self.sl_oy = reg(Z, Slider(u, "Vertical elevation (Y offset)", -0.6, 0.6, lambda: s.offset_y, lambda v: setattr(s, "offset_y", v), "{:+.0%}"))
        self.sl_ks = reg(Z, Slider(u, "Vertical keystone", -0.4, 0.4, lambda: s.keystone, lambda v: setattr(s, "keystone", v), "{:+.0%}"))
        self.btn_zone_reset = reg(Z, Button(u, "Reset Zone to Full Frame", self._reset_zone, "secondary", "small"))

        # --- TAB 3: SCANNER
        S = self.scan = []
        self.st_pps = reg(S, Stepper(u, "Points per second", lambda: s.pps, lambda v: (setattr(s, "pps", int(v)), self._fps()),
                                     1000, 5000, 60000, "{:,}"))
        self.st_pts = reg(S, Stepper(u, "Points per frame", lambda: s.points, lambda v: (setattr(s, "points", int(v)), self._fps()),
                                     50, 100, 3000, "{:,}"))
        self.fps_lbl = reg(S, Label(u, "", "mid", C["ok"], "w"))
        self.cap_galvo = reg(S, Label(u, "GALVO & CORNER SHARPNESS", "tiny", C["dim"]))
        self.st_lag = reg(S, Stepper(u, "Color shift (galvo lag)", lambda: s.galvo_lag, lambda v: setattr(s, "galvo_lag", int(v)),
                                     1, -12, 12, "{:+d} pts"))
        self.st_dwell = reg(S, Stepper(u, "Corner dwell", lambda: s.corner_dwell, lambda v: setattr(s, "corner_dwell", int(v)),
                                       1, 0, 12, "{:d} pts"))
        self.st_blank = reg(S, Stepper(u, "Blanking delay", lambda: s.blank_delay, lambda v: setattr(s, "blank_delay", int(v)),
                                       1, 0, 12, "{:d} pts"))
        self.sl_brake = reg(S, Slider(u, "Corner deceleration", 0.0, 1.0, lambda: s.corner_brake,
                                      lambda v: setattr(s, "corner_brake", v), "{:.0%}"))
        patterns = [("off", "Off (Live Show)")] + [(k, v) for k, v in TEST_PATTERNS.items()]
        self.dd_test = reg(S, Dropdown(u, "Test pattern", patterns, lambda: s.test_pattern,
                                       lambda v: setattr(s, "test_pattern", v)))
        self.cap_flip = reg(S, Label(u, "ORIENTATION", "tiny", C["dim"]))
        self.chips_flip = [reg(S, Chip(u, t, lambda n=n: getattr(s, n), lambda v, n=n: setattr(s, n, v)))
                           for t, n in (("Flip X", "flip_x"), ("Flip Y", "flip_y"), ("Swap X/Y", "swap_xy"))]

        # --- TAB 4: SETUP (Audio Source + Art-Net DMX RGB Override + Laser Output)
        T = self.setup = []
        self.cap_src = reg(T, Label(u, "AUDIO SOURCE", "tiny", C["dim"]))
        self.seg_src = reg(T, Segmented(u, [("file", "Audio file"), ("live", "Live input")], lambda: self.src_mode,
                                        lambda v: setattr(self, "src_mode", v), self._mode_changed))
        self.btn_file = reg(T, Button(u, "Choose file…", self._open_file))
        self.lbl_file = reg(T, Label(u, "No file selected", "body", C["muted"], "w"))
        self.dd_dev = reg(T, Dropdown(u, "Input device", [(i, n) for i, n in self.devs] or [(None, "None found")],
                                      lambda: self.dev_idx, lambda v: setattr(self, "dev_idx", v)))
        self.tg_play = reg(T, Toggle(u, "Play through speakers", lambda: self.play, lambda v: setattr(self, "play", v)))

        self.cap_artnet = reg(T, Label(u, "ART-NET DMX512 RGB OVERRIDE (UDP 6454)", "tiny", C["dim"]))
        self.tg_artnet = reg(T, Toggle(u, "Enable Art-Net RGB", lambda: s.artnet_enabled, self._toggle_artnet))
        self.artnet_pill = reg(T, Pill(u, "Art-Net Off", C["dim"]))
        self.st_uni = reg(T, Stepper(u, "Universe (0–15)", lambda: s.artnet_universe,
                                     lambda v: (setattr(s, "artnet_universe", int(v)), self.session.sync_artnet_config()),
                                     1, 0, 15, "{}"))
        self.st_ch = reg(T, Stepper(u, "DMX Start Ch", lambda: s.artnet_channel,
                                    lambda v: (setattr(s, "artnet_channel", int(v)), self.session.sync_artnet_config()),
                                    1, 1, 507, "Ch {}"))
        self.seg_dmx_mode = reg(
            T,
            Segmented(
                u,
                [(False, "Dim+RGB+Harm"), (True, "Dual RGB1+RGB2")],
                lambda: s.artnet_dual_rgb,
                lambda v: (setattr(s, "artnet_dual_rgb", v), self.session.sync_artnet_config()),
            ),
        )
        self.artnet_info = reg(T, Label(u, "UDP 6454 idle · using internal harmony engine", "small", C["dim"], "w"))

        self.cap_out = reg(T, Label(u, "LASER OUTPUT", "tiny", C["dim"]))
        self.seg_out = reg(T, Segmented(u, [("preview", "Preview only"), ("helios", "Helios USB")], lambda: self.out_mode,
                                        lambda v: setattr(self, "out_mode", v), self._mode_changed))
        self.tg_arm = reg(T, Toggle(u, "Armed", lambda: self.armed, lambda v: setattr(self, "armed", v),
                                    "Beam masked, aimed safely, e-stop in reach"))

        self.btn_start = Button(u, "Start", self._start_stop, "success", "h")
        self.btn_black = Button(u, "Blackout  ·  Space", self._toggle_blackout, "outline_danger", "h")

    def _reset_zone(self):
        s = self.settings
        s.size = 0.70
        s.scale_x = 1.0
        s.scale_y = 1.0
        s.offset_x = 0.0
        s.offset_y = 0.0
        s.keystone = 0.0
        self.ui.redraw_all()

    # ---------------------------------------------------------------- layout
    def _layout(self, W_, H_):
        u = self.ui
        u.logical_h = H_
        self.bar.place(0, 0, W_, 60)
        self.status.place(W_ - M - 190, 15, 190, 30)
        top = 60 + M
        lw = W_ - RW - 3 * M
        now_h = 204
        self.preview.place(M, top, lw, H_ - top - M - now_h - 14)
        ny = top + self.preview.h + 14
        self.now_card.place(M, ny, lw, now_h)
        self.bpm_cap.place(M + 20, ny + 36, 60, 14)
        self.bpm.place(M + 20, ny + 48, 130, 52)
        self.section.place(M + 154, ny + 52, 136, 30)
        self.btn_tap.place(M + 298, ny + 52, 92, 30)
        self.btn_auto_bpm.place(M + 396, ny + 52, 48, 30)
        self.beats.place(M + lw - 20 - 150, ny + 52, 150, 30)
        mw = (lw - 40 - 2 * 18) / 3
        for i, m in enumerate(self.meters.values()):
            m.place(M + 20 + i * (mw + 18), ny + 112, mw, 30)
        self.tl_cap.place(M + 20, ny + 154, 90, 12)
        self.timeline.place(M + 20, ny + 170, lw - 40, 16)

        rx = W_ - M - RW
        self.tabs.place(rx, top, RW, 38)
        by = top + 48
        bh = H_ - by - M - 2 * 46 - 2 * 10
        self.body.place(rx, by, RW, bh)
        ix, iw = rx + 18, RW - 36
        self.btn_start.place(rx, H_ - M - 46 - 10 - 46, RW, 46)
        self.btn_black.place(rx, H_ - M - 46, RW, 46)

        # --- LIVE TAB
        y = by + 12
        self.cap_cues.place(ix, y, iw - 110, 12)
        self.cue_auto.place(ix + iw - 104, y - 5, 104, 22)
        cw = (iw - 8) / 2
        for i, chip in enumerate(self.cue_chips):
            col_i = i % 2
            row_i = i // 2
            chip.place(ix + col_i * (cw + 8), y + 20 + row_i * 30, cw, 25)
        ly = y + 148
        self.sl_bright.place(ix, ly, iw, 36)
        self.sl_detail.place(ix, ly + 40, iw, 36)
        self.dd_harmony.place(ix, ly + 82, iw, 34)
        self.dd_scene.place(ix, ly + 122, iw, 34)
        self.dd_sec.place(ix, ly + 162, iw, 34)
        self.tg_strobe.place(ix, ly + 202, iw, 38)

        # --- ZONE TAB
        y = by + 14
        self.cap_zone.place(ix, y, iw, 12)
        for i, sl in enumerate((self.sl_size, self.sl_sx, self.sl_sy, self.sl_ox, self.sl_oy, self.sl_ks)):
            sl.place(ix, y + 20 + i * 44, iw, 38)
        self.btn_zone_reset.place(ix, y + 292, iw, 34)

        # --- SCANNER TAB
        y = by + 12
        self.st_pps.place(ix, y, iw, 30)
        self.st_pts.place(ix, y + 34, iw, 30)
        self.fps_lbl.place(ix, y + 68, iw, 18)
        self.cap_galvo.place(ix, y + 92, iw, 12)
        self.st_lag.place(ix, y + 108, iw, 30)
        self.st_dwell.place(ix, y + 142, iw, 30)
        self.st_blank.place(ix, y + 176, iw, 30)
        self.sl_brake.place(ix, y + 210, iw, 38)
        self.dd_test.place(ix, y + 254, iw, 34)
        self.cap_flip.place(ix, y + 298, iw, 12)
        fw = (iw - 16) / 3
        for i, c in enumerate(self.chips_flip):
            c.place(ix + i * (fw + 8), y + 314, fw, 28)

        # --- SETUP TAB
        y = by + 12
        self.cap_src.place(ix, y, iw, 12)
        self.seg_src.place(ix, y + 16, iw, 32)
        self.btn_file.place(ix, y + 54, 130, 32)
        self.lbl_file.place(ix + 140, y + 54, iw - 140, 32)
        self.dd_dev.place(ix, y + 54, iw, 32)
        self.tg_play.place(ix, y + 90, iw, 28)

        ay = y + 128
        self.cap_artnet.place(ix, ay, iw, 12)
        self.tg_artnet.place(ix, ay + 16, iw - 154, 28)
        self.artnet_pill.place(ix + iw - 148, ay + 16, 148, 28)
        self.st_uni.place(ix, ay + 48, iw, 28)
        self.st_ch.place(ix, ay + 80, iw, 28)
        self.seg_dmx_mode.place(ix, ay + 112, iw, 28)
        self.artnet_info.place(ix, ay + 144, iw, 18)

        oy = ay + 172
        self.cap_out.place(ix, oy, iw, 12)
        self.seg_out.place(ix, oy + 16, iw, 32)
        self.tg_arm.place(ix, oy + 54, iw, 42)
        self._apply_visibility()

    def _apply_visibility(self):
        if self.ui.active_dropdown is not None:
            self.ui.active_dropdown.close()
        for w in self.live:
            w.visible = self.tab == "live"
        for w in self.zone:
            w.visible = self.tab == "zone"
        for w in self.scan:
            w.visible = self.tab == "scanner"
        for w in self.setup:
            w.visible = self.tab == "setup"
        file_mode = self.src_mode == "file"
        self.btn_file.visible = self.lbl_file.visible = self.tab == "setup" and file_mode
        self.dd_dev.visible = self.tab == "setup" and not file_mode
        self.tg_play.visible = self.tab == "setup" and file_mode
        self.tg_arm.sub = ("Beam masked, aimed safely, e-stop in reach" if self.out_mode == "helios"
                           else "Only needed for Helios output")
        self._fps()
        self._update_artnet_status()
        self.ui.redraw_all()

    def _tab_changed(self):
        self._apply_visibility()

    def _mode_changed(self):
        self._apply_visibility()

    def _fps(self):
        s = self.settings
        n = max(90, min(s.points, s.pps // max(s.min_fps, 1)))
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
            if self.ui.active_dropdown is not None:
                self.ui.active_dropdown.close()
            self.session.stop()
            self.session.artnet.stop()
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
                self._update_artnet_status()
        elif self.settings.artnet_enabled:
            now = time.perf_counter()
            if now - self._last_hud_t >= 0.15:
                self._last_hud_t = now
                self._update_artnet_status()
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
    root.geometry(f"{int(1140 * k)}x{int(820 * k)}")
    App(root, k)
    root.mainloop()


if __name__ == "__main__":
    main()