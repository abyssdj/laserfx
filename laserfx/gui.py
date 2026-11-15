"""LaserFX desktop GUI (tkinter: ships with Python, so the packaged app needs nothing else)."""
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from . import audio
from .audio import BREAKDOWN, BUILD, DROP, MAIN, SILENCE
from .fx import SCENES
from .session import Session, Settings

SECTION_COLORS = {MAIN: "#2d7ff9", BREAKDOWN: "#6a5acd", BUILD: "#e0a800", DROP: "#e0245e", SILENCE: "#555555"}
BG, FG, PANEL = "#15161a", "#e8e8ee", "#1f2026"
CANVAS = 520


class App:
    def __init__(self, root):
        self.root = root
        root.title("LaserFX - sound to light")
        root.configure(bg=BG)
        root.minsize(980, 640)
        self.settings = Settings()
        self.session = Session(self.settings)
        self.samples, self.file_name = None, ""
        self._style()
        self._build()
        root.bind("<space>", lambda e: self._toggle_blackout() if not isinstance(root.focus_get(), (tk.Entry, ttk.Entry, ttk.Combobox, ttk.Spinbox)) else None)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()

    # ------------------------------------------------------------------ layout
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground=FG, fieldbackground=PANEL)
        st.configure("TLabelframe", background=BG, foreground=FG)
        st.configure("TLabelframe.Label", background=BG, foreground="#9aa0b4")
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.configure("TRadiobutton", background=BG, foreground=FG)
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Horizontal.TProgressbar", troughcolor=PANEL, background="#2d7ff9")

    def _build(self):
        r = self.root
        left = tk.Frame(r, bg=BG)
        left.pack(side="left", padx=10, pady=10)
        self.canvas = tk.Canvas(left, width=CANVAS, height=CANVAS, bg="black", highlightthickness=1,
                                highlightbackground="#333")
        self.canvas.pack()
        row = tk.Frame(left, bg=BG)
        row.pack(fill="x", pady=(8, 0))
        self.bpm_var = tk.StringVar(value="--- BPM")
        tk.Label(row, textvariable=self.bpm_var, font=("Helvetica", 26, "bold"), bg=BG, fg=FG).pack(side="left")
        self.sec_lbl = tk.Label(row, text="IDLE", font=("Helvetica", 16, "bold"), width=11, bg="#444", fg="white")
        self.sec_lbl.pack(side="right")
        beats = tk.Frame(left, bg=BG)
        beats.pack(fill="x", pady=4)
        self.beat_dots = [tk.Label(beats, text=str(i + 1), width=4, bg="#333", fg="#888", font=("Helvetica", 12, "bold"))
                          for i in range(4)]
        for d in self.beat_dots:
            d.pack(side="left", padx=2)
        self.info_var = tk.StringVar(value="")
        tk.Label(beats, textvariable=self.info_var, bg=BG, fg="#9aa0b4").pack(side="right")
        bars = tk.Frame(left, bg=BG)
        bars.pack(fill="x")
        self.meters = {}
        for n in ("bass", "mid", "high"):
            tk.Label(bars, text=n, bg=BG, fg="#9aa0b4", width=5).pack(side="left")
            pb = ttk.Progressbar(bars, length=110, maximum=1.0)
            pb.pack(side="left", padx=(0, 8))
            self.meters[n] = pb

        right = tk.Frame(r, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        # source
        f = ttk.LabelFrame(right, text="Audio source")
        f.pack(fill="x", pady=4)
        self.src_mode = tk.StringVar(value="file")
        ttk.Radiobutton(f, text="File", variable=self.src_mode, value="file").grid(row=0, column=0, sticky="w")
        ttk.Button(f, text="Open audio file...", command=self._open_file).grid(row=0, column=1, sticky="w")
        self.file_lbl = ttk.Label(f, text="(none)", width=26)
        self.file_lbl.grid(row=0, column=2, sticky="w", padx=6)
        ttk.Radiobutton(f, text="Live input", variable=self.src_mode, value="live").grid(row=1, column=0, sticky="w")
        self.devs = self._input_devices()
        self.dev_box = ttk.Combobox(f, values=[d[1] for d in self.devs], state="readonly", width=38)
        if self.devs:
            self.dev_box.current(0)
        self.dev_box.grid(row=1, column=1, columnspan=2, sticky="w", pady=2)
        self.play_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Play file through speakers", variable=self.play_var).grid(row=2, column=1, columnspan=2, sticky="w")
        if audio.sd is None:
            ttk.Label(f, text="No audio backend: live input and playback unavailable", foreground="#e0a800").grid(
                row=3, column=0, columnspan=3, sticky="w")

        # output
        f = ttk.LabelFrame(right, text="Laser output")
        f.pack(fill="x", pady=4)
        self.out_mode = tk.StringVar(value="preview")
        ttk.Radiobutton(f, text="Preview only (no laser)", variable=self.out_mode, value="preview").pack(anchor="w")
        ttk.Radiobutton(f, text="Helios DAC (USB)", variable=self.out_mode, value="helios").pack(anchor="w")
        self.arm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="ARMED: beam is physically masked, aimed away from people, e-stop within reach",
                        variable=self.arm_var, command=self._arm_changed).pack(anchor="w", pady=(4, 0))
        self.start_btn = tk.Button(f, text="START", font=("Helvetica", 14, "bold"), bg="#1e8e3e", fg="white",
                                   command=self._start_stop, height=1, takefocus=0)
        self.start_btn.pack(fill="x", pady=6)

        # look
        f = ttk.LabelFrame(right, text="Look")
        f.pack(fill="x", pady=4)
        self.bright = self._slider(f, "Brightness", 0, 1, self.settings.brightness, 0, lambda v: setattr(self.settings, "brightness", v))
        self.size = self._slider(f, "Size", 0.1, 1, self.settings.size, 1, lambda v: setattr(self.settings, "size", v))
        self.detail = self._slider(f, "Detail", 0.2, 1, self.settings.detail, 7, lambda v: setattr(self.settings, "detail", v))
        ttk.Label(f, text="Points/sec").grid(row=2, column=0, sticky="w")
        self.pps_sp = ttk.Spinbox(f, from_=5000, to=60000, increment=1000, width=8, command=self._apply_numbers)
        self.pps_sp.set(self.settings.pps)
        self.pps_sp.grid(row=2, column=1, sticky="w")
        ttk.Label(f, text="Points/frame").grid(row=3, column=0, sticky="w")
        self.pts_sp = ttk.Spinbox(f, from_=200, to=3000, increment=100, width=8, command=self._apply_numbers)
        self.pts_sp.set(self.settings.points)
        self.pts_sp.grid(row=3, column=1, sticky="w")
        for sp in (self.pps_sp, self.pts_sp):
            sp.bind("<FocusOut>", lambda e: self._apply_numbers())
            sp.bind("<Return>", lambda e: self._apply_numbers())
        self.flips = {}
        for i, n in enumerate(("flip_x", "flip_y", "swap_xy")):
            v = tk.BooleanVar(value=False)
            ttk.Checkbutton(f, text=n.replace("_", " ").upper().replace("FLIP ", "Flip "), variable=v,
                            command=lambda n=n, v=v: setattr(self.settings, n, v.get())).grid(row=4, column=i, sticky="w")
            self.flips[n] = v
        ttk.Label(f, text="Scene").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.scene_box = ttk.Combobox(f, values=["Auto"] + sorted(SCENES), state="readonly", width=16)
        self.scene_box.current(0)
        self.scene_box.grid(row=5, column=1, columnspan=2, sticky="w", pady=(6, 0))
        self.scene_box.bind("<<ComboboxSelected>>", lambda e: setattr(
            self.settings, "scene", None if self.scene_box.get() == "Auto" else self.scene_box.get()))
        ttk.Label(f, text="Section").grid(row=6, column=0, sticky="w")
        self.sec_box = ttk.Combobox(f, values=["Auto", MAIN, BREAKDOWN, BUILD, DROP], state="readonly", width=16)
        self.sec_box.current(0)
        self.sec_box.grid(row=6, column=1, columnspan=2, sticky="w")
        self.sec_box.bind("<<ComboboxSelected>>", lambda e: setattr(
            self.settings, "section", None if self.sec_box.get() == "Auto" else self.sec_box.get()))
        self.strobe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Strobe effects (flashing: disable for photosensitive audiences)", variable=self.strobe_var,
                        command=lambda: setattr(self.settings, "strobe", self.strobe_var.get())).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(4, 0))
        f.columnconfigure(2, weight=1)

        self.black_btn = tk.Button(right, text="BLACKOUT  (space)", font=("Helvetica", 14, "bold"), bg="#8a1c1c", fg="white",
                                   command=self._toggle_blackout, height=2, takefocus=0)
        self.black_btn.pack(fill="x", pady=6)

        f = ttk.LabelFrame(right, text="Detected sections")
        f.pack(fill="both", expand=True, pady=4)
        self.log_box = tk.Listbox(f, height=6, bg=PANEL, fg=FG, highlightthickness=0, borderwidth=0)
        self.log_box.pack(fill="both", expand=True)
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(right, textvariable=self.status_var, bg=BG, fg="#9aa0b4", anchor="w", justify="left", wraplength=420).pack(fill="x")

    def _slider(self, parent, label, lo, hi, val, row, cb):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        var = tk.DoubleVar(value=val)
        s = ttk.Scale(parent, from_=lo, to=hi, variable=var, command=lambda v: cb(float(v)))
        s.grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        return var

    # ---------------------------------------------------------------- actions
    @staticmethod
    def _input_devices():
        if audio.sd is None:
            return []
        try:
            return [(i, f"{i}: {d['name']}") for i, d in enumerate(audio.sd.query_devices()) if d["max_input_channels"] > 0]
        except Exception:  # noqa: BLE001
            return []

    def _open_file(self):
        p = filedialog.askopenfilename(title="Choose audio", filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All", "*.*")])
        if not p:
            return
        try:
            self.status_var.set("Loading...")
            self.root.update_idletasks()
            self.samples = audio.load_audio(p, 44100)
            self.file_name = os.path.basename(p)
            self.file_lbl.config(text=f"{self.file_name} ({len(self.samples) / 44100:.0f}s)")
            self.src_mode.set("file")
            self.status_var.set("Loaded. Press START.")
        except Exception as e:  # noqa: BLE001
            self.samples = None
            messagebox.showerror("Could not load audio", f"{e}\n\nWAV files always work. Other formats need ffmpeg installed.")
            self.status_var.set("Ready.")

    def _arm_changed(self):
        pass

    def _apply_numbers(self):
        try:
            self.settings.pps = int(min(max(int(float(self.pps_sp.get())), 5000), 60000))
            self.settings.points = int(min(max(int(float(self.pts_sp.get())), 200), 3000))
        except ValueError:
            pass

    def _toggle_blackout(self):
        self.settings.blackout = not self.settings.blackout
        self.black_btn.config(bg="#ff2a2a" if self.settings.blackout else "#8a1c1c",
                              text="BLACKED OUT - click to resume" if self.settings.blackout else "BLACKOUT  (space)")

    def _start_stop(self):
        if self.session.running:
            self.session.stop()
            self._set_idle()
            return
        self._apply_numbers()
        kind = self.out_mode.get()
        if kind == "helios" and not self.arm_var.get():
            messagebox.showwarning("Not armed", "Tick the ARMED box to confirm the laser is masked and safe before outputting to the Helios.")
            return
        try:
            if self.src_mode.get() == "file":
                if self.samples is None:
                    messagebox.showinfo("No file", "Open an audio file first (or choose Live input).")
                    return
                src = audio.FilePlayer(self.samples, 44100, play=self.play_var.get())
            else:
                if not self.devs:
                    messagebox.showerror("No input", "No audio input devices were found.")
                    return
                src = audio.LiveInput(self.devs[self.dev_box.current()][0], 44100)
            self.session.start(src, kind)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not start", str(e))
            return
        self.start_btn.config(text="STOP", bg="#b3261e")
        self.status_var.set("Running" + (" - output to Helios" if kind == "helios" else " - preview only"))

    def _set_idle(self):
        self.start_btn.config(text="START", bg="#1e8e3e")
        self.canvas.delete("all")
        self.sec_lbl.config(text="IDLE", bg="#444")

    def _close(self):
        try:
            self.session.stop()
        finally:
            self.root.destroy()

    # ------------------------------------------------------------------- tick
    def _tick(self):
        ses = self.session
        snap = ses.snapshot
        if ses.finished and self.start_btn.cget("text") == "STOP":
            self._set_idle()
            self.status_var.set(f"Stopped. {ses.error}" if ses.error else "Finished.")
            if ses.error:
                messagebox.showerror("Output stopped", ses.error)
        if snap is not None and ses.running:
            f = snap.feats
            self.bpm_var.set(f"{f.bpm:5.1f} BPM" if f.locked else "listening...")
            self.sec_lbl.config(text=f.section, bg=SECTION_COLORS.get(f.section, "#444"))
            for i, d in enumerate(self.beat_dots):
                on = i == f.bar_pos and f.beat_pulse > 0.25
                d.config(bg="#e8e8ee" if on else "#333", fg="black" if on else "#888")
            for n in self.meters:
                self.meters[n]["value"] = min(getattr(f, n), 1.0)
            self.info_var.set(f"scene: {snap.scene or '-'}   {len(snap.frame)} pts  {self.settings.pps / len(snap.frame):.0f} fps" + ("   STATIC GUARD" if snap.tripped else ""))
            self._draw(snap.frame)
            log = [f"{int(t // 60)}:{t % 60:04.1f}  {s}" for t, s in snap.log][-8:]
            if self.log_box.size() != len(log) or (log and self.log_box.get(tk.END) != log[-1]):
                self.log_box.delete(0, tk.END)
                for x in log:
                    self.log_box.insert(tk.END, x)
        self.root.after(33, self._tick)

    def _draw(self, fr):
        c, W = self.canvas, CANVAS
        c.delete("all")
        px = ((fr[:, 0] + 1) * 0.5 * W)
        py = ((1 - fr[:, 1]) * 0.5 * W)
        lit = fr[:, 2:5].max(1) > 0.01
        step = 6
        for i in range(0, len(fr) - 1, step):
            j = min(i + step, len(fr) - 1)
            if lit[i] and lit[j]:
                rgb = np.clip(fr[i, 2:5] * 2.0, 0, 1) * 255
                coords = np.empty((j - i + 1) * 2)
                coords[0::2], coords[1::2] = px[i:j + 1], py[i:j + 1]
                c.create_line(*coords.tolist(), fill="#%02x%02x%02x" % tuple(int(v) for v in rgb), width=2)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
