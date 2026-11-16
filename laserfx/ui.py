"""Custom-drawn dark UI widgets on a single tkinter Canvas (zero external dependencies)."""
from __future__ import annotations

import math
import time
import tkinter as tk
from dataclasses import dataclass

import numpy as np

C = {
    "bg": "#0b0f17",
    "card": "#131926",
    "card2": "#192234",
    "line": "#243047",
    "text": "#f1f5f9",
    "muted": "#94a3b8",
    "dim": "#64748b",
    "accent": "#3b82f6",
    "accent2": "#8b5cf6",
    "ok": "#10b981",
    "warn": "#f59e0b",
    "danger": "#ef4444",
    "white": "#ffffff",
}


def mix(c1: str, c2: str, t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    return f"#{int(r1 + (r2 - r1) * t):02x}{int(g1 + (g2 - g1) * t):02x}{int(b1 + (b2 - b1) * t):02x}"


class Draw:
    def __init__(self, canvas: tk.Canvas, k: float):
        self.cv = canvas
        self.k = k

    def s(self, v):
        return v * self.k

    def rect(self, x1, y1, x2, y2, fill="", outline="", w=1, tag=""):
        k = self.k
        return self.cv.create_rectangle(x1 * k, y1 * k, x2 * k, y2 * k, fill=fill, outline=outline, width=w * k, tags=tag)

    def rrect(self, x1, y1, x2, y2, r, fill="", outline="", w=1, tag=""):
        k = self.k
        x1, y1, x2, y2 = x1 * k, y1 * k, x2 * k, y2 * k
        r = min(r * k, max(0.0, (x2 - x1) / 2), max(0.0, (y2 - y1) / 2))
        pts = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.cv.create_polygon(pts, smooth=True, fill=fill, outline=outline, width=w * k, tags=tag)

    def line(self, pts, fill, w=1, tag="", smooth=False):
        k = self.k
        flat = [c * k for c in pts]
        return self.cv.create_line(*flat, fill=fill, width=w * k, capstyle="round", joinstyle="round", smooth=smooth, tags=tag)

    def oval(self, x1, y1, x2, y2, fill="", outline="", w=1, tag=""):
        k = self.k
        return self.cv.create_oval(x1 * k, y1 * k, x2 * k, y2 * k, fill=fill, outline=outline, width=w * k, tags=tag)

    def text(self, x, y, txt, fill, font, anchor="w", tag="", width=0):
        k = self.k
        kw = {"width": width * k} if width else {}
        return self.cv.create_text(x * k, y * k, text=txt, fill=fill, font=font, anchor=anchor, tags=tag, **kw)


class UI:
    def __init__(self, root: tk.Tk, canvas: tk.Canvas, k: float = 1.0):
        self.root = root
        self.cv = canvas
        self.k = k
        self.d = Draw(canvas, k)
        self.logical_h = 760.0
        fam = "Segoe UI" if tk.TkVersion >= 8.5 else "Helvetica"
        self.f = {
            "title": (fam, int(round(13 * k)), "bold"),
            "big": (fam, int(round(28 * k)), "bold"),
            "h": (fam, int(round(11 * k)), "bold"),
            "body": (fam, int(round(10 * k))),
            "mid": (fam, int(round(10 * k)), "bold"),
            "small": (fam, int(round(9 * k))),
            "tiny": (fam, int(round(8 * k)), "bold"),
        }
        self.widgets: list[W] = []
        self.active_dropdown: Dropdown | None = None
        self._last_popup_close: float = 0.0
        canvas.bind("<ButtonPress-1>", self._on_press, add="+")
        canvas.bind("<B1-Motion>", self._on_drag, add="+")
        canvas.bind("<ButtonRelease-1>", self._on_release, add="+")
        canvas.bind("<Motion>", self._on_hover, add="+")
        self._drag_target: W | None = None
        self._hover_target: W | None = None

    def register(self, w: W):
        self.widgets.append(w)
        return w

    def redraw_all(self):
        self.cv.delete("all")
        for w in self.widgets:
            if w.visible:
                w.draw()

    def _hit(self, x, y) -> W | None:
        lx, ly = x / self.k, y / self.k
        for w in reversed(self.widgets):
            if w.visible and w.interactive and w.contains(lx, ly):
                return w
        return None

    def _on_press(self, e):
        # If a dropdown popup is open, close it cleanly and swallow this click so it never reopens
        if self.active_dropdown is not None:
            self.active_dropdown.close()
            return
        # Swallow stray click events within 180ms of a popup dismissal
        if (time.monotonic() - self._last_popup_close) < 0.18:
            return
        w = self._hit(e.x, e.y)
        self._drag_target = w
        if w:
            w.on_press(e.x / self.k, e.y / self.k)

    def _on_drag(self, e):
        if self._drag_target:
            self._drag_target.on_drag(e.x / self.k, e.y / self.k)

    def _on_release(self, e):
        if self._drag_target:
            self._drag_target.on_release(e.x / self.k, e.y / self.k)
            self._drag_target = None

    def _on_hover(self, e):
        if self.active_dropdown is not None:
            return
        w = self._hit(e.x, e.y)
        if w is not self._hover_target:
            if self._hover_target:
                self._hover_target.hovered = False
                self._hover_target.redraw()
            self._hover_target = w
            if w:
                w.hovered = True
                w.redraw()
            self.cv.configure(cursor="hand2" if w else "")


class W:
    interactive = True

    def __init__(self, ui: UI):
        self.ui = ui
        self.x1 = self.y1 = self.x2 = self.y2 = 0.0
        self.w = self.h = 0.0
        self.visible = True
        self.hovered = False
        self.tag = f"w_{id(self)}"
        ui.register(self)

    def place(self, x, y, w, h):
        self.x1, self.y1, self.w, self.h = float(x), float(y), float(w), float(h)
        self.x2, self.y2 = self.x1 + self.w, self.y1 + self.h

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def redraw(self):
        self.ui.cv.delete(self.tag)
        if self.visible:
            self.draw()

    def draw(self):
        pass

    def on_press(self, x, y):
        pass

    def on_drag(self, x, y):
        pass

    def on_release(self, x, y):
        pass


class Card(W):
    interactive = False

    def __init__(self, ui: UI, title: str = ""):
        super().__init__(ui)
        self.title = title

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, 12, C["card"], C["line"], 1, self.tag)
        if self.title:
            d.text(self.x1 + 20, self.y1 + 18, self.title.upper(), C["dim"], self.ui.f["tiny"], "w", self.tag)


class Label(W):
    interactive = False

    def __init__(self, ui: UI, text="", font="body", color=C["text"], anchor="w"):
        super().__init__(ui)
        self.text, self.font, self.color, self.anchor = text, font, color, anchor

    def set(self, text, color=None):
        if text == self.text and (color is None or color == self.color):
            return
        self.text = text
        if color is not None:
            self.color = color
        self.redraw()

    def draw(self):
        ax = self.x1 if "w" in self.anchor else (self.x2 if "e" in self.anchor else (self.x1 + self.x2) / 2)
        ay = (self.y1 + self.y2) / 2
        self.ui.d.text(ax, ay, self.text, self.color, self.ui.f[self.font], self.anchor, self.tag, width=self.w)


class Pill(W):
    interactive = False

    def __init__(self, ui: UI, text="", color=C["dim"]):
        super().__init__(ui)
        self.text, self.color = text, color

    def set(self, text, color=None):
        c = color or self.color
        if text == self.text and c == self.color:
            return
        self.text, self.color = text, c
        self.redraw()

    def draw(self):
        d = self.ui.d
        bg = mix(C["card"], self.color, 0.22)
        bd = mix(C["line"], self.color, 0.55)
        d.rrect(self.x1, self.y1, self.x2, self.y2, self.h / 2, bg, bd, 1, self.tag)
        cy = (self.y1 + self.y2) / 2
        d.oval(self.x1 + 12, cy - 4, self.x1 + 20, cy + 4, self.color, "", 0, self.tag)
        d.text(self.x1 + 28, cy, self.text, C["text"], self.ui.f["mid"], "w", self.tag)


class Button(W):
    def __init__(self, ui: UI, text, cb, style="secondary", font="mid"):
        super().__init__(ui)
        self.text, self.cb, self.style, self.font = text, cb, style, font

    def set(self, text, style=None):
        self.text = text
        if style:
            self.style = style
        self.redraw()

    def draw(self):
        d = self.ui.d
        styles = {
            "primary": (C["accent"], C["accent"], C["white"]),
            "success": (C["ok"], C["ok"], "#04130d"),
            "danger": (C["danger"], C["danger"], C["white"]),
            "outline_danger": (mix(C["card"], C["danger"], 0.14), mix(C["line"], C["danger"], 0.55), "#fca5a5"),
            "secondary": (C["card2"], C["line"], C["text"]),
        }
        bg, bd, fg = styles.get(self.style, styles["secondary"])
        if self.hovered:
            bg = mix(bg, C["white"], 0.10)
        d.rrect(self.x1, self.y1, self.x2, self.y2, 10, bg, bd, 1, self.tag)
        d.text((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.text, fg, self.ui.f[self.font], "center", self.tag)

    def on_release(self, x, y):
        if self.contains(x, y) and self.cb:
            self.cb()


class Chip(W):
    def __init__(self, ui: UI, text, getter, setter):
        super().__init__(ui)
        self.text, self.get, self.set = text, getter, setter

    def draw(self):
        d = self.ui.d
        on = bool(self.get())
        bg = mix(C["card2"], C["accent"], 0.28) if on else C["card2"]
        bd = C["accent"] if on else (mix(C["line"], C["white"], 0.15) if self.hovered else C["line"])
        fg = C["text"] if on else C["muted"]
        d.rrect(self.x1, self.y1, self.x2, self.y2, 8, bg, bd, 1, self.tag)
        d.text((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.text, fg, self.ui.f["small"], "center", self.tag)

    def on_release(self, x, y):
        if self.contains(x, y):
            self.set(not self.get())
            self.redraw()


class Toggle(W):
    def __init__(self, ui: UI, title, getter, setter, sub=""):
        super().__init__(ui)
        self.title, self.get, self.set, self.sub = title, getter, setter, sub

    def draw(self):
        d = self.ui.d
        on = bool(self.get())
        cy = (self.y1 + self.y2) / 2
        if self.sub:
            d.text(self.x1, cy - 8, self.title, C["text"], self.ui.f["mid"], "w", self.tag)
            d.text(self.x1, cy + 9, self.sub, C["dim"], self.ui.f["small"], "w", self.tag)
        else:
            d.text(self.x1, cy, self.title, C["text"], self.ui.f["body"], "w", self.tag)
        sw, sh = 42, 24
        sx1, sy1 = self.x2 - sw, cy - sh / 2
        bg = C["accent"] if on else C["card2"]
        bd = C["accent"] if on else C["line"]
        d.rrect(sx1, sy1, self.x2, sy1 + sh, sh / 2, bg, bd, 1, self.tag)
        kx = (self.x2 - sh / 2) if on else (sx1 + sh / 2)
        d.oval(kx - 8, cy - 8, kx + 8, cy + 8, C["white"], "", 0, self.tag)

    def on_release(self, x, y):
        if self.contains(x, y):
            self.set(not self.get())
            self.redraw()


class Segmented(W):
    def __init__(self, ui: UI, items, getter, setter, on_change=None):
        super().__init__(ui)
        self.items, self.get, self.set, self.on_change = items, getter, setter, on_change

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, 10, C["card2"], C["line"], 1, self.tag)
        cur = self.get()
        n = max(1, len(self.items))
        iw = (self.w - 6) / n
        for i, (val, lbl) in enumerate(self.items):
            ax1 = self.x1 + 3 + i * iw
            ax2 = ax1 + iw
            active = val == cur
            if active:
                d.rrect(ax1, self.y1 + 3, ax2, self.y2 - 3, 8, C["accent"], "", 0, self.tag)
            d.text((ax1 + ax2) / 2, (self.y1 + self.y2) / 2, lbl,
                   C["white"] if active else C["muted"], self.ui.f["mid"], "center", self.tag)

    def on_release(self, x, y):
        if not self.contains(x, y):
            return
        n = max(1, len(self.items))
        idx = min(n - 1, max(0, int((x - self.x1) / (self.w / n))))
        val = self.items[idx][0]
        if val != self.get():
            self.set(val)
            self.redraw()
            if self.on_change:
                self.on_change()


class Slider(W):
    def __init__(self, ui: UI, title, lo, hi, getter, setter, fmt="{:.0%}"):
        super().__init__(ui)
        self.title, self.lo, self.hi, self.get, self.set, self.fmt = title, lo, hi, getter, setter, fmt

    def draw(self):
        d = self.ui.d
        v = float(self.get())
        u = min(max((v - self.lo) / max(1e-9, self.hi - self.lo), 0.0), 1.0)
        d.text(self.x1, self.y1 + 10, self.title, C["text"], self.ui.f["body"], "w", self.tag)
        d.text(self.x2, self.y1 + 10, self.fmt.format(v), C["muted"], self.ui.f["mid"], "e", self.tag)
        ty = self.y1 + 30
        d.rrect(self.x1, ty - 3, self.x2, ty + 3, 3, C["card2"], "", 0, self.tag)
        kx = self.x1 + u * self.w
        if kx > self.x1 + 2:
            d.rrect(self.x1, ty - 3, kx, ty + 3, 3, C["accent"], "", 0, self.tag)
        d.oval(kx - 7, ty - 7, kx + 7, ty + 7, C["white"], C["accent"], 2, self.tag)

    def _update(self, x):
        u = min(max((x - self.x1) / max(1.0, self.w), 0.0), 1.0)
        self.set(self.lo + u * (self.hi - self.lo))
        self.redraw()

    def on_press(self, x, y):
        self._update(x)

    def on_drag(self, x, y):
        self._update(x)


class Stepper(W):
    def __init__(self, ui: UI, title, getter, setter, step, lo, hi, fmt="{}"):
        super().__init__(ui)
        self.title, self.get, self.set, self.step, self.lo, self.hi, self.fmt = title, getter, setter, step, lo, hi, fmt

    def draw(self):
        d = self.ui.d
        v = self.get()
        cy = (self.y1 + self.y2) / 2
        d.text(self.x1, cy, self.title, C["text"], self.ui.f["body"], "w", self.tag)
        bw = 26
        vw = 78
        px2 = self.x2
        px1 = px2 - bw
        vx2 = px1 - 4
        vx1 = vx2 - vw
        mx2 = vx1 - 4
        mx1 = mx2 - bw
        by1, by2 = cy - 13, cy + 13
        d.rrect(mx1, by1, mx2, by2, 7, C["card2"], C["line"], 1, self.tag)
        d.text((mx1 + mx2) / 2, cy, "−", C["text"], self.ui.f["mid"], "center", self.tag)
        d.text((vx1 + vx2) / 2, cy, self.fmt.format(v), C["text"], self.ui.f["mid"], "center", self.tag)
        d.rrect(px1, by1, px2, by2, 7, C["card2"], C["line"], 1, self.tag)
        d.text((px1 + px2) / 2, cy, "+", C["text"], self.ui.f["mid"], "center", self.tag)

    def on_release(self, x, y):
        if not self.contains(x, y):
            return
        bw = 26
        vw = 78
        px1 = self.x2 - bw
        mx2 = px1 - 4 - vw - 4
        mx1 = mx2 - bw
        v = self.get()
        if mx1 <= x <= mx2:
            self.set(max(self.lo, v - self.step))
            self.redraw()
        elif px1 <= x <= self.x2:
            self.set(min(self.hi, v + self.step))
            self.redraw()


class Dropdown(W):
    """Canvas-drawn selector with a clean floating Toplevel list that never reopens on click."""

    def __init__(self, ui: UI, title, items, getter, setter):
        super().__init__(ui)
        self.title, self.items, self.get, self.set = title, items, getter, setter
        self._popup: tk.Toplevel | None = None

    def _current_label(self) -> str:
        cur = self.get()
        for val, lbl in self.items:
            if val == cur:
                return lbl
        return self.items[0][1] if self.items else ""

    def draw(self):
        d = self.ui.d
        cy = (self.y1 + self.y2) / 2
        lw = min(115.0, self.w * 0.36)
        d.text(self.x1, cy, self.title, C["muted"], self.ui.f["body"], "w", self.tag)
        bx1 = self.x1 + lw
        bd = C["accent"] if (self._popup is not None or self.hovered) else C["line"]
        d.rrect(bx1, self.y1 + 2, self.x2, self.y2 - 2, 8, C["card2"], bd, 1, self.tag)
        lbl = self._current_label()
        if len(lbl) > 24:
            lbl = lbl[:22] + "…"
        d.text(bx1 + 12, cy, lbl, C["text"], self.ui.f["mid"], "w", self.tag)
        d.text(self.x2 - 12, cy, "▾", C["muted"], self.ui.f["small"], "e", self.tag)

    def close(self):
        self.ui._last_popup_close = time.monotonic()
        if self.ui.active_dropdown is self:
            self.ui.active_dropdown = None
        pop = self._popup
        self._popup = None
        if pop is not None:
            try:
                pop.grab_release()
            except Exception:  # noqa: BLE001
                pass
            try:
                pop.destroy()
            except Exception:  # noqa: BLE001
                pass
        self.redraw()

    def _select(self, val):
        self.set(val)
        self.close()

    def on_press(self, x, y):
        # Toggle closed if already open; otherwise open on press (not release) so release never retriggers
        if self._popup is not None:
            self.close()
            return
        self.open()

    def open(self):
        if self._popup is not None or not self.items:
            return
        k = self.ui.k
        lw = min(115.0, self.w * 0.36)
        bx1 = self.x1 + lw
        box_w = int(round((self.x2 - bx1) * k))
        root_x = self.ui.cv.winfo_rootx() + int(round(bx1 * k))
        root_y = self.ui.cv.winfo_rooty() + int(round((self.y2 + 2) * k))

        pop = tk.Toplevel(self.ui.root)
        pop.overrideredirect(True)
        pop.configure(bg=C["line"])
        try:
            pop.attributes("-topmost", True)
        except Exception:  # noqa: BLE001
            pass

        row_h = max(28, int(round(28 * k)))
        max_rows = min(len(self.items), 11)
        pop_h = max_rows * row_h + 4

        # Flip upward if near bottom of screen
        scr_h = self.ui.root.winfo_screenheight()
        if root_y + pop_h > scr_h - 24:
            root_y = max(12, self.ui.cv.winfo_rooty() + int(round((self.y1 - 2) * k)) - pop_h)

        pop.geometry(f"{box_w}x{pop_h}+{root_x}+{root_y}")
        self._popup = pop
        self.ui.active_dropdown = self
        self.redraw()

        inner = tk.Frame(pop, bg=C["card2"], bd=0)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        cv = tk.Canvas(inner, bg=C["card2"], highlightthickness=0, bd=0, width=box_w - 2, height=pop_h - 2)
        cv.pack(side="left", fill="both", expand=True)

        total_h = len(self.items) * row_h
        cv.configure(scrollregion=(0, 0, box_w - 2, total_h))

        cur = self.get()
        row_items: list[tuple[int, int, object]] = []
        for idx, (val, lbl) in enumerate(self.items):
            ry1 = idx * row_h
            ry2 = ry1 + row_h
            is_cur = val == cur
            bg = C["accent"] if is_cur else C["card2"]
            fg = C["white"] if is_cur else C["text"]
            r_id = cv.create_rectangle(0, ry1, box_w, ry2, fill=bg, outline="")
            t_id = cv.create_text(int(12 * k), (ry1 + ry2) // 2, text=lbl, fill=fg, font=self.ui.f["body"], anchor="w")
            row_items.append((r_id, t_id, val))

        def _row_at(event_y: int) -> int:
            cy = int(cv.canvasy(event_y))
            return min(len(self.items) - 1, max(0, cy // row_h))

        def _on_motion(e):
            hi = _row_at(e.y)
            for idx, (r_id, t_id, val) in enumerate(row_items):
                if val == cur:
                    continue
                cv.itemconfigure(r_id, fill=mix(C["card2"], C["accent"], 0.35) if idx == hi else C["card2"])

        def _on_click(e):
            idx = _row_at(e.y)
            val = self.items[idx][0]
            self.ui.root.after(1, lambda v=val: self._select(v))
            return "break"

        def _on_wheel(e):
            if total_h > pop_h:
                delta = -1 if e.delta > 0 else 1
                cv.yview_scroll(delta, "units")

        def _on_outside_click(e):
            # Close if user clicked anywhere outside the popup window
            w = e.widget
            while w is not None:
                if w is pop:
                    return
                w = getattr(w, "master", None)
            self.close()
            return "break"

        cv.bind("<Motion>", _on_motion)
        cv.bind("<ButtonPress-1>", _on_click)
        cv.bind("<MouseWheel>", _on_wheel)
        pop.bind("<Escape>", lambda e: self.close())
        pop.bind("<ButtonPress-1>", _on_outside_click)
        try:
            pop.grab_set_global()
        except Exception:  # noqa: BLE001
            try:
                pop.grab_set()
            except Exception:  # noqa: BLE001
                pass
        pop.focus_force()


class BeatDots(W):
    interactive = False

    def __init__(self, ui: UI):
        super().__init__(ui)
        self.pos = -1
        self.pulse = 0.0

    def set(self, pos: int, pulse: float):
        if pos == self.pos and abs(pulse - self.pulse) < 0.05:
            return
        self.pos, self.pulse = pos, pulse
        self.redraw()

    def draw(self):
        d = self.ui.d
        cy = (self.y1 + self.y2) / 2
        gap = 36
        sx = self.x2 - 16 - 3 * gap
        for i in range(4):
            cx = sx + i * gap
            active = i == self.pos
            r = (9 + 3 * self.pulse) if active else 7
            col = (C["danger"] if i == 0 else C["accent"]) if active else C["card2"]
            bd = col if active else C["line"]
            d.oval(cx - r, cy - r, cx + r, cy + r, col, bd, 1, self.tag)


class Meter(W):
    interactive = False

    def __init__(self, ui: UI, name: str, color: str):
        super().__init__(ui)
        self.name, self.color = name.upper(), color
        self.val = 0.0

    def set(self, v: float):
        v = min(max(float(v), 0.0), 1.0)
        if abs(v - self.val) < 0.015:
            return
        self.val = v
        self.redraw()

    def draw(self):
        d = self.ui.d
        d.text(self.x1, self.y1 + 7, self.name, C["dim"], self.ui.f["tiny"], "w", self.tag)
        d.text(self.x2, self.y1 + 7, f"{int(round(self.val * 100))}%", C["muted"], self.ui.f["small"], "e", self.tag)
        ty = self.y1 + 21
        d.rrect(self.x1, ty - 3, self.x2, ty + 3, 3, C["card2"], "", 0, self.tag)
        fw = self.w * self.val
        if fw > 2:
            d.rrect(self.x1, ty - 3, self.x1 + fw, ty + 3, 3, self.color, "", 0, self.tag)


class Timeline(W):
    interactive = False

    def __init__(self, ui: UI, colors: dict):
        super().__init__(ui)
        self.colors = colors
        self.log: list = []
        self.t = 0.0

    def set(self, log: list, t: float):
        self.log, self.t = log, t
        self.redraw()

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, 6, C["card2"], C["line"], 1, self.tag)
        if not self.log or self.t <= 0.05:
            return
        total = max(self.t, 0.1)
        for idx, (ts, sec) in enumerate(self.log):
            te = self.log[idx + 1][0] if idx + 1 < len(self.log) else total
            bx1 = self.x1 + 2 + (self.w - 4) * min(max(ts / total, 0.0), 1.0)
            bx2 = self.x1 + 2 + (self.w - 4) * min(max(te / total, 0.0), 1.0)
            if bx2 - bx1 >= 1.5:
                d.rect(bx1, self.y1 + 2, bx2, self.y2 - 2, self.colors.get(sec, C["dim"]), "", 0, self.tag)


class Preview(W):
    interactive = False

    def __init__(self, ui: UI):
        super().__init__(ui)
        self.frame: np.ndarray | None = None
        self.header = ""
        self.blackout = False

    def update(self, frame: np.ndarray | None, header: str, blackout: bool):
        self.frame, self.header, self.blackout = frame, header, blackout
        self.redraw()

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, 12, "#060911", C["line"], 1, self.tag)
        pad = 24
        side = max(40.0, min(self.w - 2 * pad, self.h - 54))
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2 + 8
        hs = side / 2
        d.rect(cx - hs, cy - hs, cx + hs, cy + hs, "#04060c", "#182133", 1, self.tag)
        d.line([cx - hs, cy, cx + hs, cy], "#101726", 1, self.tag)
        d.line([cx, cy - hs, cx, cy + hs], "#101726", 1, self.tag)
        d.text(self.x1 + 16, self.y1 + 16, self.header or "PREVIEW", C["muted"], self.ui.f["small"], "w", self.tag)

        if self.blackout:
            d.text(cx, cy, "BLACKOUT", C["danger"], self.ui.f["title"], "center", self.tag)
            return
        fr = self.frame
        if fr is None or len(fr) == 0:
            d.text(cx, cy, "Ready", C["dim"], self.ui.f["body"], "center", self.tag)
            return

        xs = cx + fr[:, 0] * hs
        ys = cy - fr[:, 1] * hs
        rgb = np.clip(fr[:, 2:5], 0.0, 1.0)
        lit = rgb.max(axis=1) > 0.02

        i = 0
        n = len(fr)
        while i < n:
            if not lit[i]:
                i += 1
                continue
            j = i
            while j < n and lit[j]:
                j += 1
            seg_rgb = rgb[i:j].max(axis=0)
            col = f"#{int(seg_rgb[0] * 255):02x}{int(seg_rgb[1] * 255):02x}{int(seg_rgb[2] * 255):02x}"
            if j - i == 1:
                px, py = float(xs[i]), float(ys[i])
                d.oval(px - 2.5, py - 2.5, px + 2.5, py + 2.5, col, "", 0, self.tag)
            else:
                pts = []
                for idx in range(i, j):
                    pts.extend([float(xs[idx]), float(ys[idx])])
                d.line(pts, col, 2.2, self.tag)
            i = j