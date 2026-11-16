"""A tiny retained-mode widget kit drawn on ONE tkinter Canvas: flat, dark, rounded, modern.

Only the standard library (tkinter) is used, so the packaged app needs nothing else.
All geometry is in logical units; `UI.k` scales for high-DPI screens.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

C = dict(
    bg="#0c0d11", panel="#14161c", panel2="#1b1e27", panel3="#232734", line="#2a2e3b", text="#eceef5",
    muted="#8d93a6", dim="#5d6376", accent="#7c6cff", accent2="#22d3ee", ok="#22c55e", warn="#f59e0b",
    danger="#ef4444", white="#ffffff",
)


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    ra, rb = _rgb(a), _rgb(b)
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * t) for x, y in zip(ra, rb))


def pick_family(root):
    have = set(tkfont.families(root))
    for f in ("Segoe UI Variable Text", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Roboto", "Ubuntu",
              "DejaVu Sans", "Helvetica", "Arial"):
        if f in have:
            return f
    return "TkDefaultFont"


class Draw:
    """Canvas drawing helpers that apply the DPI scale and tag items."""

    def __init__(self, canvas, k=1.0):
        self.c, self.k = canvas, k

    def _s(self, pts):
        return [v * self.k for v in pts]

    def rect(self, x1, y1, x2, y2, fill="", outline="", w=0, tag=None):
        return self.c.create_rectangle(*self._s((x1, y1, x2, y2)), fill=fill, outline=outline,
                                       width=w * self.k if outline else 0, tags=tag)

    def rrect(self, x1, y1, x2, y2, r, fill="", outline="", w=1, tag=None):
        r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        if r < 1:
            return self.rect(x1, y1, x2, y2, fill, outline, w, tag)
        pts = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
               x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.c.create_polygon(self._s(pts), smooth=True, fill=fill, outline=outline,
                                     width=w * self.k if outline else 0, tags=tag)

    def oval(self, x1, y1, x2, y2, fill="", outline="", w=1, tag=None):
        return self.c.create_oval(*self._s((x1, y1, x2, y2)), fill=fill, outline=outline,
                                  width=w * self.k if outline else 0, tags=tag)

    def line(self, pts, fill, w=1, tag=None, smooth=False):
        return self.c.create_line(self._s(pts), fill=fill, width=max(1, w * self.k), capstyle="round",
                                  joinstyle="round", smooth=smooth, tags=tag)

    def poly(self, pts, fill, tag=None):
        return self.c.create_polygon(self._s(pts), fill=fill, outline="", tags=tag)

    def text(self, x, y, text, fill, font, anchor="nw", tag=None):
        return self.c.create_text(x * self.k, y * self.k, text=text, fill=fill, font=font, anchor=anchor, tags=tag)


class W:
    """Base widget: a rectangle that can draw itself and react to the mouse."""
    interactive = True
    _n = 0

    def __init__(self, ui):
        self.ui = ui
        W._n += 1
        self.tag = f"w{W._n}"
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        self.visible, self.hover, self.pressed = True, False, False
        ui.widgets.append(self)

    def place(self, x, y, w, h):
        self.x1, self.y1, self.x2, self.y2 = x, y, x + w, y + h
        return self

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    def hit(self, x, y):
        return self.visible and self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def draw(self):
        pass

    def redraw(self):
        if not self.visible:
            return
        self.ui.c.delete(self.tag)
        self.draw()
        self.ui.raise_popup()

    # mouse events (logical coordinates)
    def press(self, x, y): self.pressed = True; self.redraw()
    def drag(self, x, y): pass
    def release(self, x, y, inside): self.pressed = False; self.redraw()


class UI:
    def __init__(self, root, canvas, k=1.0):
        self.root, self.c, self.k = root, canvas, k
        self.d = Draw(canvas, k)
        fam = pick_family(root)
        mk = lambda size, weight="normal": tkfont.Font(root=root, family=fam, size=size, weight=weight)  # noqa: E731
        self.f = dict(body=mk(10), small=mk(9), tiny=mk(8, "bold"), bold=mk(10, "bold"), h=mk(11, "bold"),
                      title=mk(15, "bold"), big=mk(40, "bold"), mid=mk(13, "bold"))
        self.widgets: list[W] = []
        self.hover = self.grab = self.popup = None
        self.logical_h = 700
        canvas.bind("<Motion>", self._motion)
        canvas.bind("<Button-1>", self._press)
        canvas.bind("<B1-Motion>", self._drag)
        canvas.bind("<ButtonRelease-1>", self._release)
        canvas.bind("<Leave>", self._leave)
        canvas.bind("<MouseWheel>", self._wheel)
        canvas.bind("<Button-4>", self._wheel)
        canvas.bind("<Button-5>", self._wheel)

    def fit(self, font, text, maxw):
        """Shorten `text` with an ellipsis so it is at most maxw logical px wide."""
        f = self.f[font]
        if f.measure(text) <= maxw * self.k:
            return text
        while len(text) > 1 and f.measure(text + "\u2026") > maxw * self.k:
            text = text[:-1]
        return text + "\u2026"

    # ------------------------------------------------------------- plumbing
    def redraw_all(self):
        self.c.delete("all")
        self.popup = None
        for w in self.widgets:
            if w.visible:
                w.draw()

    def raise_popup(self):
        if self.popup:
            self.c.tag_raise("popup")

    def open_popup(self, p):
        self.close_popup()
        self.popup = p
        p.draw()

    def close_popup(self):
        if self.popup:
            self.c.delete("popup")
            self.popup = None

    def _xy(self, e):
        return e.x / self.k, e.y / self.k

    def _top(self, x, y):
        if self.popup and self.popup.hit(x, y):
            return self.popup
        for w in reversed(self.widgets):
            if w.interactive and w.hit(x, y):
                return w
        return None

    def _motion(self, e):
        x, y = self._xy(e)
        if self.grab:
            return
        w = self._top(x, y)
        if w is not self.hover:
            old, self.hover = self.hover, w
            if old:
                old.hover = False
                old.redraw()
            if w:
                w.hover = True
                w.redraw()
            self.c.config(cursor="hand2" if w else "")
        if w is not None and hasattr(w, "move"):
            w.move(x, y)

    def _leave(self, e):
        if self.hover and not self.grab:
            self.hover.hover = False
            self.hover.redraw()
            self.hover = None

    def _press(self, e):
        x, y = self._xy(e)
        if self.popup and not self.popup.hit(x, y):
            self.close_popup()
            return
        w = self._top(x, y)
        if w:
            self.grab = w
            w.press(x, y)

    def _drag(self, e):
        if self.grab:
            self.grab.drag(*self._xy(e))

    def _release(self, e):
        if self.grab:
            x, y = self._xy(e)
            g, self.grab = self.grab, None
            g.release(x, y, g.hit(x, y))

    def _wheel(self, e):
        x, y = self._xy(e)
        up = getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0
        if self.popup and hasattr(self.popup, "scroll"):
            self.popup.scroll(-1 if up else 1)


# ------------------------------------------------------------------- widgets
class Label(W):
    interactive = False

    def __init__(self, ui, text="", font="body", color=None, anchor="nw"):
        super().__init__(ui)
        self.text, self.font, self.color, self.anchor = text, font, color or C["text"], anchor

    def set(self, text=None, color=None):
        changed = (text is not None and text != self.text) or (color is not None and color != self.color)
        if text is not None:
            self.text = text
        if color is not None:
            self.color = color
        if changed:
            self.redraw()

    def draw(self):
        x = {"nw": self.x1, "w": self.x1, "ne": self.x2, "e": self.x2, "center": (self.x1 + self.x2) / 2}[self.anchor]
        y = (self.y1 + self.y2) / 2 if self.anchor in ("w", "e", "center") else self.y1
        self.ui.d.text(x, y, self.text, self.color, self.ui.f[self.font], self.anchor, self.tag)


class Card(W):
    interactive = False

    def __init__(self, ui, title="", fill=None, radius=14):
        super().__init__(ui)
        self.title, self.fill, self.radius = title, fill or C["panel"], radius

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, self.radius, self.fill, C["line"], 1, self.tag)
        if self.title:
            d.text(self.x1 + 16, self.y1 + 12, self.title.upper(), C["dim"], self.ui.f["tiny"], "nw", self.tag)


class Button(W):
    KINDS = {"primary": (C["accent"], C["white"], None), "success": ("#16a34a", C["white"], None),
             "danger": ("#dc2626", C["white"], None), "ghost": (C["panel2"], C["text"], C["line"]),
             "outline_danger": (C["panel"], "#f87171", "#7f1d1d")}

    def __init__(self, ui, text, cb, kind="ghost", font="bold", radius=10):
        super().__init__(ui)
        self.text, self.cb, self.kind, self.font, self.radius = text, cb, kind, font, radius

    def set(self, text=None, kind=None):
        if (text, kind) != (None, None) and ((text or self.text) != self.text or (kind or self.kind) != self.kind):
            self.text, self.kind = text or self.text, kind or self.kind
            self.redraw()

    def draw(self):
        bg, fg, ol = self.KINDS[self.kind]
        if self.pressed:
            bg = mix(bg, "#000000", 0.25)
        elif self.hover:
            bg = mix(bg, "#ffffff", 0.12)
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, self.radius, bg, ol or bg, 1, self.tag)
        d.text((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.text, fg, self.ui.f[self.font], "center", self.tag)

    def release(self, x, y, inside):
        self.pressed = False
        self.redraw()
        if inside and self.cb:
            self.cb()


class Segmented(W):
    def __init__(self, ui, options, get, set_, on_change=None):
        super().__init__(ui)
        self.options, self.get, self.set_, self.on_change, self._hover_i = options, get, set_, on_change, -1

    def _seg(self, i):
        w = (self.w - 8) / len(self.options)
        return self.x1 + 4 + i * w, self.x1 + 4 + (i + 1) * w

    def draw(self):
        d, cur = self.ui.d, self.get()
        d.rrect(self.x1, self.y1, self.x2, self.y2, 10, C["panel2"], C["line"], 1, self.tag)
        for i, (val, label) in enumerate(self.options):
            a, b = self._seg(i)
            on = val == cur
            if on:
                d.rrect(a, self.y1 + 4, b, self.y2 - 4, 7, C["accent"], C["accent"], 1, self.tag)
            elif self.hover and i == self._hover_i:
                d.rrect(a, self.y1 + 4, b, self.y2 - 4, 7, C["panel3"], C["panel3"], 1, self.tag)
            d.text((a + b) / 2, (self.y1 + self.y2) / 2, label, C["white"] if on else C["muted"], self.ui.f["bold"],
                   "center", self.tag)

    def _idx(self, x):
        return min(max(int((x - self.x1 - 4) // ((self.w - 8) / len(self.options))), 0), len(self.options) - 1)

    def move(self, x, y):
        i = self._idx(x)
        if i != self._hover_i:
            self._hover_i = i
            self.redraw()

    def press(self, x, y):
        self.set_(self.options[self._idx(x)][0])
        if self.on_change:
            self.on_change()
        self.redraw()

    def release(self, x, y, inside):
        pass


class Toggle(W):
    def __init__(self, ui, text, get, set_, sub=None):
        super().__init__(ui)
        self.text, self.get, self.set_, self.sub = text, get, set_, sub

    def draw(self):
        d, on = self.ui.d, self.get()
        cy = (self.y1 + self.y2) / 2
        if self.sub:
            d.text(self.x1, cy - 8, self.text, C["text"], self.ui.f["body"], "w", self.tag)
            d.text(self.x1, cy + 9, self.sub, C["dim"], self.ui.f["small"], "w", self.tag)
        else:
            d.text(self.x1, cy, self.text, C["text"], self.ui.f["body"], "w", self.tag)
        tx2, tw, th = self.x2, 40, 22
        col = C["accent"] if on else C["panel3"]
        if self.hover:
            col = mix(col, "#ffffff", 0.1)
        d.rrect(tx2 - tw, cy - th / 2, tx2, cy + th / 2, th / 2, col, col, 1, self.tag)
        kx = tx2 - th / 2 - 0 if on else tx2 - tw + th / 2
        d.oval(kx - 8, cy - 8, kx + 8, cy + 8, C["white"], "", 0, self.tag)

    def press(self, x, y):
        self.set_(not self.get())
        self.redraw()

    def release(self, x, y, inside):
        pass


class Chip(W):
    def __init__(self, ui, text, get, set_):
        super().__init__(ui)
        self.text, self.get, self.set_ = text, get, set_

    def draw(self):
        on = self.get()
        bg = C["accent"] if on else C["panel2"]
        if self.hover:
            bg = mix(bg, "#ffffff", 0.1)
        self.ui.d.rrect(self.x1, self.y1, self.x2, self.y2, self.h / 2, bg, C["accent"] if on else C["line"], 1, self.tag)
        self.ui.d.text((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.text, C["white"] if on else C["muted"],
                       self.ui.f["small"], "center", self.tag)

    def press(self, x, y):
        self.set_(not self.get())
        self.redraw()

    def release(self, x, y, inside):
        pass


class Slider(W):
    def __init__(self, ui, text, lo, hi, get, set_, fmt="{:.2f}"):
        super().__init__(ui)
        self.text, self.lo, self.hi, self.get, self.set_, self.fmt = text, lo, hi, get, set_, fmt

    def _track(self):
        return self.x1 + 8, self.x2 - 8, self.y2 - 10

    def draw(self):
        d, (a, b, y) = self.ui.d, self._track()
        v = self.get()
        t = (v - self.lo) / (self.hi - self.lo)
        d.text(self.x1, self.y1 + 2, self.text, C["text"], self.ui.f["body"], "nw", self.tag)
        d.text(self.x2, self.y1 + 2, self.fmt.format(v), C["muted"], self.ui.f["body"], "ne", self.tag)
        d.rrect(a, y - 2, b, y + 2, 2, C["panel3"], C["panel3"], 1, self.tag)
        if t > 0.01:
            d.rrect(a, y - 2, a + (b - a) * t, y + 2, 2, C["accent"], C["accent"], 1, self.tag)
        r = 8 if (self.hover or self.pressed) else 7
        kx = a + (b - a) * t
        d.oval(kx - r, y - r, kx + r, y + r, C["white"], C["accent"], 2, self.tag)

    def _set_from(self, x):
        a, b, _ = self._track()
        t = min(max((x - a) / (b - a), 0.0), 1.0)
        self.set_(self.lo + t * (self.hi - self.lo))
        self.redraw()

    def press(self, x, y):
        self.pressed = True
        self._set_from(x)

    def drag(self, x, y):
        self._set_from(x)


class Stepper(W):
    def __init__(self, ui, text, get, set_, step, lo, hi, fmt="{}"):
        super().__init__(ui)
        self.text, self.get, self.set_, self.step, self.lo, self.hi, self.fmt = text, get, set_, step, lo, hi, fmt

    def _btns(self):
        cy = (self.y1 + self.y2) / 2
        return (self.x2 - 124, cy - 13, self.x2 - 98, cy + 13), (self.x2 - 26, cy - 13, self.x2, cy + 13)

    def draw(self):
        d = self.ui.d
        cy = (self.y1 + self.y2) / 2
        d.text(self.x1, cy, self.text, C["text"], self.ui.f["body"], "w", self.tag)
        for (a, b, c_, e), glyph in zip(self._btns(), ("−", "+")):
            d.rrect(a, b, c_, e, 7, C["panel2"], C["line"], 1, self.tag)
            d.text((a + c_) / 2, (b + e) / 2, glyph, C["text"], self.ui.f["h"], "center", self.tag)
        d.text(self.x2 - 62, cy, self.fmt.format(self.get()), C["text"], self.ui.f["bold"], "center", self.tag)

    def press(self, x, y):
        m, p = self._btns()
        if m[0] <= x <= m[2]:
            self.set_(max(self.lo, self.get() - self.step))
        elif p[0] <= x <= p[2]:
            self.set_(min(self.hi, self.get() + self.step))
        self.redraw()

    def release(self, x, y, inside):
        pass


class DropList(W):
    ROW, MAXROWS = 28, 8

    def __init__(self, ui, anchor, options, current, on_pick):
        super().__init__(ui)
        ui.widgets.remove(self)             # popups are managed by UI, not the widget list
        self.options, self.current, self.on_pick, self.off, self.hrow = options, current, on_pick, 0, -1
        n = min(len(options), self.MAXROWS)
        h = n * self.ROW + 8
        ax1, ay1, ax2, ay2 = anchor
        self.x1, self.x2 = ax1, ax2
        below = ay2 + 6
        self.y1 = below if below + h < ui.logical_h - 8 else ay1 - 6 - h
        self.y2 = self.y1 + h
        if current in [v for v, _ in options]:
            i = [v for v, _ in options].index(current)
            self.off = min(max(i - n // 2, 0), max(len(options) - n, 0))
        self.tag = "popup"

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1 + 2, self.y1 + 4, self.x2 + 2, self.y2 + 4, 10, "#05060a", "", 0, "popup")
        d.rrect(self.x1, self.y1, self.x2, self.y2, 10, C["panel2"], C["line"], 1, "popup")
        n = min(len(self.options), self.MAXROWS)
        for r in range(n):
            val, label = self.options[self.off + r]
            y = self.y1 + 4 + r * self.ROW
            if self.off + r == self.hrow:
                d.rrect(self.x1 + 4, y, self.x2 - 4, y + self.ROW, 6, C["panel3"], C["panel3"], 1, "popup")
            on = val == self.current
            d.text(self.x1 + 14, y + self.ROW / 2, label, C["accent2"] if on else C["text"], self.ui.f["body"], "w", "popup")
        if len(self.options) > n:
            frac, tot = n / len(self.options), self.y2 - self.y1 - 12
            y0 = self.y1 + 6 + tot * (self.off / len(self.options))
            d.rrect(self.x2 - 7, y0, self.x2 - 4, y0 + tot * frac, 1.5, C["dim"], C["dim"], 1, "popup")

    def _row(self, y):
        return self.off + int((y - self.y1 - 4) // self.ROW)

    def move(self, x, y):
        r = self._row(y)
        if r != self.hrow and 0 <= r < len(self.options):
            self.hrow = r
            self.ui.c.delete("popup")
            self.draw()

    def scroll(self, d):
        self.off = min(max(self.off + d, 0), max(len(self.options) - self.MAXROWS, 0))
        self.ui.c.delete("popup")
        self.draw()

    def press(self, x, y):
        r = self._row(y)
        if 0 <= r < len(self.options):
            self.ui.close_popup()
            self.on_pick(self.options[r][0])

    def release(self, x, y, inside):
        pass


class Dropdown(W):
    def __init__(self, ui, text, options, get, set_):
        super().__init__(ui)
        self.text, self.options, self.get, self.set_ = text, options, get, set_

    def _field(self):
        return self.x2 - 190, self.y1 + 2, self.x2, self.y2 - 2

    def draw(self):
        d = self.ui.d
        cy = (self.y1 + self.y2) / 2
        d.text(self.x1, cy, self.text, C["text"], self.ui.f["body"], "w", self.tag)
        a, b, c_, e = self._field()
        bg = C["panel3"] if self.hover else C["panel2"]
        d.rrect(a, b, c_, e, 9, bg, C["line"], 1, self.tag)
        cur = self.ui.fit("body", dict(self.options).get(self.get(), str(self.get())), c_ - a - 40)
        d.text(a + 12, cy, cur, C["text"], self.ui.f["body"], "w", self.tag)
        d.poly([c_ - 22, cy - 2, c_ - 12, cy - 2, c_ - 17, cy + 4], C["muted"], self.tag)

    def press(self, x, y):
        if x >= self._field()[0]:
            self.ui.open_popup(DropList(self.ui, self._field(), self.options, self.get(), self._pick))

    def release(self, x, y, inside):
        pass

    def _pick(self, v):
        self.set_(v)
        self.redraw()


class Meter(W):
    interactive = False

    def __init__(self, ui, text, color=None):
        super().__init__(ui)
        self.text, self.color, self.v = text, color or C["accent2"], 0.0

    def set(self, v):
        v = min(max(v, 0.0), 1.0)
        if abs(v - self.v) > 0.01:
            self.v = v
            self.redraw()

    def draw(self):
        d = self.ui.d
        d.text(self.x1, self.y1, self.text.upper(), C["dim"], self.ui.f["tiny"], "nw", self.tag)
        y = self.y1 + 22
        d.rrect(self.x1, y, self.x2, y + 6, 3, C["panel3"], C["panel3"], 1, self.tag)
        if self.v > 0.02:
            d.rrect(self.x1, y, self.x1 + (self.x2 - self.x1) * self.v, y + 6, 3,
                    mix(self.color, C["danger"], max(0.0, self.v - 0.8) * 4), "", 0, self.tag)


class BeatDots(W):
    interactive = False

    def __init__(self, ui):
        super().__init__(ui)
        self.active, self.pulse = -1, 0.0

    def set(self, active, pulse):
        pulse = round(pulse, 1)
        if (active, pulse) != (self.active, self.pulse):
            self.active, self.pulse = active, pulse
            self.redraw()

    def draw(self):
        d, gap = self.ui.d, 8
        w = (self.w - 3 * gap) / 4
        for i in range(4):
            a = self.x1 + i * (w + gap)
            if i == self.active:
                col = mix(C["panel3"], C["white"], 0.35 + 0.65 * self.pulse)
                fg = C["bg"]
            else:
                col, fg = C["panel2"], C["dim"]
            d.rrect(a, self.y1, a + w, self.y2, 8, col, C["line"] if i != self.active else col, 1, self.tag)
            d.text(a + w / 2, (self.y1 + self.y2) / 2, str(i + 1), fg, self.ui.f["bold"], "center", self.tag)


class Pill(W):
    interactive = False

    def __init__(self, ui, text="IDLE", color=None):
        super().__init__(ui)
        self.text, self.color = text, color or C["dim"]

    def set(self, text, color):
        if (text, color) != (self.text, self.color):
            self.text, self.color = text, color
            self.redraw()

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, self.h / 2, mix(self.color, C["panel"], 0.78), self.color, 1, self.tag)
        d.oval(self.x1 + 12, (self.y1 + self.y2) / 2 - 4, self.x1 + 20, (self.y1 + self.y2) / 2 + 4, self.color, "", 0, self.tag)
        d.text(self.x1 + 28, (self.y1 + self.y2) / 2, self.text, C["text"], self.ui.f["bold"], "w", self.tag)


class Timeline(W):
    interactive = False

    def __init__(self, ui, colors):
        super().__init__(ui)
        self.colors, self.log, self.now, self._sig = colors, [], 0.0, None

    def set(self, log, now):
        sig = (len(log), int(now))
        if sig != self._sig:
            self._sig, self.log, self.now = sig, list(log), now
            self.redraw()

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, 6, C["panel2"], C["panel2"], 1, self.tag)
        if not self.log:
            return
        span = max(self.now, 60.0)
        t0 = max(0.0, self.now - span)
        wpx = self.w - 4
        ends = [t for t, _ in self.log[1:]] + [self.now]
        for (t, sec), e in zip(self.log, ends):
            a = self.x1 + 2 + wpx * (max(t, t0) - t0) / span
            b = self.x1 + 2 + wpx * (e - t0) / span
            if b - a < 1.5:
                continue
            d.rrect(a, self.y1 + 3, b, self.y2 - 3, 3, self.colors.get(sec, C["dim"]), "", 0, self.tag)
            if (b - a) * self.ui.k > self.ui.f["tiny"].measure(sec) + 16 * self.ui.k:
                d.text((a + b) / 2, (self.y1 + self.y2) / 2, sec, C["white"], self.ui.f["tiny"], "center", self.tag)
        x = self.x1 + 2 + wpx * (self.now - t0) / span
        d.line([x, self.y1 + 1, x, self.y2 - 1], C["white"], 2, self.tag)


class Preview(W):
    """Laser stage: dark panel, faint guides, and the live frame drawn with a soft glow."""
    interactive = False

    def __init__(self, ui):
        super().__init__(ui)
        self.frame, self.msg = None, "Press START to preview"
        self.blackout, self.header = False, ""

    def square(self):
        s = min(self.w, self.h) - 36
        cx, cy = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2 + 6
        return cx - s / 2, cy - s / 2, s

    def draw(self):
        d = self.ui.d
        d.rrect(self.x1, self.y1, self.x2, self.y2, 16, "#07080b", C["line"], 1, self.tag)
        x0, y0, s = self.square()
        d.rrect(x0, y0, x0 + s, y0 + s, 6, "#0a0b10", "#171a22", 1, self.tag)
        for t in (0.25, 0.5, 0.75):
            col = "#14171f" if t != 0.5 else "#1b1f2a"
            d.line([x0 + s * t, y0 + 4, x0 + s * t, y0 + s - 4], col, 1, self.tag)
            d.line([x0 + 4, y0 + s * t, x0 + s - 4, y0 + s * t], col, 1, self.tag)
        d.text(self.x1 + 16, self.y1 + 12, "LIVE PREVIEW", C["dim"], self.ui.f["tiny"], "nw", self.tag)
        self._dyn()

    def _dyn(self):
        d = self.ui.d
        t = self.tag + "d"
        self.ui.c.delete(t)
        d.text(self.x2 - 16, self.y1 + 12, self.header, C["muted"], self.ui.f["small"], "ne", t)
        x0, y0, s = self.square()
        fr = self.frame
        if fr is None:
            d.text(x0 + s / 2, y0 + s / 2, self.msg, C["dim"], self.ui.f["mid"], "center", t)
            return
        px, py = x0 + (fr[:, 0] + 1) * 0.5 * s, y0 + (1 - fr[:, 1]) * 0.5 * s
        lit = fr[:, 2:5].max(1) > 0.01
        step = 6
        chunks = []
        for i in range(0, len(fr) - 1, step):
            j = min(i + step, len(fr) - 1)
            if lit[i] and lit[j]:
                rgb = tuple(min(255, int(max(v * 2.2, 0.0) * 255)) for v in fr[i, 2:5])
                chunks.append((i, j, "#%02x%02x%02x" % rgb))
        glow = len(chunks) <= 220
        for i, j, col in chunks:
            pts = []
            for q in range(i, j + 1):
                pts += [float(px[q]), float(py[q])]
            if glow:
                d.line(pts, mix(col, "#07080b", 0.72), 5, t)
        for i, j, col in chunks:
            pts = []
            for q in range(i, j + 1):
                pts += [float(px[q]), float(py[q])]
            d.line(pts, col, 2, t)
        if self.blackout:
            d.rrect(x0 + s / 2 - 90, y0 + s / 2 - 22, x0 + s / 2 + 90, y0 + s / 2 + 22, 10, "#3b0d0d", "#ef4444", 1, t)
            d.text(x0 + s / 2, y0 + s / 2, "BLACKOUT", "#fecaca", self.ui.f["mid"], "center", t)
        self.ui.raise_popup()

    def update(self, frame, header, blackout):
        self.frame, self.header, self.blackout = frame, header, blackout
        self._dyn()
