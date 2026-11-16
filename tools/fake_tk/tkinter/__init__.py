"""STRICT stand-in for tkinter, used only by tools/gui_preview.py to render the Canvas UI with Pillow.

It rejects canvas options that real Tk would not accept, so typos are caught without a display.
"""
from __future__ import annotations
import math
from PIL import Image, ImageDraw

_ALLOWED = {
    "rectangle": {"fill", "outline", "width", "tags"},
    "polygon": {"fill", "outline", "width", "smooth", "tags"},
    "oval": {"fill", "outline", "width", "tags"},
    "line": {"fill", "width", "capstyle", "joinstyle", "smooth", "tags"},
    "text": {"text", "fill", "font", "anchor", "tags", "width", "justify"},
}
_ANCHORS = {"nw", "n", "ne", "w", "center", "e", "sw", "s", "se"}


class TclError(Exception):
    pass


class Event:
    def __init__(self, x=0, y=0, delta=0, num=0, width=0, height=0):
        self.x, self.y, self.delta, self.num, self.width, self.height = x, y, delta, num, width, height


class Misc:
    def bind(self, seq, fn):
        self._binds = getattr(self, "_binds", {})
        self._binds[seq] = fn

    def fire(self, seq, **kw):
        fn = getattr(self, "_binds", {}).get(seq)
        if fn:
            fn(Event(**kw))


class Tk(Misc):
    def __init__(self):
        self.w, self.h, self._after = 1100, 700, []

    def title(self, t): pass
    def configure(self, **kw):
        assert set(kw) <= {"bg"}, kw
    def minsize(self, w, h): pass
    def geometry(self, g): self.w, self.h = map(int, g.split("x"))
    def protocol(self, n, f): pass
    def after(self, ms, fn): self._after.append(fn)
    def mainloop(self): pass
    def destroy(self): pass
    def update_idletasks(self): pass
    def winfo_fpixels(self, s): return 96.0


class Canvas(Misc):
    def __init__(self, master, **kw):
        bad = set(kw) - {"bg", "highlightthickness", "bd", "width", "height"}
        if bad:
            raise TclError(f"unknown canvas option {bad}")
        self.bg, self.items, self.cursor, self._id = kw.get("bg", "#fff"), [], "", 0

    def pack(self, **kw): pass

    def config(self, **kw):
        bad = set(kw) - {"cursor"}
        if bad:
            raise TclError(f"unknown config {bad}")
        self.cursor = kw.get("cursor", self.cursor)

    def _add(self, kind, coords, kw):
        bad = set(kw) - _ALLOWED[kind]
        if bad:
            raise TclError(f"unknown option(s) {bad} for {kind}")
        if kind == "text" and kw.get("anchor", "center") not in _ANCHORS:
            raise TclError("bad anchor")
        for k in ("fill", "outline"):
            v = kw.get(k)
            if v not in (None, "") and not (isinstance(v, str) and v.startswith("#") and len(v) == 7):
                raise TclError(f"bad colour {v!r}")
        t = kw.get("tags")
        tags = () if t is None else ((t,) if isinstance(t, str) else tuple(t))
        self._id += 1
        self.items.append(dict(id=self._id, kind=kind, coords=[float(c) for c in coords], opts=kw, tags=tags))
        return self._id

    @staticmethod
    def _flat(a):
        out = []
        for v in a:
            out += list(v) if isinstance(v, (list, tuple)) else [v]
        return out

    def create_rectangle(self, *a, **kw): return self._add("rectangle", self._flat(a), kw)
    def create_polygon(self, *a, **kw): return self._add("polygon", self._flat(a), kw)
    def create_oval(self, *a, **kw): return self._add("oval", self._flat(a), kw)
    def create_line(self, *a, **kw): return self._add("line", self._flat(a), kw)
    def create_text(self, *a, **kw): return self._add("text", self._flat(a), kw)

    def delete(self, tag):
        self.items = [i for i in self.items if tag != "all" and tag not in i["tags"]]

    def tag_raise(self, tag):
        a = [i for i in self.items if tag not in i["tags"]]
        b = [i for i in self.items if tag in i["tags"]]
        self.items = a + b

    # ------------------------------------------------------------ rendering
    @staticmethod
    def _smooth(pts, closed):
        P = [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)]
        n = len(P)
        if n < 3:
            return P
        mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)  # noqa: E731
        out = []
        rng = range(n) if closed else range(1, n - 1)
        if not closed:
            out.append(P[0])
        for i in rng:
            m0, m1 = mid(P[i - 1], P[i]), mid(P[i], P[(i + 1) % n])
            if not closed and i == 1:
                m0 = P[0]
            if not closed and i == n - 2:
                m1 = P[-1]
            for s in range(0, 13):
                t = s / 12
                out.append(((1 - t) ** 2 * m0[0] + 2 * (1 - t) * t * P[i][0] + t * t * m1[0],
                            (1 - t) ** 2 * m0[1] + 2 * (1 - t) * t * P[i][1] + t * t * m1[1]))
        if not closed:
            out.append(P[-1])
        return out

    def render(self, w, h, ss=3):
        from .font import _pil
        im = Image.new("RGB", (w * ss, h * ss), self.bg)
        d = ImageDraw.Draw(im)
        S = lambda v: v * ss  # noqa: E731
        for it in self.items:
            o, c, k = it["opts"], it["coords"], it["kind"]
            fill, outl = o.get("fill") or None, o.get("outline") or None
            wd = max(1, int(round(S(o.get("width", 1) or 1)))) if (outl or k == "line") else 0
            if k == "rectangle":
                d.rectangle([S(c[0]), S(c[1]), S(c[2]), S(c[3])], fill=fill, outline=outl, width=wd)
            elif k == "oval":
                d.ellipse([S(c[0]), S(c[1]), S(c[2]), S(c[3])], fill=fill, outline=outl, width=wd)
            elif k == "polygon":
                pts = self._smooth(c, True) if o.get("smooth") else [(c[i], c[i + 1]) for i in range(0, len(c), 2)]
                pts = [(S(x), S(y)) for x, y in pts]
                d.polygon(pts, fill=fill)
                if outl:
                    d.line(pts + [pts[0]], fill=outl, width=wd, joint="curve")
            elif k == "line":
                pts = self._smooth(c, False) if o.get("smooth") else [(c[i], c[i + 1]) for i in range(0, len(c), 2)]
                pts = [(S(x), S(y)) for x, y in pts]
                d.line(pts, fill=fill, width=wd, joint="curve")
                if o.get("capstyle") == "round" and wd > 2:
                    r = wd / 2
                    for x, y in (pts[0], pts[-1]):
                        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            elif k == "text":
                f = o["font"]
                fnt = _pil(f, ss)
                a = {"nw": "la", "w": "lm", "ne": "ra", "e": "rm", "center": "mm", "n": "ma", "sw": "ld", "s": "md", "se": "rd"}[o.get("anchor", "center")]
                d.text((S(c[0]), S(c[1])), o["text"], fill=fill, font=fnt, anchor=a)
        return im.resize((w, h), Image.LANCZOS)


class Frame(Misc):
    pass
