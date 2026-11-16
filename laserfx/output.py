"""Outputs: Helios DAC (USB, via native pyusb or official HeliosDacAPI shared library),
high-FPS blitted matplotlib simulator, and null output.

Optimized for high frame rates and low-latency smooth motion:
- High-resolution Windows multimedia timer (timeBeginPeriod(1)) so sub-millisecond sleeps don't stall at 15ms.
- Non-blocking latest-frame async writer option / fast spin-poll for HeliosUSB & HeliosDAC.
- Matplotlib canvas blitting in SimDisplay (120+ FPS preview without blocking frame generation).
- Zero-loop vectorized Helios USB packet packing.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
import threading
import time

import numpy as np

HELIOS_MAX_POINTS = 0x1000
HELIOS_MAX_PPS = 65535
_PT = np.dtype([("x", "<u2"), ("y", "<u2"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("i", "u1")])


def _enable_high_res_timer():
    """Sets Windows OS sleep granularity to 1 ms instead of the default 15.6 ms."""
    if sys.platform == "win32":
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:  # noqa: BLE001
            pass


_enable_high_res_timer()


def _precise_sleep(seconds: float):
    """High-precision sleep with short spin-wait tail for jitter-free frame pacing."""
    if seconds <= 0.0:
        return
    target = time.perf_counter() + seconds
    if seconds > 0.002:
        time.sleep(seconds - 0.0015)
    while time.perf_counter() < target:
        pass


def to_helios_array(frame: np.ndarray, flip_x: bool = False, flip_y: bool = False, swap_xy: bool = False) -> np.ndarray:
    n = len(frame)
    a = np.empty(n, _PT)
    x = frame[:, 1] if swap_xy else frame[:, 0]
    y = frame[:, 0] if swap_xy else frame[:, 1]
    if flip_x:
        x = -x
    if flip_y:
        y = -y
    a["x"] = np.clip((x + 1.0) * 2047.5, 0, 4095)
    a["y"] = np.clip((y + 1.0) * 2047.5, 0, 4095)
    rgb = np.clip(frame[:, 2:5] * 255.0, 0, 255).astype(np.uint8)
    a["r"] = rgb[:, 0]
    a["g"] = rgb[:, 1]
    a["b"] = rgb[:, 2]
    a["i"] = 255
    return a


def pack_helios_frame(frame: np.ndarray, pps: int, flip_x: bool = False, flip_y: bool = False, swap_xy: bool = False) -> bytes:
    """Vectorized Helios bulk packet: 7 bytes/point (12-bit x,y packed, r,g,b,i) + 5-byte footer."""
    pts = frame[:HELIOS_MAX_POINTS]
    n = len(pts)
    if n >= 45 and (n - 45) % 64 == 0:   # firmware cannot receive transfers of these sizes
        pts = pts[:-1]
        n -= 1

    x_f = pts[:, 1] if swap_xy else pts[:, 0]
    y_f = pts[:, 0] if swap_xy else pts[:, 1]
    if flip_x:
        x_f = -x_f
    if flip_y:
        y_f = -y_f

    x = np.clip((x_f + 1.0) * 2047.5, 0, 4095).astype(np.uint16)
    y = np.clip((y_f + 1.0) * 2047.5, 0, 4095).astype(np.uint16)
    rgb = np.clip(pts[:, 2:5] * 255.0, 0, 255).astype(np.uint8)

    b = np.empty((n, 7), np.uint8)
    b[:, 0] = (x >> 4).astype(np.uint8)
    b[:, 1] = (((x & 0xF) << 4) | (y >> 8)).astype(np.uint8)
    b[:, 2] = (y & 0xFF).astype(np.uint8)
    b[:, 3:6] = rgb
    b[:, 6] = 255

    pps = int(min(max(pps, 7), HELIOS_MAX_PPS))
    return b.tobytes() + bytes([pps & 0xFF, pps >> 8, n & 0xFF, n >> 8, 0])


class _AsyncFrameSender:
    """Latest-frame background sender so USB I/O never stalls frame generation or adds queue latency."""

    def __init__(self, write_fn):
        self._write_fn = write_fn
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: np.ndarray | None = None
        self._error: Exception | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray):
        if self._error is not None:
            err, self._error = self._error, None
            raise err
        with self._lock:
            self._latest = frame
        self._event.set()

    def _run(self):
        while self._running:
            if not self._event.wait(timeout=0.05):
                continue
            self._event.clear()
            with self._lock:
                frame = self._latest
                self._latest = None
            if frame is not None and self._running:
                try:
                    self._write_fn(frame)
                except Exception as e:  # noqa: BLE001
                    self._error = e
                    self._running = False
                    break

    def stop(self):
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
            self._thread = None


class HeliosUSB:
    """Native Helios USB driver (pyusb + libusb) with async latest-frame dispatch."""
    VID, PID = 0x1209, 0xE500
    EP_BULK_OUT, EP_INT_OUT, EP_INT_IN = 0x02, 0x06, 0x83
    SDK_VERSION = 6

    def __init__(self, index=0, pps=30000, flip_x=False, flip_y=False, swap_xy=False, async_io=True):
        self.index, self.pps = index, pps
        self.flip = dict(flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)
        self.dev = None
        self._async_io = async_io
        self._sender: _AsyncFrameSender | None = None

    def open(self):
        try:
            import usb.core
            import usb.util
        except ImportError as e:
            raise RuntimeError("pyusb is not installed (pip install pyusb libusb-package)") from e
        backend = None
        try:
            import libusb_package
            backend = libusb_package.get_libusb1_backend()
        except Exception:  # noqa: BLE001
            pass
        try:
            devs = list(usb.core.find(find_all=True, idVendor=self.VID, idProduct=self.PID, backend=backend))
        except usb.core.NoBackendError as e:
            raise RuntimeError("No libusb backend available (pip install libusb-package)") from e
        if self.index >= len(devs):
            raise RuntimeError("No Helios DAC found. Check the USB cable"
                               + (" and the Linux udev rule (see README)." if sys.platform.startswith("linux") else "."))
        self.dev = devs[self.index]
        try:
            self.dev.set_configuration()
        except Exception:  # noqa: BLE001
            pass
        try:
            usb.util.claim_interface(self.dev, 0)
            self.dev.set_interface_altsetting(interface=0, alternate_setting=1)
        except usb.core.USBError as e:
            raise RuntimeError(f"Could not open the Helios (is another program using it? {e})") from e
        self._ctl(0x04, read=True)                 # firmware version (also wakes the interface)
        self._ctl(0x07, self.SDK_VERSION)          # announce SDK version
        self._ctl(0x02, 1)                         # shutter open
        if self._async_io:
            self._sender = _AsyncFrameSender(self._write_sync)
            self._sender.start()
        return 1

    def _ctl(self, cmd, arg=0, read=False):
        self.dev.write(self.EP_INT_OUT, bytes([cmd, arg]), 100)
        if read:
            return bytes(self.dev.read(self.EP_INT_IN, 32, 100))
        return None

    def _ready(self):
        r = self._ctl(0x03, read=True)
        return len(r) >= 2 and r[0] == 0x83 and r[1] == 1

    def _write_sync(self, frame, timeout=1.0):
        data = pack_helios_frame(frame, self.pps, **self.flip)
        t0 = time.perf_counter()
        while not self._ready():
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError("Helios not ready (timeout)")
            time.sleep(0.0001)
        self.dev.write(self.EP_BULK_OUT, data, max((8 + len(data)) >> 5, 100))

    def send(self, frame, timeout=1.0):
        if self._sender is not None:
            self._sender.submit(frame)
        else:
            self._write_sync(frame, timeout=timeout)

    def close(self):
        if self._sender is not None:
            self._sender.stop()
            self._sender = None
        if self.dev is None:
            return
        try:
            self._write_sync(np.zeros((64, 5), np.float32), timeout=0.2)
            self._ctl(0x02, 0)                     # shutter closed
            self._ctl(0x01)                        # stop
        except Exception:  # noqa: BLE001
            pass
        try:
            import usb.util
            usb.util.dispose_resources(self.dev)
        except Exception:  # noqa: BLE001
            pass
        self.dev = None


class HeliosDAC:
    def __init__(self, lib_path=None, device=0, pps=30000, flip_x=False, flip_y=False, swap_xy=False, async_io=True):
        self.dev, self.pps = device, min(pps, HELIOS_MAX_PPS)
        self.flip = dict(flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)
        self.lib = self._load(lib_path)
        L = self.lib
        L.OpenDevices.restype = ctypes.c_int
        L.CloseDevices.restype = ctypes.c_int
        L.GetStatus.argtypes, L.GetStatus.restype = [ctypes.c_uint], ctypes.c_int
        L.Stop.argtypes, L.Stop.restype = [ctypes.c_uint], ctypes.c_int
        L.SetShutter.argtypes, L.SetShutter.restype = [ctypes.c_uint, ctypes.c_bool], ctypes.c_int
        L.WriteFrame.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint8,
                                 ctypes.POINTER(HeliosPoint), ctypes.c_uint]
        L.WriteFrame.restype = ctypes.c_int
        self.opened = False
        self._async_io = async_io
        self._sender: _AsyncFrameSender | None = None

    @staticmethod
    def _load(path):
        sysname = platform.system()
        name = {"Windows": "HeliosDacAPI.dll", "Darwin": "libHeliosDacAPI.dylib"}.get(sysname, "libHeliosDacAPI.so")
        here = os.path.dirname(os.path.abspath(__file__))
        cands = [path, os.environ.get("HELIOS_LIB"), os.path.join(os.getcwd(), name),
                 os.path.join(here, name), os.path.join(here, "..", name), ctypes.util.find_library("HeliosDacAPI"), name]
        for c in filter(None, cands):
            try:
                return ctypes.CDLL(c)
            except OSError:
                continue
        raise RuntimeError(
            f"Could not load {name}. Build it from the Helios SDK (github.com/Grix/helios_dac, folder sdk/) "
            "and put it next to this script, or set HELIOS_LIB=/path/to/the/library.")

    def open(self):
        n = self.lib.OpenDevices()
        if n <= 0 or self.dev >= n:
            raise RuntimeError(f"No Helios DAC found (OpenDevices returned {n}). On Linux you need udev rules or root.")
        self.opened = True
        self.lib.SetShutter(self.dev, True)
        if self._async_io:
            self._sender = _AsyncFrameSender(self._write_sync)
            self._sender.start()
        return n

    def _write_sync(self, frame, timeout=1.0):
        pts = to_helios_array(frame[:HELIOS_MAX_POINTS], **self.flip)
        buf = (HeliosPoint * len(pts)).from_buffer_copy(pts.tobytes())
        t0 = time.perf_counter()
        while True:
            st = self.lib.GetStatus(self.dev)
            if st == 1:
                break
            if st < 0:
                raise RuntimeError(f"Helios GetStatus error {st}")
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError("Helios not ready (timeout)")
            time.sleep(0.0001)
        r = self.lib.WriteFrame(self.dev, self.pps, 0, buf, len(pts))
        if r < 0:
            raise RuntimeError(f"Helios WriteFrame error {r}")

    def send(self, frame, timeout=1.0):
        """Queues the latest frame without blocking the generator thread."""
        if self._sender is not None:
            self._sender.submit(frame)
        else:
            self._write_sync(frame, timeout=timeout)

    def blank(self):
        try:
            if self.lib.GetStatus(self.dev) == 1:
                self._write_sync(np.zeros((64, 5), np.float32), timeout=0.2)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
        if self._sender is not None:
            self._sender.stop()
            self._sender = None
        if self.opened:
            self.blank()
            self.lib.SetShutter(self.dev, False)
            self.lib.Stop(self.dev)
            self.lib.CloseDevices()
            self.opened = False


class NullOutput:
    def __init__(self, pps=30000):
        self.pps, self._next = pps, 0.0

    def open(self):
        self._next = time.perf_counter()

    def send(self, frame):
        self._next += len(frame) / self.pps
        d = self._next - time.perf_counter()
        if d > 0:
            _precise_sleep(d)
        else:
            self._next = time.perf_counter()

    def close(self):
        pass


def draw_frame(ax, frame, collection=None):
    from matplotlib.collections import LineCollection
    p = frame[:, :2]
    lit = (frame[:-1, 2:5].max(1) > 0.01) & (frame[1:, 2:5].max(1) > 0.01)
    segs = np.stack([p[:-1], p[1:]], 1)[lit]
    cols = np.clip(frame[:-1, 2:5][lit] * 1.6, 0, 1)
    if collection is None:
        collection = LineCollection(segs, colors=cols, linewidths=1.6)
        ax.add_collection(collection)
    else:
        collection.set_segments(segs)
        collection.set_color(cols)
    return collection


def _style(ax):
    ax.set_facecolor("black")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


class SimDisplay(NullOutput):
    """High-FPS blitted matplotlib preview, paced like the real DAC."""

    def __init__(self, pps=30000, target_ui_fps=120.0):
        super().__init__(pps=pps)
        self._ui_interval = 1.0 / max(target_ui_fps, 1.0)
        self._last_ui_draw = 0.0
        self._bg = None

    def open(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor("black")
        _style(self.ax)
        self.lc = draw_frame(self.ax, np.zeros((3, 5)))
        self.lc.set_animated(True)
        self.fig.canvas.draw()
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self._last_ui_draw = time.perf_counter()
        super().open()

    def send(self, frame):
        now = time.perf_counter()
        if now - self._last_ui_draw >= self._ui_interval:
            self._last_ui_draw = now
            canvas = self.fig.canvas
            if self._bg is None:
                canvas.draw()
                self._bg = canvas.copy_from_bbox(self.ax.bbox)
            canvas.restore_region(self._bg)
            draw_frame(self.ax, frame, self.lc)
            self.ax.draw_artist(self.lc)
            canvas.blit(self.ax.bbox)
            canvas.flush_events()
        super().send(frame)

    def close(self):
        self.plt.close("all")


def save_png(frames, titles, path, cols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = (len(frames) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    fig.patch.set_facecolor("black")
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, fr, t in zip(np.atleast_1d(axs).ravel(), frames, titles):
        ax.axis("on"); _style(ax)
        draw_frame(ax, fr)
        ax.set_title(t, color="white", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80, facecolor="black")
    plt.close(fig)
