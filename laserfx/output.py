"""Outputs: Helios DAC (USB, via the official HeliosDacAPI shared library), matplotlib simulator, null."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import sys
import time

import numpy as np

HELIOS_MAX_POINTS = 0x1000
HELIOS_MAX_PPS = 65535
_PT = np.dtype([("x", "<u2"), ("y", "<u2"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("i", "u1")])


class HeliosPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint16), ("y", ctypes.c_uint16),
                ("r", ctypes.c_uint8), ("g", ctypes.c_uint8), ("b", ctypes.c_uint8), ("i", ctypes.c_uint8)]


def to_helios_array(frame, flip_x=False, flip_y=False, swap_xy=False):
    xy = frame[:, :2].copy()
    if swap_xy:
        xy = xy[:, ::-1]
    if flip_x:
        xy[:, 0] = -xy[:, 0]
    if flip_y:
        xy[:, 1] = -xy[:, 1]
    a = np.zeros(len(frame), _PT)
    a["x"] = np.clip((xy[:, 0] + 1) * 0.5 * 4095, 0, 4095)
    a["y"] = np.clip((xy[:, 1] + 1) * 0.5 * 4095, 0, 4095)
    for k, n in enumerate("rgb"):
        a[n] = np.clip(frame[:, 2 + k] * 255, 0, 255)
    a["i"] = 255
    return a


def pack_helios_frame(frame, pps, flip_x=False, flip_y=False, swap_xy=False):
    """Helios bulk packet: 7 bytes/point (12-bit x,y packed, r,g,b,i) + 5-byte footer (pps, count, flags)."""
    a = to_helios_array(frame[:HELIOS_MAX_POINTS], flip_x, flip_y, swap_xy)
    n = len(a)
    if n >= 45 and (n - 45) % 64 == 0:   # firmware cannot receive transfers of these sizes
        a, n = a[:-1], n - 1
    x, y = a["x"].astype(np.uint16), a["y"].astype(np.uint16)
    b = np.empty((n, 7), np.uint8)
    b[:, 0] = x >> 4
    b[:, 1] = (((x & 0xF) << 4) | (y >> 8)).astype(np.uint8)
    b[:, 2] = y & 0xFF
    b[:, 3], b[:, 4], b[:, 5], b[:, 6] = a["r"], a["g"], a["b"], a["i"]
    pps = int(min(max(pps, 7), HELIOS_MAX_PPS))
    return b.tobytes() + bytes([pps & 0xFF, pps >> 8, n & 0xFF, n >> 8, 0])


class HeliosUSB:
    """Native Helios USB driver (pyusb + libusb). No vendor DLL needed."""
    VID, PID = 0x1209, 0xE500
    EP_BULK_OUT, EP_INT_OUT, EP_INT_IN = 0x02, 0x06, 0x83
    SDK_VERSION = 6

    def __init__(self, index=0, pps=30000, flip_x=False, flip_y=False, swap_xy=False):
        self.index, self.pps = index, pps
        self.flip = dict(flip_x=flip_x, flip_y=flip_y, swap_xy=swap_xy)
        self.dev = None

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
        except Exception:  # noqa: BLE001  (already configured / not supported on WinUSB)
            pass
        try:
            usb.util.claim_interface(self.dev, 0)
            self.dev.set_interface_altsetting(interface=0, alternate_setting=1)
        except usb.core.USBError as e:
            raise RuntimeError(f"Could not open the Helios (is another program using it? {e})") from e
        self._ctl(0x04, read=True)                 # firmware version (also wakes the interface)
        self._ctl(0x07, self.SDK_VERSION)          # announce SDK version
        self._ctl(0x02, 1)                         # shutter open
        return 1

    def _ctl(self, cmd, arg=0, read=False):
        self.dev.write(self.EP_INT_OUT, bytes([cmd, arg]), 100)
        if read:
            return bytes(self.dev.read(self.EP_INT_IN, 32, 100))
        return None

    def _ready(self):
        r = self._ctl(0x03, read=True)
        return len(r) >= 2 and r[0] == 0x83 and r[1] == 1

    def send(self, frame, timeout=1.0):
        t0 = time.perf_counter()
        while not self._ready():
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError("Helios not ready (timeout)")
            time.sleep(0.0003)
        data = pack_helios_frame(frame, self.pps, **self.flip)
        self.dev.write(self.EP_BULK_OUT, data, max((8 + len(data)) >> 5, 100))

    def close(self):
        if self.dev is None:
            return
        try:
            self.send(np.zeros((64, 5), np.float32), timeout=0.2)
            self._ctl(0x02, 0)                      # shutter closed
            self._ctl(0x01)                         # stop
        except Exception:  # noqa: BLE001
            pass
        try:
            import usb.util
            usb.util.dispose_resources(self.dev)
        except Exception:  # noqa: BLE001
            pass
        self.dev = None


class HeliosDAC:
    def __init__(self, lib_path=None, device=0, pps=30000, flip_x=False, flip_y=False, swap_xy=False):
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
        return n

    def send(self, frame, timeout=1.0):
        """Blocks until the DAC has finished the previous frame, then queues this one."""
        pts = to_helios_array(frame[:HELIOS_MAX_POINTS], **self.flip)
        t0 = time.perf_counter()
        while True:
            st = self.lib.GetStatus(self.dev)
            if st == 1:
                break
            if st < 0:
                raise RuntimeError(f"Helios GetStatus error {st}")
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError("Helios not ready (timeout)")
            time.sleep(0.0003)
        buf = (HeliosPoint * len(pts)).from_buffer_copy(pts.tobytes())
        r = self.lib.WriteFrame(self.dev, self.pps, 0, buf, len(pts))
        if r < 0:
            raise RuntimeError(f"Helios WriteFrame error {r}")

    def blank(self):
        try:
            if self.lib.GetStatus(self.dev) == 1:
                self.send(np.zeros((64, 5), np.float32), timeout=0.2)
        except Exception:  # noqa: BLE001
            pass

    def close(self):
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
            time.sleep(d)
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
    """Live matplotlib preview, paced like the real DAC."""

    def open(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor("black")
        _style(self.ax)
        self.lc = draw_frame(self.ax, np.zeros((3, 5)))
        super().open()

    def send(self, frame):
        draw_frame(self.ax, frame, self.lc)
        self.plt.pause(0.001)
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
