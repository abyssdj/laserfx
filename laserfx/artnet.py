"""Zero-dependency Art-Net (DMX512 over UDP :6454) receiver for live RGB & color-harmony override."""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass

ARTNET_PORT = 6454
ARTNET_HEADER = b"Art-Net\x00"
OP_POLL = 0x2000
OP_POLL_REPLY = 0x2100
OP_DMX = 0x5000


@dataclass
class ArtNetState:
    connected: bool = False
    last_packet_t: float = 0.0
    source_ip: str = ""
    universe: int = 0
    dimmer: float = 1.0
    rgb1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgb2: tuple[float, float, float] | None = None
    accent_hue: float | None = None
    harmony_override: str | None = None
    raw_channels: tuple[int, ...] = ()


def _build_poll_reply(local_ip: str, universe: int) -> bytes:
    """Builds a minimal valid 239-byte ArtPollReply packet so consoles auto-discover LaserFX."""
    pkt = bytearray(239)
    pkt[0:8] = ARTNET_HEADER
    struct.pack_into("<H", pkt, 8, OP_POLL_REPLY)
    try:
        pkt[10:14] = socket.inet_aton(local_ip)
    except OSError:
        pkt[10:14] = b"\x00\x00\x00\x00"
    struct.pack_into("<H", pkt, 14, ARTNET_PORT)
    pkt[16:18] = b"\x01\x00"                      # VersInfo
    pkt[18] = (universe >> 8) & 0x7F              # NetSwitch
    pkt[19] = (universe >> 4) & 0x0F              # SubSwitch
    pkt[26:44] = b"LaserFX RGB\x00".ljust(18, b"\x00")
    pkt[44:108] = b"LaserFX Aerial Controller (6ch RGB Override)\x00".ljust(64, b"\x00")
    pkt[173] = 1                                  # NumPorts = 1
    pkt[174] = 0x80                               # PortTypes[0]: Output from Art-Net
    pkt[182] = 0x80                               # GoodOutput[0]: Data transmitted
    pkt[190] = universe & 0x0F                    # SwOut[0]
    return bytes(pkt)


def _decode_harmony_channel(val: int) -> str | None:
    if val < 43:
        return None  # Auto / follow LaserFX setting
    if val < 86:
        return "complementary"
    if val < 128:
        return "split_complementary"
    if val < 171:
        return "rectangular"
    if val < 213:
        return "triadic"
    return "analogous"


class ArtNetReceiver:
    """Background UDP listener parsing Art-Net DMX512 frames into live RGB override values."""

    def __init__(self, timeout_sec: float = 2.5):
        self.timeout_sec = timeout_sec
        self.enabled: bool = False
        self.net: int = 0
        self.subnet: int = 0
        self.universe: int = 0
        self.start_channel: int = 1        # 1-indexed DMX address (1..507)
        self.dual_rgb_mode: bool = False   # False = 6ch [Dim, R, G, B, AccentHue, Harmony], True = 6ch [R1,G1,B1, R2,G2,B2]

        self._lock = threading.Lock()
        self._state = ArtNetState()
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.bind_error: str | None = None

    @property
    def target_port_address(self) -> int:
        return ((self.net & 0x7F) << 8) | ((self.subnet & 0x0F) << 4) | (self.universe & 0x0F)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self.bind_error = None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        sock.settimeout(0.25)
        try:
            sock.bind(("0.0.0.0", ARTNET_PORT))
        except OSError as exc:
            sock.close()
            self.bind_error = f"UDP {ARTNET_PORT} busy: {exc}"
            return
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_active_override(self) -> ArtNetState | None:
        if not self.enabled:
            return None
        with self._lock:
            st = self._state
            if st.last_packet_t <= 0.0:
                return None
            if (time.monotonic() - st.last_packet_t) > self.timeout_sec:
                st.connected = False
                return None
            return ArtNetState(
                connected=True,
                last_packet_t=st.last_packet_t,
                source_ip=st.source_ip,
                universe=st.universe,
                dimmer=st.dimmer,
                rgb1=st.rgb1,
                rgb2=st.rgb2,
                accent_hue=st.accent_hue,
                harmony_override=st.harmony_override,
                raw_channels=st.raw_channels,
            )

    def _loop(self):
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 10 or data[:8] != ARTNET_HEADER:
                continue

            opcode = struct.unpack_from("<H", data, 8)[0]
            if opcode == OP_POLL:
                try:
                    reply = _build_poll_reply("0.0.0.0", self.target_port_address)
                    self._sock.sendto(reply, (addr[0], ARTNET_PORT))
                except OSError:
                    pass
                continue

            if opcode != OP_DMX or len(data) < 18:
                continue

            sub_uni = data[14]
            net = data[15] & 0x7F
            port_addr = (net << 8) | sub_uni
            if port_addr != self.target_port_address:
                continue

            dmx_len = struct.unpack_from(">H", data, 16)[0]
            dmx = data[18 : 18 + min(dmx_len, len(data) - 18)]
            ch0 = max(0, min(self.start_channel - 1, 511))
            if ch0 >= len(dmx):
                continue

            def _ch(offset: int, default: int = 0) -> int:
                idx = ch0 + offset
                return dmx[idx] if idx < len(dmx) else default

            raw = tuple(_ch(i, 0) for i in range(6))

            if self.dual_rgb_mode:
                r1, g1, b1 = raw[0] / 255.0, raw[1] / 255.0, raw[2] / 255.0
                r2, g2, b2 = raw[3] / 255.0, raw[4] / 255.0, raw[5] / 255.0
                dim = 1.0
                rgb1 = (r1, g1, b1)
                rgb2 = (r2, g2, b2) if (raw[3] or raw[4] or raw[5]) else None
                accent_hue = None
                harmony_ov = None
            else:
                dim = raw[0] / 255.0
                rgb1 = (raw[1] / 255.0, raw[2] / 255.0, raw[3] / 255.0)
                rgb2 = None
                accent_hue = (raw[4] / 255.0) if raw[4] > 0 else None
                harmony_ov = _decode_harmony_channel(raw[5])

            with self._lock:
                self._state = ArtNetState(
                    connected=True,
                    last_packet_t=time.monotonic(),
                    source_ip=addr[0],
                    universe=port_addr,
                    dimmer=dim,
                    rgb1=rgb1,
                    rgb2=rgb2,
                    accent_hue=accent_hue,
                    harmony_override=harmony_ov,
                    raw_channels=raw,
                )