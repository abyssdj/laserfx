"""Build ONE standalone executable for the OS you run this on:  python build.py
Needs (build machine only): python 3.9+ with tkinter, then
    pip install numpy sounddevice pyusb libusb-package pyinstaller
The result in dist/ runs with nothing installed (Python, numpy, libusb and PortAudio are bundled).
PyInstaller cannot cross-compile: build on Windows for .exe, on macOS for .app, on Linux for a Linux binary
(or let the GitHub Action in .github/workflows/build.yml do all three)."""
import subprocess, sys

args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", "LaserFX",
        "--collect-all", "libusb_package", "--collect-all", "_sounddevice_data",
        "--hidden-import", "usb.backend.libusb1", "--hidden-import", "sounddevice",
        "--exclude-module", "matplotlib", "--exclude-module", "scipy", "--exclude-module", "pandas",
        "laserfx_app.py"]
sys.exit(subprocess.call(args))
