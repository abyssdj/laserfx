# Get LaserFX.exe without installing Python (about 10 minutes, one time)

1. Create a free account at github.com and click **New repository** (name it `laserfx`, Private is fine).
2. On the new repo page click **uploading an existing file**, unzip `laserfx.zip` and drag in EVERYTHING inside the
   `laserfx` folder (including the hidden `.github` folder: enable "show hidden files" in Explorer, or drag the whole
   folder contents with Ctrl+A). Click **Commit changes**.
   Check that `.github/workflows/build.yml` appears in the repo.
3. Open the repo's **Actions** tab. The *Build LaserFX* run starts automatically (if it asks, click
   "I understand my workflows, go ahead and enable them"). Wait ~5 minutes for green ticks.
4. Click the finished run, scroll to **Artifacts**, download **LaserFX-Windows**, unzip: `LaserFX.exe`.
5. Copy `LaserFX.exe` to any Windows PC and double-click. Nothing else to install.

Notes
* Windows SmartScreen may warn about an unsigned app: **More info -> Run anyway**.
* If the Helios is not detected, Windows may need the WinUSB driver bound to it (Zadig tool, once), and no other laser
  software may be running (only one program can use the DAC at a time).
* If a build step fails (red cross), open it and send me the error text; I'll fix it.
