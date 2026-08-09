# Building the Windows executable

The executable must be built **on Windows**. PyInstaller does not cross-compile: it
bundles the interpreter and the compiled libraries of the machine it runs on, so a Linux
or macOS build produces a Linux or macOS application. If you only have access to Linux,
see "Building without a Windows machine" at the end.

---

## Quick route

From the repository root, in a Command Prompt:

```bat
build\build_exe.bat
```

The script creates an isolated environment, installs the dependencies, runs PyInstaller
and reports the result. Expect five to fifteen minutes, most of it downloading PyTorch.

Output: `dist\PECCD-Detect\`, containing `PECCD-Detect.exe` beside an `_internal` folder.

---

## Manual route

```bat
python -m venv .buildenv
.buildenv\Scripts\activate

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics pillow opencv-python-headless pyinstaller

pyinstaller build\PECCD_Detector.spec --noconfirm
```

---

## Decisions built into the specification

**One folder, not one file.** `--onefile` produces a single `.exe`, which looks tidier
but unpacks several hundred megabytes of PyTorch libraries into a temporary directory on
every launch. That adds 20–60 seconds to startup and is frequently flagged by antivirus
software. One-folder mode starts in two or three seconds. Distribute the folder as a ZIP.

**CPU build of PyTorch.** The CUDA build adds roughly 2 GB of libraries to the
distributable. A CPU-only executable is around 700 MB before compression, 300–400 MB
zipped, and analyses a photograph in one to three seconds — perfectly usable for review
work. Researchers who need GPU speed can run from source. If you do want a GPU
executable, change the index URL in `build_exe.bat` to the CUDA one and warn users about
the download size.

**No UPX compression.** UPX corrupts several PyTorch DLLs, producing an executable that
builds cleanly and then crashes at import. It is disabled in the spec; leave it disabled.

**Isolated build environment.** PyInstaller bundles what it finds. Building inside a
general-purpose environment that has accumulated packages over months produces an
executable several times larger than necessary.

---

## Verifying the build

Test on the build machine first, then — importantly — on a **second computer that has
never had Python installed**. This is the only reliable way to catch a missing bundled
dependency, because the build machine can silently satisfy an import from its own
installation.

Check that:

1. the application starts and the window is drawn correctly;
2. `Browse .pt` → `Load model` succeeds, and the header shows six categories;
3. detection on a test photograph produces boxes;
4. `Save annotated image` and `Export session report (CSV)` write files;
5. `Process whole folder and export` completes on a small folder.

If step 1 fails silently, rebuild with `console=True` in the spec so the traceback is
visible, then set it back to `False`.

---

## Common build problems

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError` at runtime | The module was not detected statically. Add its name to `hiddenimports` in `PECCD_Detector.spec` and rebuild. |
| Application starts then closes instantly | Set `console=True` in the spec to see the traceback. Usually a missing data file. |
| Ultralytics YAML not found | `collect_all("ultralytics")` was removed or failed. It is required. |
| Executable is 2 GB or more | The CUDA build of torch was installed. Reinstall the CPU build in a clean environment. |
| Antivirus quarantines the executable | Common with unsigned PyInstaller binaries. One-folder mode reduces it. Submit a false-positive report to the vendor if it persists. |

---

## Publishing a release

1. Compress the whole `dist\PECCD-Detect` folder as `PECCD-Detect-win64.zip`.
2. On GitHub: **Releases → Draft a new release**, tag `v1.0.0`.
3. Attach two files: `PECCD-Detect-win64.zip` and `bestV12CrackClass.pt`.
4. Record the SHA-256 checksums in the release notes so users can verify their downloads:

   ```powershell
   Get-FileHash PECCD-Detect-win64.zip -Algorithm SHA256
   Get-FileHash bestV12CrackClass.pt   -Algorithm SHA256
   ```

Release assets have a 2 GB per-file limit, well above what is needed here. Do not commit
the ZIP into the repository itself — Git handles large binaries poorly and the repository
would become slow to clone.

---

## Building without a Windows machine

* **A Windows virtual machine.** Microsoft distributes free, time-limited Windows
  development images for VirtualBox, VMware and Hyper-V.
* **GitHub Actions.** A `windows-latest` runner builds the executable on every tagged
  commit and attaches it to the release automatically. This is the most maintainable
  option if you expect to publish updates; a workflow file of about thirty lines is
  enough.
* **Wine is not a solution.** Building Windows binaries under Wine works for simple
  scripts but fails on PyTorch's native libraries.
