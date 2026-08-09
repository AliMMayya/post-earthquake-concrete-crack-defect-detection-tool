# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for PECCD-Detect.

Build (from the repository root, on Windows):

    pyinstaller build/PECCD_Detector.spec --noconfirm

The result is dist/PECCD-Detect/, containing PECCD-Detect.exe next to an
_internal folder. Distribute the whole folder as a ZIP: the executable will not
run without _internal beside it.

One-folder mode is used deliberately in preference to --onefile. A one-file
build unpacks several hundred megabytes of PyTorch libraries into a temporary
directory on every launch, which adds 20 to 60 seconds of startup time and is
frequently quarantined by antivirus software.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

# --------------------------------------------------------------------------- #
#  Paths                                                                       #
# --------------------------------------------------------------------------- #
# SPECPATH is provided by PyInstaller and points at the directory of this file.
ROOT = Path(SPECPATH).parent
SRC = ROOT / "src" / "PECCD_Detector_GUI.py"
ICON = ROOT / "docs" / "assets" / "pecdd_icon.ico"

# --------------------------------------------------------------------------- #
#  Dependency collection                                                       #
# --------------------------------------------------------------------------- #
datas = []
binaries = []
hiddenimports = []

# Ultralytics ships YAML configuration files that are read at runtime and are
# not discovered by static analysis, so the package must be collected whole.
for package in ("ultralytics",):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# PyInstaller has built-in hooks for torch and torchvision; invoking
# collect_all on them as well produces a far larger build and occasionally
# duplicates DLLs. Only the pieces the hooks miss are named explicitly.
hiddenimports += [
    "torchvision.ops",
    "torchvision.transforms",
    "torchvision.models",
    "PIL._tkinter_finder",
    "scipy._lib.array_api_compat.numpy.fft",
    "scipy.special._cdflib",
]

# matplotlib is imported by ultralytics for its plotting utilities, which the
# application never calls, but the import happens at module level.
datas += collect_data_files("matplotlib", includes=["mpl-data/matplotlibrc"])

# --------------------------------------------------------------------------- #
#  Excluded modules                                                            #
# --------------------------------------------------------------------------- #
# Training-time and notebook dependencies that inflate the build without being
# reachable from the inference path.
excludes = [
    "tensorboard", "tensorflow", "wandb", "clearml", "comet_ml", "dvclive",
    "IPython", "jupyter", "notebook", "ipykernel",
    "pytest", "sphinx", "setuptools._distutils",
    "torch.distributed.elastic", "torch.testing",
]

# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #
a = Analysis(
    [str(SRC)],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PECCD-Detect",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX compression of torch DLLs corrupts them
    console=False,             # windowed application, no terminal
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="PECCD-Detect",
)
