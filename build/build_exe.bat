@echo off
REM ==========================================================================
REM  PECCD-Detect - Windows executable build script
REM
REM  Run this from the repository root:
REM       build\build_exe.bat
REM
REM  It creates an isolated environment containing only what the application
REM  needs, installs the CPU build of PyTorch to keep the distributable small,
REM  and runs PyInstaller. The result is dist\PECCD-Detect\.
REM ==========================================================================

setlocal

echo.
echo ========================================================
echo   PECCD-Detect - building the Windows executable
echo ========================================================
echo.

REM ---- 1. clean previous builds ------------------------------------------
if exist dist\PECCD-Detect rmdir /s /q dist\PECCD-Detect
if exist build\pyinstaller  rmdir /s /q build\pyinstaller

REM ---- 2. isolated build environment -------------------------------------
REM Building inside a clean environment is what keeps the executable small.
REM A general-purpose environment drags in every package it has ever held.
if not exist .buildenv (
    echo Creating the build environment...
    python -m venv .buildenv
)
call .buildenv\Scripts\activate.bat

REM ---- 3. dependencies ----------------------------------------------------
REM The CPU build of PyTorch is used deliberately: the CUDA build adds roughly
REM 2 GB of libraries. Users with a graphics card can still install the CUDA
REM build and run from source. If you want a GPU-enabled executable, replace
REM the index URL below with the CUDA one and expect a much larger ZIP.
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics pillow opencv-python-headless
python -m pip install pyinstaller

REM ---- 4. build -----------------------------------------------------------
echo.
echo Running PyInstaller. This takes several minutes.
echo.
pyinstaller build\PECCD_Detector.spec --noconfirm --workpath build\pyinstaller --distpath dist

REM ---- 5. report ----------------------------------------------------------
if exist dist\PECCD-Detect\PECCD-Detect.exe (
    echo.
    echo ========================================================
    echo   BUILD SUCCEEDED
    echo   Output: dist\PECCD-Detect\
    echo.
    echo   Test it now, before packaging:
    echo       dist\PECCD-Detect\PECCD-Detect.exe
    echo.
    echo   Then compress the whole PECCD-Detect folder as
    echo   PECCD-Detect-win64.zip and attach it to a GitHub Release.
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   BUILD FAILED - see the PyInstaller output above.
    echo   A missing module is usually fixed by adding its name
    echo   to hiddenimports in build\PECCD_Detector.spec.
    echo ========================================================
)

echo.
pause
endlocal
