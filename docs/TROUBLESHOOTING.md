# Troubleshooting

Errors are grouped by who is likely to hit them: users of the packaged executable first,
then people running from source.

---

## Using the executable

### Windows blocks the application ("Windows protected your PC")

Click **More info → Run anyway**. The executable is not code-signed, so SmartScreen has
no publisher to verify. Code-signing certificates are commercial products and are not
usually purchased for academic software.

### The application will not start, or closes immediately

Run it from inside the extracted folder, not from within the ZIP file. The executable
depends on the `_internal` folder that sits beside it; if that folder is missing or was
moved, startup fails silently.

### "Failed to load the model"

Check that you selected the `.pt` file and not a different file type, and that the
download completed (the file should be several megabytes, not a few hundred bytes).

### Detection finds nothing

Lower the confidence threshold to 0.10–0.15 and raise the inference size to 960 or 1280,
then press RUN DETECTION again. Very large photographs are downscaled for analysis, and
hairline cracks are the first thing lost.

---

## Running from source

### `RuntimeError: operator torchvision::nms does not exist`

The most common installation problem. Ultralytics performs non-maximum suppression
through a **compiled** torchvision operator. If `torch` and `torchvision` come from
different builds — a CUDA `torch` beside a CPU-only `torchvision`, or two different
release versions — the operator is never registered. The model loads normally and the
failure appears only at the first inference, which makes it look like a bug in the
application.

Check what is installed:

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda); print(torchvision.__version__, torchvision.version.cuda)"
```

Both CUDA strings must be identical, or both `None` for CPU builds. The versions must
also be a released pair:

| torch | torchvision |
|---|---|
| 2.5.1 | 0.20.1 |
| 2.6.0 | 0.21.0 |
| 2.7.0 | 0.22.0 |
| 2.8.0 | 0.23.0 |

Reinstall both together, from one index, in a single command — that is what stops the
mismatch reappearing:

```bash
pip uninstall -y torch torchvision torchaudio

# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# or CPU only
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

PECCD-Detect probes this operator when the model is loaded. If it is missing, the
application warns you and activates a pure-PyTorch replacement so that detection still
works — numerically equivalent, marginally slower. Repair the installation before
reporting any inference timings.

### `AttributeError: Can't get attribute 'DualAttentionDS' on <module '__main__'>`

The checkpoint contains custom attention modules that were declared in the training
script, so PyTorch stored a reference to `__main__.DualAttentionDS`. Any program loading
that file must be able to resolve the class.

Section 1 of `src/PECCD_Detector_GUI.py` re-declares `DualAttention` and
`DualAttentionDS` exactly as they were declared during training and registers them in
`__main__`, which resolves this. If you produce a checkpoint whose custom blocks differ,
copy those class definitions into the same section.

### `_pickle.UnpicklingError: Weights only load failed`

PyTorch 2.6 and later refuse to unpickle arbitrary Python objects by default. The
application whitelists its own attention classes and retries with `weights_only=False`
if the safe load fails, so this should not surface. If it does, the checkpoint contains a
class the application does not know about — add it to Section 1.

### `ModuleNotFoundError: No module named 'tkinter'`

Tkinter is not a pip package.

```bash
sudo apt-get install python3-tk      # Debian / Ubuntu
sudo dnf install python3-tkinter     # Fedora
```

On Windows and macOS, reinstall Python with the Tcl/Tk option enabled.

### The interface freezes during analysis

It should not: inference runs on a worker thread. If it does freeze, the image is
probably extremely large and is being decoded rather than analysed. Check the status bar
at the bottom, which reports the resolution when an image is opened.

### CUDA out of memory

Lower the inference size to 512, or force processor execution by uninstalling the CUDA
build of torch in favour of the CPU build. The models in this project are small, so CPU
inference remains practical.

---

## Reporting a problem

Open an issue including:

* the full text of the error dialogue,
* the model information line from the top right of the window,
* the output of the `torch` / `torchvision` version command above,
* your operating system, and whether you used the executable or ran from source.
