# Model weights

Trained weights are **not committed to this repository**. Git stores binary files poorly:
every revision of a checkpoint is kept in full, so a repository that tracks weights grows
without bound and becomes slow to clone. Weights are published as **release assets**
instead.

## Getting the weights

Download `bestV12CrackClass.pt` from the [Releases](../../releases) page and place it in
this folder, or point the application at it from wherever you saved it — the *Browse .pt*
button accepts any location.

## Loading from the command line

```bash
python src/PECCD_Detector_GUI.py --weights models/bestV12CrackClass.pt
```

## Publishing new weights

1. **Releases → Draft a new release**, or edit an existing one.
2. Attach the `.pt` file (the per-file limit is 2 GB; these checkpoints are a few
   megabytes).
3. Record the SHA-256 checksum in the release notes:

   ```bash
   sha256sum bestV12CrackClass.pt          # Linux / macOS
   Get-FileHash bestV12CrackClass.pt -Algorithm SHA256   # Windows PowerShell
   ```

4. State in the release notes which architecture the file contains and what it scores, so
   that a user can tell two checkpoints apart. For example:

   > `bestV12CrackClass.pt` — YOLOV12SDSDA, 75 epochs, 640 px, imgsz 640,
   > P 0.767 / R 0.668 / mAP50 0.719 on the PECCD validation split.

## Checking what a checkpoint contains

Before publishing, confirm that the file is the architecture you intend to release. The
following prints the architecture, the class names and the parameter count:

```python
import torch
from ultralytics import YOLO

ckpt = torch.load("bestV12CrackClass.pt", map_location="cpu", weights_only=False)
print("trained from :", ckpt.get("train_args", {}).get("model"))
print("ultralytics  :", ckpt.get("version"))
print("classes      :", ckpt["model"].names)

model = YOLO("bestV12CrackClass.pt")
print("parameters   :", sum(p.numel() for p in model.model.parameters()) / 1e6, "M")
print("DS-DA blocks :", sum(
    1 for m in model.model.modules() if type(m).__name__ == "DualAttentionDS"))
```

The last line matters for this project: a checkpoint of the proposed architecture must
report a non-zero number of `DualAttentionDS` blocks. A count of zero means the file is a
plain YOLO baseline, whatever its filename says. Parameter counts are also a quick
discriminator — roughly 2.6 M for YOLOv12n, 9.3 M for YOLOv12s.
