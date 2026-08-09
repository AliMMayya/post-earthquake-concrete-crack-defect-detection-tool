#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 PECCD-Detect  -  Post-Earthquake Concrete Crack Detection Desktop Application
===============================================================================

 A standalone, offline graphical user interface (GUI) for automated multi-class
 concrete defect detection using the YOLOV12S-DSDA model trained on the
 Post-Earthquake Concrete Crack Dataset (PECCD).

 Six defect categories are detected and localised:
     Scaling | Spalling | MultiBranch | SimpleCrack | DeepCrack | Hole

 -----------------------------------------------------------------------------
 FEATURES
 -----------------------------------------------------------------------------
   * Load any Ultralytics-compatible .pt checkpoint (baseline or DS-DA model)
   * Single-image mode and batch (folder) mode with image navigation
   * Live control of confidence threshold, NMS IoU, inference size, max det.
   * Per-class colour-coded bounding boxes with category name + confidence
   * Per-class visibility toggles (show/hide a defect category on the fly)
   * Quantitative panel: instance counts, mean confidence, and the fraction of
     the surface area occupied by each defect category (damage-extent proxy)
   * Export: annotated image (PNG), YOLO-format label file (.txt),
     per-image CSV report, and a cumulative batch/session CSV report
   * Runs on CPU or GPU; inference executes on a worker thread so the
     interface never freezes

 -----------------------------------------------------------------------------
 REQUIREMENTS
 -----------------------------------------------------------------------------
   python >= 3.9
   ultralytics >= 8.3.0
   torch >= 2.0
   pillow >= 9.0
   tkinter  (bundled with most Python distributions; on Debian/Ubuntu:
             sudo apt-get install python3-tk)

 -----------------------------------------------------------------------------
 USAGE
 -----------------------------------------------------------------------------
   python PECCD_Detector_GUI.py
   python PECCD_Detector_GUI.py --weights bestV12CrackClass.pt
   python PECCD_Detector_GUI.py --weights best.pt --image sample.jpg

 -----------------------------------------------------------------------------
 LICENCE / CITATION
 -----------------------------------------------------------------------------
 If you use this tool, please cite the accompanying paper describing the PECCD
 dataset and the YOLOV12SDSDA architecture.
===============================================================================
"""

import argparse
import csv
import os
import sys
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

# Ultralytics may try to pip-install missing requirements at runtime. That
# cannot work inside a frozen executable and would surface as an obscure
# error, so the behaviour is disabled before the library is imported.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# --------------------------------------------------------------------------- #
#  Tkinter                                                                     #
# --------------------------------------------------------------------------- #
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "\n[FATAL] Tkinter is not available in this Python installation.\n"
        "        Debian/Ubuntu : sudo apt-get install python3-tk\n"
        "        Fedora        : sudo dnf install python3-tkinter\n"
        "        macOS/Windows : reinstall Python with the Tcl/Tk option.\n\n"
    )
    raise

from PIL import Image, ImageDraw, ImageFont, ImageTk


# =========================================================================== #
#  SECTION 1.  CUSTOM MODULES REQUIRED TO UNPICKLE THE YOLOV12SDSDA CHECKPOINT
# =========================================================================== #
#
#  The proposed model is built by *injecting* a Depthwise-Separable Dual
#  Attention (DS-DA) block into the YOLOV12S backbone at training time. Because
#  the injection is performed in the training script, PyTorch serialises those
#  blocks with a reference to the module in which they were declared
#  (typically "__main__"). Any program that later loads the checkpoint must
#  therefore be able to resolve `__main__.DualAttentionDS`, otherwise
#  unpickling fails with:
#
#      AttributeError: Can't get attribute 'DualAttentionDS' on <module
#      '__main__'>
#
#  The class definitions below are byte-identical to the ones used during
#  training, which makes this application self-contained: it can load both the
#  original YOLO baselines and the modified DS-DA checkpoints.
#
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn

    class DualAttention(nn.Module):
        """Channel attention followed by standard 7x7 spatial attention."""

        def __init__(self, in_channels, reduction=8):
            super().__init__()
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.ca = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
                nn.Sigmoid(),
            )
            self.sa = nn.Sequential(
                nn.Conv2d(2, 1, 7, padding=3, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            c = self.ca(self.avg(x))
            a = torch.cat([x.mean(1, True), x.max(1, True)[0]], dim=1)
            s = self.sa(a)
            return x * c * s

    class DualAttentionDS(nn.Module):
        """Depthwise-separable variant of the dual-attention block (DS-DA)."""

        def __init__(self, in_channels, reduction=8):
            super().__init__()
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.ca = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
                nn.Sigmoid(),
            )
            self.depthwise = nn.Conv2d(2, 2, kernel_size=7, padding=3,
                                       groups=2, bias=False)
            self.pointwise = nn.Conv2d(2, 1, kernel_size=1, bias=False)
            self.sa = nn.Sequential(self.depthwise, self.pointwise, nn.Sigmoid())

        def forward(self, x):
            c = self.ca(self.avg(x))
            a = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]],
                          dim=1)
            s = self.sa(a)
            return x * c * s

    # Make the classes resolvable under every name the checkpoint may use.
    _main = sys.modules.get("__main__")
    if _main is not None:
        setattr(_main, "DualAttention", DualAttention)
        setattr(_main, "DualAttentionDS", DualAttentionDS)

    # PyTorch >= 2.6 defaults to weights_only=True; whitelist our classes so
    # that a safe load still succeeds.
    try:
        torch.serialization.add_safe_globals([DualAttention, DualAttentionDS])
    except Exception:
        pass

    TORCH_AVAILABLE = True
except Exception:  # torch missing -> reported later in the GUI
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
#  Torchvision NMS safety net                                                   #
# --------------------------------------------------------------------------- #
#
#  Ultralytics performs non-maximum suppression through `torchvision.ops.nms`,
#  which is a compiled C++ operator. When torch and torchvision are installed
#  from different builds (a very common situation, e.g. a CUDA torch next to a
#  CPU-only torchvision), that operator is not registered and inference fails
#  with:
#
#      RuntimeError: operator torchvision::nms does not exist
#
#  The correct remedy is to reinstall a matching torch / torchvision pair (see
#  README, "Troubleshooting"). The pure-PyTorch implementation below is a
#  fallback so that the application remains usable in the meantime; it is
#  numerically equivalent to the compiled operator and marginally slower.
#
# --------------------------------------------------------------------------- #

NMS_FALLBACK_ACTIVE = False


def _python_nms(boxes, scores, iou_threshold):
    """Greedy non-maximum suppression written in plain PyTorch.

    boxes  : (N, 4) tensor in (x1, y1, x2, y2) format
    scores : (N,) tensor
    returns: LongTensor of kept indices, sorted by descending score
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[rest] - inter
        iou = inter / union.clamp(min=1e-7)

        order = rest[iou <= iou_threshold]

    return torch.stack(keep) if keep else torch.empty(
        (0,), dtype=torch.long, device=boxes.device)


def _python_batched_nms(boxes, scores, idxs, iou_threshold):
    """Class-aware NMS: offset each class into its own coordinate region."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    max_coord = boxes.max()
    offsets = idxs.to(boxes) * (max_coord + torch.tensor(1).to(boxes))
    return _python_nms(boxes + offsets[:, None], scores, iou_threshold)


def check_torchvision_nms():
    """Verify that the compiled NMS operator works; patch it if it does not.

    Returns (ok, message). `ok` is False when the fallback had to be installed.
    """
    global NMS_FALLBACK_ACTIVE
    if not TORCH_AVAILABLE:
        return True, ""
    try:
        import torchvision
    except Exception as exc:
        return False, f"torchvision could not be imported ({exc})."

    probe_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]])
    probe_scores = torch.tensor([0.9, 0.8])
    try:
        torchvision.ops.nms(probe_boxes, probe_scores, 0.5)
        return True, ""
    except Exception as exc:
        torchvision.ops.nms = _python_nms
        torchvision.ops.batched_nms = _python_batched_nms
        NMS_FALLBACK_ACTIVE = True
        try:
            tv_version = torchvision.__version__
        except Exception:
            tv_version = "unknown"
        return False, (
            f"The compiled torchvision NMS operator is unavailable "
            f"(torch {torch.__version__} / torchvision {tv_version}): {exc}\n\n"
            f"A pure-PyTorch replacement has been activated so that detection "
            f"still works. For full speed, reinstall a matching torch and "
            f"torchvision pair - see the Troubleshooting section of the README."
        )


# =========================================================================== #
#  SECTION 2.  CONFIGURATION                                                   #
# =========================================================================== #

APP_NAME = "PECCD-Detect"
APP_SUBTITLE = "Post-Earthquake Concrete Defect Detection - YOLOV12SDSDA"
APP_VERSION = "1.0"

# Class order must match the `names` field of the trained checkpoint:
#   0 Scaling | 1 Spalling | 2 MultiBranch | 3 SimpleCrack | 4 DeepCrack | 5 Hole
DEFAULT_CLASS_NAMES = ["Scaling", "Spalling", "MultiBranch",
                       "SimpleCrack", "DeepCrack", "Hole"]

# Human-readable labels shown in the interface and in the exported reports.
PRETTY_NAMES = {
    "Scaling": "Scaling",
    "Spalling": "Spalling",
    "MultiBranch": "Multi-branched crack",
    "SimpleCrack": "Simple crack",
    "DeepCrack": "Deep crack",
    "Hole": "Hole",
}

# One clearly separable colour per defect category (RGB).
CLASS_COLORS = {
    "Scaling":     (255, 140,   0),   # orange
    "Spalling":    (230,  57,  70),   # red
    "MultiBranch": (157,  78, 221),   # violet
    "SimpleCrack": ( 46, 204, 113),   # green
    "DeepCrack":   ( 58, 134, 255),   # blue
    "Hole":        (255, 214,  10),   # yellow
}
FALLBACK_PALETTE = [
    (255, 140, 0), (230, 57, 70), (157, 78, 221), (46, 204, 113),
    (58, 134, 255), (255, 214, 10), (0, 191, 179), (247, 37, 133),
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Interface palette
BG_DARK = "#1e2229"
BG_PANEL = "#272c35"
BG_CANVAS = "#14171c"
FG_TEXT = "#e8eaed"
FG_MUTED = "#9aa3ad"
ACCENT = "#3a86ff"


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def text_color_for(rgb):
    """Return black or white depending on the luminance of the background."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def load_font(size):
    """Load a TrueType font, falling back to the PIL bitmap font."""
    candidates = [
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# =========================================================================== #
#  SECTION 3.  DETECTION BACK-END                                              #
# =========================================================================== #

class CrackDetector:
    """Thin wrapper around the Ultralytics YOLO predictor."""

    def __init__(self):
        self.model = None
        self.weights_path = None
        self.class_names = list(DEFAULT_CLASS_NAMES)
        self.device = "cpu"
        self.model_info = {}

    # ------------------------------------------------------------------ #
    def load(self, weights_path, device="auto"):
        """Load a checkpoint. Returns a dictionary describing the model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Run:  pip install torch ultralytics")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "Ultralytics is not installed. Run:  pip install ultralytics")

        # Detect a broken torch/torchvision pairing before the first inference,
        # so that the user gets an explanatory message instead of a traceback.
        nms_ok, nms_message = check_torchvision_nms()

        weights_path = str(weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

        try:
            model = YOLO(weights_path)
        except Exception as first_error:
            # PyTorch >= 2.6 refuses to unpickle arbitrary classes unless
            # weights_only=False is requested explicitly. Retry once with the
            # restriction lifted (safe here: the user selected the file).
            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return original_load(*args, **kwargs)

            torch.load = patched_load
            try:
                model = YOLO(weights_path)
            except Exception:
                raise first_error
            finally:
                torch.load = original_load

        # Device selection
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            model.to(device)
        except Exception:
            device = "cpu"
            model.to(device)

        names = model.names
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]
        self.class_names = list(names) if names else list(DEFAULT_CLASS_NAMES)

        n_params = 0
        try:
            n_params = sum(p.numel() for p in model.model.parameters())
        except Exception:
            pass

        self.model = model
        self.weights_path = weights_path
        self.device = device
        self.model_info = {
            "file": os.path.basename(weights_path),
            "classes": len(self.class_names),
            "params_M": round(n_params / 1e6, 2) if n_params else None,
            "device": device,
            "torch": torch.__version__,
            "nms_ok": nms_ok,
            "nms_message": nms_message,
        }
        return self.model_info

    # ------------------------------------------------------------------ #
    def predict(self, image_path, conf=0.25, iou=0.45, imgsz=640, max_det=300):
        """Run inference on one image.

        Returns (detections, elapsed_ms) where each detection is a dict with
        keys: class_id, class_name, confidence, x1, y1, x2, y2.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        t0 = time.time()
        results = self.model.predict(
            source=str(image_path),
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            max_det=int(max_det),
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        detections = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    name = (self.class_names[k]
                            if 0 <= k < len(self.class_names) else f"class_{k}")
                    detections.append({
                        "class_id": int(k),
                        "class_name": name,
                        "confidence": float(c),
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                    })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, elapsed_ms


# =========================================================================== #
#  SECTION 4.  ANNOTATION RENDERER                                             #
# =========================================================================== #

def color_for_class(name, class_id=0):
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    return FALLBACK_PALETTE[class_id % len(FALLBACK_PALETTE)]


def draw_detections(pil_image, detections, show_labels=True, show_conf=True,
                    line_scale=1.0, hidden_classes=None):
    """Return a copy of `pil_image` with colour-coded boxes and labels."""
    hidden_classes = hidden_classes or set()
    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    # Scale line width and font with image resolution so that annotations stay
    # legible on both 640 px thumbnails and 4000 px field photographs.
    base = max(img.width, img.height)
    line_w = max(2, int(round(base / 500.0 * line_scale)))
    font_size = max(12, int(round(base / 55.0 * line_scale)))
    font = load_font(font_size)

    visible = [d for d in detections if d["class_name"] not in hidden_classes]

    # ---- pass 1: bounding boxes ------------------------------------- #
    for det in visible:
        color = color_for_class(det["class_name"], det["class_id"])
        draw.rectangle([det["x1"], det["y1"], det["x2"], det["y2"]],
                       outline=color, width=line_w)

    if not show_labels:
        return img

    # ---- pass 2: labels on top, with simple collision avoidance ------ #
    def overlaps(a, b):
        return not (a[2] <= b[0] or b[2] <= a[0] or
                    a[3] <= b[1] or b[3] <= a[1])

    placed = []
    for det in visible:
        color = color_for_class(det["class_name"], det["class_id"])
        label = PRETTY_NAMES.get(det["class_name"], det["class_name"])
        if show_conf:
            label = f"{label} {det['confidence']:.2f}"

        try:
            tb = draw.textbbox((0, 0), label, font=font)
        except Exception:                      # very old Pillow versions
            tw, th = font.getsize(label)
            tb = (0, 0, tw, th)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        off_x, off_y = tb[0], tb[1]

        pad = max(3, line_w)
        tag_h = th + 2 * pad
        tag_w = tw + 2 * pad

        tx = max(0, min(det["x1"], img.width - tag_w))
        # Candidate vertical positions: above the box, just inside the top,
        # then progressively lower inside the box, then below it.
        candidates = [det["y1"] - tag_h, det["y1"]]
        candidates += [det["y1"] + k * tag_h for k in range(1, 4)]
        candidates.append(det["y2"])

        chosen = None
        for ty in candidates:
            ty = max(0, min(ty, img.height - tag_h))
            rect = (tx, ty, tx + tag_w, ty + tag_h)
            if not any(overlaps(rect, p) for p in placed):
                chosen = rect
                break
        if chosen is None:
            ty = max(0, min(det["y1"] - tag_h, img.height - tag_h))
            chosen = (tx, ty, tx + tag_w, ty + tag_h)
        placed.append(chosen)

        draw.rectangle(list(chosen), fill=color)
        draw.text((chosen[0] + pad - off_x, chosen[1] + pad - off_y), label,
                  fill=text_color_for(color), font=font)

    return img


def summarise(detections, image_w, image_h, hidden_classes=None):
    """Aggregate per-class statistics used by the quantitative panel."""
    hidden_classes = hidden_classes or set()
    stats = {}
    area_img = float(max(1, image_w * image_h))
    for det in detections:
        if det["class_name"] in hidden_classes:
            continue
        name = det["class_name"]
        entry = stats.setdefault(name, {"count": 0, "conf_sum": 0.0,
                                        "area": 0.0})
        entry["count"] += 1
        entry["conf_sum"] += det["confidence"]
        entry["area"] += max(0.0, det["x2"] - det["x1"]) * \
                         max(0.0, det["y2"] - det["y1"])
    for name, entry in stats.items():
        entry["mean_conf"] = entry["conf_sum"] / max(1, entry["count"])
        entry["area_pct"] = 100.0 * entry["area"] / area_img
    return stats


# =========================================================================== #
#  SECTION 5.  MAIN APPLICATION WINDOW                                         #
# =========================================================================== #

class PECCDApp(tk.Tk):

    def __init__(self, weights=None, image=None):
        super().__init__()
        self.title(f"{APP_NAME} v{APP_VERSION} - {APP_SUBTITLE}")
        self.geometry("1500x900")
        self.minsize(1150, 720)
        self.configure(bg=BG_DARK)

        self.detector = CrackDetector()

        # State ---------------------------------------------------------- #
        self.image_paths = []
        self.current_index = -1
        self.original_image = None      # PIL.Image of the current photograph
        self.annotated_image = None     # PIL.Image with boxes drawn
        self.detections = []
        self.last_elapsed_ms = 0.0
        self.display_photo = None       # keeps a reference for Tk
        self.session_rows = []          # cumulative report rows
        self.busy = False

        # Tk variables --------------------------------------------------- #
        self.var_weights = tk.StringVar(value=weights or "")
        self.var_conf = tk.DoubleVar(value=0.25)
        self.var_iou = tk.DoubleVar(value=0.45)
        self.var_imgsz = tk.StringVar(value="640")
        self.var_maxdet = tk.IntVar(value=300)
        self.var_labels = tk.BooleanVar(value=True)
        self.var_conf_txt = tk.BooleanVar(value=True)
        self.var_thickness = tk.DoubleVar(value=1.0)
        self.var_show_original = tk.BooleanVar(value=False)
        self.var_auto = tk.BooleanVar(value=True)
        self.var_status = tk.StringVar(value="Ready. Load a model to begin.")
        self.var_model_info = tk.StringVar(value="No model loaded")
        self.class_visibility = {}      # name -> BooleanVar

        self._build_style()
        self._build_layout()
        self._build_class_filter(DEFAULT_CLASS_NAMES)
        self.bind("<Left>", lambda e: self.prev_image())
        self.bind("<Right>", lambda e: self.next_image())
        self.bind("<Control-o>", lambda e: self.open_images())
        self.bind("<Control-s>", lambda e: self.save_annotated())

        if weights:
            self.after(300, lambda: self.load_model(weights))
        if image:
            self.after(900, lambda: self.set_image_list([image]))

    # ------------------------------------------------------------------ #
    #  Styling                                                            #
    # ------------------------------------------------------------------ #
    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=BG_PANEL, foreground=FG_TEXT,
                        fieldbackground=BG_DARK, bordercolor="#3a4049")
        style.configure("TFrame", background=BG_PANEL)
        style.configure("Dark.TFrame", background=BG_DARK)
        style.configure("TLabel", background=BG_PANEL, foreground=FG_TEXT)
        style.configure("Muted.TLabel", background=BG_PANEL, foreground=FG_MUTED)
        style.configure("Title.TLabel", background=BG_PANEL, foreground=FG_TEXT,
                        font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", background=BG_DARK, foreground=FG_TEXT,
                        font=("Segoe UI", 14, "bold"))
        style.configure("Sub.TLabel", background=BG_DARK, foreground=FG_MUTED)
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", padding=7,
                        font=("Segoe UI", 9, "bold"))
        style.map("Accent.TButton",
                  background=[("!disabled", ACCENT), ("disabled", "#39424f")],
                  foreground=[("!disabled", "#ffffff")])
        style.configure("TCheckbutton", background=BG_PANEL, foreground=FG_TEXT)
        style.configure("TLabelframe", background=BG_PANEL, foreground=FG_TEXT,
                        bordercolor="#3a4049")
        style.configure("TLabelframe.Label", background=BG_PANEL,
                        foreground=ACCENT, font=("Segoe UI", 9, "bold"))
        style.configure("Treeview", background=BG_DARK, fieldbackground=BG_DARK,
                        foreground=FG_TEXT, rowheight=22, borderwidth=0)
        style.configure("Treeview.Heading", background="#333a45",
                        foreground=FG_TEXT, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])
        style.configure("Horizontal.TScale", background=BG_PANEL)

    # ------------------------------------------------------------------ #
    #  Layout                                                             #
    # ------------------------------------------------------------------ #
    def _build_layout(self):
        # ---------------- header ---------------- #
        header = ttk.Frame(self, style="Dark.TFrame")
        header.pack(side="top", fill="x", padx=14, pady=(10, 6))
        ttk.Label(header, text=APP_NAME, style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="   " + APP_SUBTITLE,
                  style="Sub.TLabel").pack(side="left", pady=(5, 0))
        ttk.Label(header, textvariable=self.var_model_info,
                  style="Sub.TLabel").pack(side="right", pady=(5, 0))

        body = ttk.Frame(self, style="Dark.TFrame")
        body.pack(side="top", fill="both", expand=True, padx=10, pady=4)

        # ---------------- left control column ---------------- #
        left = ttk.Frame(body, width=310)
        left.pack(side="left", fill="y", padx=(4, 8), pady=4)
        left.pack_propagate(False)

        self._build_model_box(left)
        self._build_input_box(left)
        self._build_params_box(left)
        self.class_box = ttk.LabelFrame(left, text=" DEFECT CATEGORIES ")
        self.class_box.pack(fill="x", pady=(0, 8))
        self._build_display_box(left)

        # ---------------- centre canvas ---------------- #
        centre = ttk.Frame(body, style="Dark.TFrame")
        centre.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(centre, bg=BG_CANVAS, highlightthickness=0,
                                bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.refresh_canvas())
        self.canvas_hint = self.canvas.create_text(
            10, 10, text="", fill=FG_MUTED, font=("Segoe UI", 12),
            anchor="center")

        nav = ttk.Frame(centre)
        nav.pack(fill="x", pady=(6, 2))
        self.btn_prev = ttk.Button(nav, text="\u25c0  Previous",
                                   command=self.prev_image, state="disabled")
        self.btn_prev.pack(side="left", padx=4)
        self.lbl_counter = ttk.Label(nav, text="0 / 0", style="Muted.TLabel")
        self.lbl_counter.pack(side="left", padx=10)
        self.btn_next = ttk.Button(nav, text="Next  \u25b6",
                                   command=self.next_image, state="disabled")
        self.btn_next.pack(side="left", padx=4)

        ttk.Checkbutton(nav, text="Show original (no boxes)",
                        variable=self.var_show_original,
                        command=self.refresh_canvas).pack(side="left", padx=18)

        self.btn_detect = ttk.Button(nav, text="RUN DETECTION",
                                     style="Accent.TButton",
                                     command=self.run_detection,
                                     state="disabled")
        self.btn_detect.pack(side="right", padx=4)

        # ---------------- right results column ---------------- #
        right = ttk.Frame(body, width=390)
        right.pack(side="right", fill="y", padx=(8, 4), pady=4)
        right.pack_propagate(False)
        self._build_results_box(right)
        self._build_summary_box(right)
        self._build_export_box(right)

        # ---------------- status bar ---------------- #
        status = ttk.Frame(self, style="Dark.TFrame")
        status.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=140)
        self.progress.pack(side="right", padx=12, pady=6)
        ttk.Label(status, textvariable=self.var_status,
                  style="Sub.TLabel").pack(side="left", padx=14, pady=6)

    # ------------------------------------------------------------------ #
    def _build_model_box(self, parent):
        box = ttk.LabelFrame(parent, text=" 1.  TRAINED MODEL ")
        box.pack(fill="x", pady=(0, 8))

        entry = ttk.Entry(box, textvariable=self.var_weights)
        entry.pack(fill="x", padx=8, pady=(8, 4))

        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(row, text="Browse .pt",
                   command=self.browse_weights).pack(side="left")
        ttk.Button(row, text="Load model", style="Accent.TButton",
                   command=lambda: self.load_model(self.var_weights.get())
                   ).pack(side="right")

    def _build_input_box(self, parent):
        box = ttk.LabelFrame(parent, text=" 2.  INPUT IMAGES ")
        box.pack(fill="x", pady=(0, 8))
        ttk.Button(box, text="Select image(s)...",
                   command=self.open_images).pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(box, text="Select folder (batch)...",
                   command=self.open_folder).pack(fill="x", padx=8, pady=(0, 4))
        ttk.Checkbutton(box, text="Detect automatically on open",
                        variable=self.var_auto).pack(anchor="w", padx=8,
                                                     pady=(0, 8))

    def _build_params_box(self, parent):
        box = ttk.LabelFrame(parent, text=" 3.  DETECTION PARAMETERS ")
        box.pack(fill="x", pady=(0, 8))

        self.lbl_conf = ttk.Label(box, text="Confidence threshold : 0.25",
                                  style="Muted.TLabel")
        self.lbl_conf.pack(anchor="w", padx=8, pady=(8, 0))
        ttk.Scale(box, from_=0.05, to=0.95, variable=self.var_conf,
                  command=self._on_conf).pack(fill="x", padx=8)

        self.lbl_iou = ttk.Label(box, text="NMS IoU threshold : 0.45",
                                 style="Muted.TLabel")
        self.lbl_iou.pack(anchor="w", padx=8, pady=(8, 0))
        ttk.Scale(box, from_=0.10, to=0.90, variable=self.var_iou,
                  command=self._on_iou).pack(fill="x", padx=8)

        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(10, 8))
        ttk.Label(row, text="Inference size", style="Muted.TLabel").pack(
            side="left")
        ttk.Combobox(row, textvariable=self.var_imgsz, width=7,
                     state="readonly",
                     values=("512", "640", "768", "960", "1280")
                     ).pack(side="right")

    def _build_class_filter(self, names):
        for child in self.class_box.winfo_children():
            child.destroy()
        self.class_visibility = {}
        for i, name in enumerate(names):
            var = tk.BooleanVar(value=True)
            self.class_visibility[name] = var
            row = ttk.Frame(self.class_box)
            row.pack(fill="x", padx=8, pady=1)
            swatch = tk.Canvas(row, width=16, height=16, highlightthickness=0,
                               bg=BG_PANEL, bd=0)
            swatch.create_rectangle(1, 1, 15, 15,
                                    fill=rgb_to_hex(color_for_class(name, i)),
                                    outline="")
            swatch.pack(side="left", padx=(0, 8))
            ttk.Checkbutton(row, text=PRETTY_NAMES.get(name, name),
                            variable=var,
                            command=self.redraw_annotation).pack(side="left")
        pad = ttk.Frame(self.class_box)
        pad.pack(pady=3)

    def _build_display_box(self, parent):
        box = ttk.LabelFrame(parent, text=" 4.  DISPLAY OPTIONS ")
        box.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(box, text="Show class labels", variable=self.var_labels,
                        command=self.redraw_annotation).pack(anchor="w", padx=8,
                                                             pady=(8, 2))
        ttk.Checkbutton(box, text="Show confidence values",
                        variable=self.var_conf_txt,
                        command=self.redraw_annotation).pack(anchor="w", padx=8,
                                                             pady=(0, 4))
        ttk.Label(box, text="Annotation thickness", style="Muted.TLabel").pack(
            anchor="w", padx=8)
        ttk.Scale(box, from_=0.5, to=2.5, variable=self.var_thickness,
                  command=lambda e: self.redraw_annotation()).pack(
            fill="x", padx=8, pady=(0, 10))

    def _build_results_box(self, parent):
        box = ttk.LabelFrame(parent, text=" DETECTED INSTANCES ")
        box.pack(fill="both", expand=True, pady=(0, 8))

        cols = ("id", "cls", "conf", "box")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=12)
        for col, text, width, anchor in (
                ("id", "#", 34, "center"),
                ("cls", "Category", 138, "w"),
                ("conf", "Conf.", 55, "center"),
                ("box", "Box  x1,y1,x2,y2", 150, "w")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor=anchor, stretch=False)
        vsb = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0),
                       pady=8)
        vsb.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

    def _build_summary_box(self, parent):
        box = ttk.LabelFrame(parent, text=" DAMAGE SUMMARY REPORT ")
        box.pack(fill="both", expand=True, pady=(0, 8))
        self.txt_summary = tk.Text(box, height=13, bg=BG_DARK, fg=FG_TEXT,
                                   relief="flat", wrap="word",
                                   font=("Consolas", 9), padx=10, pady=8,
                                   insertbackground=FG_TEXT)
        self.txt_summary.pack(fill="both", expand=True, padx=8, pady=8)
        self.txt_summary.insert("1.0", "No analysis performed yet.")
        self.txt_summary.configure(state="disabled")

    def _build_export_box(self, parent):
        box = ttk.LabelFrame(parent, text=" EXPORT ")
        box.pack(fill="x")
        grid = ttk.Frame(box)
        grid.pack(fill="x", padx=8, pady=8)
        ttk.Button(grid, text="Save annotated image",
                   command=self.save_annotated).grid(row=0, column=0,
                                                     sticky="ew", padx=2,
                                                     pady=2)
        ttk.Button(grid, text="Save YOLO labels (.txt)",
                   command=self.save_labels).grid(row=0, column=1, sticky="ew",
                                                  padx=2, pady=2)
        ttk.Button(grid, text="Export image report (CSV)",
                   command=self.export_image_csv).grid(row=1, column=0,
                                                       sticky="ew", padx=2,
                                                       pady=2)
        ttk.Button(grid, text="Export session report (CSV)",
                   command=self.export_session_csv).grid(row=1, column=1,
                                                         sticky="ew", padx=2,
                                                         pady=2)
        ttk.Button(grid, text="Process whole folder and export",
                   style="Accent.TButton",
                   command=self.batch_process).grid(row=2, column=0,
                                                    columnspan=2, sticky="ew",
                                                    padx=2, pady=(6, 2))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ #
    #  Small callbacks                                                    #
    # ------------------------------------------------------------------ #
    def _on_conf(self, _=None):
        self.lbl_conf.configure(
            text=f"Confidence threshold : {self.var_conf.get():.2f}")

    def _on_iou(self, _=None):
        self.lbl_iou.configure(
            text=f"NMS IoU threshold : {self.var_iou.get():.2f}")

    def _on_select_row(self, _=None):
        """Highlight the selected detection by dimming the others."""
        self.redraw_annotation()

    def set_status(self, message):
        self.var_status.set(message)
        self.update_idletasks()

    # ------------------------------------------------------------------ #
    #  Model loading                                                      #
    # ------------------------------------------------------------------ #
    def browse_weights(self):
        path = filedialog.askopenfilename(
            title="Select the trained YOLO checkpoint",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.var_weights.set(path)

    def load_model(self, path):
        if not path:
            messagebox.showwarning(APP_NAME, "Please select a .pt checkpoint.")
            return
        self.set_status("Loading model, please wait...")
        self.progress.start(12)

        def worker():
            try:
                info = self.detector.load(path)
                self.after(0, lambda: self._model_loaded(info))
            except Exception as exc:
                msg = str(exc)
                self.after(0, lambda: self._model_failed(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _model_loaded(self, info):
        self.progress.stop()
        params = f"{info['params_M']} M par." if info["params_M"] else ""
        self.var_model_info.set(
            f"{info['file']}   |   {info['classes']} classes   |   "
            f"{params}   |   device: {info['device']}")
        self._build_class_filter(self.detector.class_names)
        self.btn_detect.configure(
            state="normal" if self.original_image else "disabled")
        self.set_status(
            f"Model loaded successfully on {info['device']}. "
            f"Categories: {', '.join(self.detector.class_names)}")

        if not info.get("nms_ok", True):
            messagebox.showwarning(
                f"{APP_NAME} - environment warning", info.get("nms_message", ""))
            self.set_status(
                "Model loaded. Note: pure-PyTorch NMS fallback is active "
                "(torchvision mismatch).")

    def _model_failed(self, message):
        self.progress.stop()
        self.set_status("Model could not be loaded.")
        messagebox.showerror(APP_NAME, f"Failed to load the model:\n\n{message}")

    # ------------------------------------------------------------------ #
    #  Image handling                                                     #
    # ------------------------------------------------------------------ #
    def open_images(self):
        paths = filedialog.askopenfilenames(
            title="Select concrete surface image(s)",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff "
                                  "*.webp"), ("All files", "*.*")])
        if paths:
            self.set_image_list(list(paths))

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select a folder of images")
        if not folder:
            return
        files = sorted(str(p) for p in Path(folder).iterdir()
                       if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            messagebox.showwarning(APP_NAME, "No supported images in folder.")
            return
        self.set_image_list(files)

    def set_image_list(self, paths):
        self.image_paths = paths
        self.current_index = 0
        self.load_current_image()

    def load_current_image(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        try:
            self.original_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Cannot open image:\n{exc}")
            return

        self.annotated_image = None
        self.detections = []
        self._clear_results()

        self.lbl_counter.configure(
            text=f"{self.current_index + 1} / {len(self.image_paths)}")
        self.btn_prev.configure(
            state="normal" if self.current_index > 0 else "disabled")
        self.btn_next.configure(
            state="normal"
            if self.current_index < len(self.image_paths) - 1 else "disabled")
        self.btn_detect.configure(
            state="normal" if self.detector.model else "disabled")

        w, h = self.original_image.size
        self.set_status(f"{os.path.basename(path)}   ({w} x {h} px)")
        self.refresh_canvas()

        if self.var_auto.get() and self.detector.model:
            self.run_detection()

    def prev_image(self):
        if self.current_index > 0 and not self.busy:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        if self.current_index < len(self.image_paths) - 1 and not self.busy:
            self.current_index += 1
            self.load_current_image()

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #
    def run_detection(self):
        if self.busy:
            return
        if self.detector.model is None:
            messagebox.showwarning(APP_NAME, "Load a trained model first.")
            return
        if self.original_image is None:
            messagebox.showwarning(APP_NAME, "Open an image first.")
            return

        path = self.image_paths[self.current_index]
        self.busy = True
        self.btn_detect.configure(state="disabled")
        self.progress.start(12)
        self.set_status("Running inference...")

        conf = self.var_conf.get()
        iou = self.var_iou.get()
        imgsz = int(self.var_imgsz.get())
        maxdet = self.var_maxdet.get()

        def worker():
            try:
                dets, ms = self.detector.predict(path, conf, iou, imgsz, maxdet)
                self.after(0, lambda: self._detection_done(dets, ms))
            except Exception as exc:
                msg = f"{exc}\n\n{traceback.format_exc(limit=2)}"
                self.after(0, lambda: self._detection_failed(msg))

        threading.Thread(target=worker, daemon=True).start()

    def _detection_done(self, detections, elapsed_ms):
        self.busy = False
        self.progress.stop()
        self.btn_detect.configure(state="normal")
        self.detections = detections
        self.last_elapsed_ms = elapsed_ms
        self._fill_results_table()
        self.redraw_annotation()
        self._record_session_rows()
        self.set_status(
            f"{len(detections)} defect instance(s) detected in "
            f"{elapsed_ms:.0f} ms  -  "
            f"{os.path.basename(self.image_paths[self.current_index])}")

    def _detection_failed(self, message):
        self.busy = False
        self.progress.stop()
        self.btn_detect.configure(state="normal")
        self.set_status("Inference failed.")
        messagebox.showerror(APP_NAME, f"Inference error:\n\n{message}")

    # ------------------------------------------------------------------ #
    #  Rendering                                                          #
    # ------------------------------------------------------------------ #
    def hidden_classes(self):
        return {name for name, var in self.class_visibility.items()
                if not var.get()}

    def redraw_annotation(self):
        if self.original_image is None:
            return
        if self.detections:
            self.annotated_image = draw_detections(
                self.original_image,
                self.detections,
                show_labels=self.var_labels.get(),
                show_conf=self.var_conf_txt.get(),
                line_scale=self.var_thickness.get(),
                hidden_classes=self.hidden_classes(),
            )
        else:
            self.annotated_image = None
        self._update_summary()
        self.refresh_canvas()

    def refresh_canvas(self):
        self.canvas.delete("all")
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        if self.original_image is None:
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Load a trained model, then open a concrete image.\n\n"
                     "File formats: JPG  PNG  BMP  TIFF  WEBP\n"
                     "Keyboard: Ctrl+O open   \u2190 \u2192 navigate   "
                     "Ctrl+S save",
                fill=FG_MUTED, font=("Segoe UI", 12), justify="center")
            return

        image = self.original_image
        if self.annotated_image is not None and not self.var_show_original.get():
            image = self.annotated_image

        scale = min(cw / image.width, ch / image.height)
        scale = min(scale, 1.0) if scale > 0 else 1.0
        new_w = max(1, int(image.width * scale))
        new_h = max(1, int(image.height * scale))
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        self.display_photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(cw // 2, ch // 2, image=self.display_photo,
                                 anchor="center")

    # ------------------------------------------------------------------ #
    #  Results table and summary                                          #
    # ------------------------------------------------------------------ #
    def _clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.txt_summary.configure(state="normal")
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("1.0", "No analysis performed yet.")
        self.txt_summary.configure(state="disabled")

    def _fill_results_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, det in enumerate(self.detections, start=1):
            colour = rgb_to_hex(color_for_class(det["class_name"],
                                                det["class_id"]))
            tag = f"c{det['class_id']}"
            self.tree.tag_configure(tag, foreground=colour)
            box = (f"{det['x1']:.0f},{det['y1']:.0f},"
                   f"{det['x2']:.0f},{det['y2']:.0f}")
            self.tree.insert(
                "", "end", tags=(tag,),
                values=(i,
                        PRETTY_NAMES.get(det["class_name"], det["class_name"]),
                        f"{det['confidence']:.3f}", box))

    def _update_summary(self):
        if self.original_image is None:
            return
        w, h = self.original_image.size
        hidden = self.hidden_classes()
        stats = summarise(self.detections, w, h, hidden)
        visible = [d for d in self.detections if d["class_name"] not in hidden]

        name = os.path.basename(self.image_paths[self.current_index]) \
            if self.image_paths else "-"

        lines = []
        lines.append("POST-EARTHQUAKE CONCRETE DEFECT REPORT")
        lines.append("=" * 44)
        lines.append(f"Image      : {name}")
        lines.append(f"Resolution : {w} x {h} px")
        lines.append(f"Model      : {self.detector.model_info.get('file', '-')}")
        lines.append(f"Conf / IoU : {self.var_conf.get():.2f} / "
                     f"{self.var_iou.get():.2f}")
        lines.append(f"Inference  : {self.last_elapsed_ms:.0f} ms "
                     f"({self.detector.device})")
        lines.append("-" * 44)

        if not visible:
            lines.append("No defect instance exceeded the confidence")
            lines.append("threshold for the selected categories.")
            lines.append("")
            lines.append("Note: a negative result is not evidence of a")
            lines.append("sound structure. Visual inspection by a qualified")
            lines.append("engineer remains mandatory.")
        else:
            lines.append(f"{'Category':<22}{'N':>4}{'Conf':>7}{'Area%':>8}")
            total_area = 0.0
            for cname in self.detector.class_names:
                if cname not in stats:
                    continue
                s = stats[cname]
                total_area += s["area_pct"]
                lines.append(f"{PRETTY_NAMES.get(cname, cname):<22}"
                             f"{s['count']:>4}"
                             f"{s['mean_conf']:>7.2f}"
                             f"{s['area_pct']:>8.2f}")
            lines.append("-" * 44)
            lines.append(f"{'TOTAL':<22}{len(visible):>4}"
                         f"{'':>7}{total_area:>8.2f}")
            lines.append("")
            dominant = max(stats.items(), key=lambda kv: kv[1]["area_pct"])[0]
            lines.append(f"Dominant defect by surface extent: "
                         f"{PRETTY_NAMES.get(dominant, dominant)}")
            lines.append("")
            lines.append("Area% = bounding-box coverage relative to the image")
            lines.append("area; it is an indicative extent proxy, not a")
            lines.append("calibrated physical measurement.")

        self.txt_summary.configure(state="normal")
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("1.0", "\n".join(lines))
        self.txt_summary.configure(state="disabled")

    def _record_session_rows(self):
        """Append the current image's detections to the cumulative report."""
        if not self.image_paths:
            return
        path = self.image_paths[self.current_index]
        w, h = self.original_image.size
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Remove earlier rows for this image so re-running does not duplicate.
        self.session_rows = [r for r in self.session_rows
                             if r["image"] != os.path.basename(path)]
        if not self.detections:
            self.session_rows.append({
                "timestamp": stamp, "image": os.path.basename(path),
                "width": w, "height": h, "category": "none", "confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "", "box_area_pct": "",
            })
            return
        for det in self.detections:
            area_pct = 100.0 * (det["x2"] - det["x1"]) * \
                       (det["y2"] - det["y1"]) / float(w * h)
            self.session_rows.append({
                "timestamp": stamp,
                "image": os.path.basename(path),
                "width": w, "height": h,
                "category": PRETTY_NAMES.get(det["class_name"],
                                             det["class_name"]),
                "confidence": round(det["confidence"], 4),
                "x1": round(det["x1"], 1), "y1": round(det["y1"], 1),
                "x2": round(det["x2"], 1), "y2": round(det["y2"], 1),
                "box_area_pct": round(area_pct, 3),
            })

    # ------------------------------------------------------------------ #
    #  Export                                                             #
    # ------------------------------------------------------------------ #
    def save_annotated(self):
        if self.annotated_image is None:
            messagebox.showwarning(APP_NAME, "Run a detection first.")
            return
        stem = Path(self.image_paths[self.current_index]).stem
        path = filedialog.asksaveasfilename(
            title="Save annotated image",
            defaultextension=".png",
            initialfile=f"{stem}_PECCD_detected.png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if not path:
            return
        try:
            self.annotated_image.save(path)
            self.set_status(f"Annotated image saved to {path}")
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not save image:\n{exc}")

    def save_labels(self):
        """Write detections in YOLO format (class cx cy w h, normalised)."""
        if not self.detections:
            messagebox.showwarning(APP_NAME, "Run a detection first.")
            return
        stem = Path(self.image_paths[self.current_index]).stem
        path = filedialog.asksaveasfilename(
            title="Save YOLO-format labels",
            defaultextension=".txt", initialfile=f"{stem}.txt",
            filetypes=[("Text file", "*.txt")])
        if not path:
            return
        w, h = self.original_image.size
        try:
            with open(path, "w", encoding="utf-8") as fh:
                for det in self.detections:
                    cx = ((det["x1"] + det["x2"]) / 2.0) / w
                    cy = ((det["y1"] + det["y2"]) / 2.0) / h
                    bw = (det["x2"] - det["x1"]) / w
                    bh = (det["y2"] - det["y1"]) / h
                    fh.write(f"{det['class_id']} {cx:.6f} {cy:.6f} "
                             f"{bw:.6f} {bh:.6f}\n")
            self.set_status(f"YOLO labels saved to {path}")
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not save labels:\n{exc}")

    def _write_csv(self, path, rows):
        fields = ["timestamp", "image", "width", "height", "category",
                  "confidence", "x1", "y1", "x2", "y2", "box_area_pct"]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def export_image_csv(self):
        if not self.image_paths:
            return
        current = os.path.basename(self.image_paths[self.current_index])
        rows = [r for r in self.session_rows if r["image"] == current]
        if not rows:
            messagebox.showwarning(APP_NAME, "Run a detection first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export report for the current image",
            defaultextension=".csv",
            initialfile=f"{Path(current).stem}_report.csv",
            filetypes=[("CSV file", "*.csv")])
        if path:
            self._write_csv(path, rows)
            self.set_status(f"Report exported to {path}")

    def export_session_csv(self):
        if not self.session_rows:
            messagebox.showwarning(APP_NAME, "Nothing analysed yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Export cumulative session report",
            defaultextension=".csv",
            initialfile="PECCD_session_report.csv",
            filetypes=[("CSV file", "*.csv")])
        if path:
            self._write_csv(path, self.session_rows)
            self.set_status(f"Session report exported to {path}")

    # ------------------------------------------------------------------ #
    #  Batch processing                                                   #
    # ------------------------------------------------------------------ #
    def batch_process(self):
        if self.detector.model is None:
            messagebox.showwarning(APP_NAME, "Load a trained model first.")
            return
        if not self.image_paths:
            messagebox.showwarning(APP_NAME, "Open a folder of images first.")
            return
        out_dir = filedialog.askdirectory(
            title="Select an output folder for annotated images and report")
        if not out_dir:
            return
        if self.busy:
            return

        self.busy = True
        self.progress.configure(mode="determinate", maximum=len(self.image_paths),
                                value=0)
        conf, iou = self.var_conf.get(), self.var_iou.get()
        imgsz, maxdet = int(self.var_imgsz.get()), self.var_maxdet.get()
        paths = list(self.image_paths)
        labels_flag = self.var_labels.get()
        conf_flag = self.var_conf_txt.get()
        thick = self.var_thickness.get()

        def worker():
            rows = []
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for i, path in enumerate(paths, start=1):
                try:
                    dets, _ = self.detector.predict(path, conf, iou, imgsz,
                                                    maxdet)
                    image = Image.open(path).convert("RGB")
                    annotated = draw_detections(image, dets,
                                                show_labels=labels_flag,
                                                show_conf=conf_flag,
                                                line_scale=thick)
                    out_name = f"{Path(path).stem}_detected.png"
                    annotated.save(os.path.join(out_dir, out_name))
                    w, h = image.size
                    if not dets:
                        rows.append({"timestamp": stamp,
                                     "image": os.path.basename(path),
                                     "width": w, "height": h,
                                     "category": "none", "confidence": "",
                                     "x1": "", "y1": "", "x2": "", "y2": "",
                                     "box_area_pct": ""})
                    for det in dets:
                        area_pct = 100.0 * (det["x2"] - det["x1"]) * \
                                   (det["y2"] - det["y1"]) / float(w * h)
                        rows.append({
                            "timestamp": stamp,
                            "image": os.path.basename(path),
                            "width": w, "height": h,
                            "category": PRETTY_NAMES.get(det["class_name"],
                                                         det["class_name"]),
                            "confidence": round(det["confidence"], 4),
                            "x1": round(det["x1"], 1),
                            "y1": round(det["y1"], 1),
                            "x2": round(det["x2"], 1),
                            "y2": round(det["y2"], 1),
                            "box_area_pct": round(area_pct, 3)})
                except Exception as exc:
                    rows.append({"timestamp": stamp,
                                 "image": os.path.basename(path),
                                 "width": "", "height": "",
                                 "category": f"ERROR: {exc}", "confidence": "",
                                 "x1": "", "y1": "", "x2": "", "y2": "",
                                 "box_area_pct": ""})
                self.after(0, lambda i=i, p=path: self._batch_tick(
                    i, len(paths), os.path.basename(p)))

            csv_path = os.path.join(out_dir, "PECCD_batch_report.csv")
            try:
                self._write_csv(csv_path, rows)
            except Exception:
                pass
            self.after(0, lambda: self._batch_done(out_dir, len(paths)))

        threading.Thread(target=worker, daemon=True).start()

    def _batch_tick(self, i, total, name):
        self.progress.configure(value=i)
        self.set_status(f"Batch processing {i}/{total}  -  {name}")

    def _batch_done(self, out_dir, total):
        self.busy = False
        self.progress.configure(mode="indeterminate", value=0)
        self.set_status(f"Batch complete: {total} image(s) written to {out_dir}")
        messagebox.showinfo(
            APP_NAME,
            f"Batch processing finished.\n\n{total} image(s) analysed.\n"
            f"Annotated images and PECCD_batch_report.csv were saved to:\n"
            f"{out_dir}")


# =========================================================================== #
#  SECTION 6.  ENTRY POINT                                                     #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="PECCD-Detect: GUI for post-earthquake concrete defect "
                    "detection with YOLOV12SDSDA.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to the trained .pt checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Optional image to open at start-up")
    args = parser.parse_args()

    app = PECCDApp(weights=args.weights, image=args.image)
    app.mainloop()


if __name__ == "__main__":
    main()
