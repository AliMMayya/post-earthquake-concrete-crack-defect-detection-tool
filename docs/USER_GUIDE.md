# PECCD-Detect — User guide

A guide for engineers and researchers who want to use the tool without installing Python
or reading any code.

---

## 1. What you need

| Item | Where to get it |
|---|---|
| `PECCD-Detect-win64.zip` | The [Releases](../../releases) page of this repository |
| `bestV12CrackClass.pt` | The same Releases page |
| A Windows 10 or 11 computer (64-bit) | — |

A graphics card is optional. Without one the application runs on the processor, which
takes roughly one to three seconds per photograph instead of a fraction of a second.

No internet connection is needed once the files are downloaded, and nothing is installed
onto the system: the application runs entirely from the folder you extract it into.

---

## 2. First run

1. **Extract the ZIP.** Right-click `PECCD-Detect-win64.zip` → *Extract All*. Choose any
   location — Desktop or Documents are both fine. Do not run the application from inside
   the compressed folder without extracting it first; it will fail to find its own files.

2. **Start the application.** Open the extracted folder and double-click
   `PECCD-Detect.exe`.

   If Windows shows a blue *Windows protected your PC* dialogue, click **More info**,
   then **Run anyway**. This appears because the executable is not code-signed, which is
   normal for academic software; a code-signing certificate is a commercial product.

3. **Load the model.** In panel *1. TRAINED MODEL*, click **Browse .pt**, select the
   downloaded `bestV12CrackClass.pt`, then click **Load model**.

   The top-right of the window then reports the file name, the number of categories, the
   parameter count and whether the computation will run on the graphics card (`cuda:0`)
   or the processor (`cpu`). This confirms the model is ready.

---

## 3. Analysing photographs

### A single image

Click **Select image(s)…** in panel *2*, choose one or more photographs, and the first
one is displayed. If *Detect automatically on open* is checked, the analysis starts
immediately otherwise press **RUN DETECTION**.

### A whole folder

Click **Select folder (batch)…** and choose a directory of photographs. Move through
them with the **Previous** / **Next** buttons or the left and right arrows.

### Reading the output

Each detected defect is drawn as a rectangle in the colour of its category, tagged with
the category name and a confidence score between 0 and 1. The same detections appear:

* in the **DETECTED INSTANCES** table on the right, with their pixel coordinates;
* in the **DAMAGE SUMMARY REPORT** panel below it, aggregated per category.

The summary reports, for each category, how many instances were found, their mean
confidence, and the percentage of the photograph enclosed by the corresponding
rectangles. That percentage indicates the *extent* of each deterioration mechanism in
the image.

---

## 4. Adjusting the analysis

| Control | What it does | When to change it |
|---|---|---|
| **Confidence threshold** | Minimum score for a detection to be shown | Lower it (0.10–0.15) if defects you can see are being missed; raise it (0.40+) if too many spurious boxes appear |
| **NMS IoU threshold** | How much two boxes may overlap before one is discarded | Raise it when overlapping defects are being merged into one box |
| **Inference size** | Resolution at which the image is analysed | Raise to 960 or 1280 for hairline cracks in large photographs but it is slower |
| **Defect categories** | Show or hide one category |

| **Annotation thickness** | Line and text size |

After changing a threshold, press **RUN DETECTION** again. Changing display options
(labels, colours, thickness, category visibility) redraws immediately without re-running
the analysis.

Select **Show original (no boxes)** to compare the annotated result against the untouched
photograph.

---

## 5. Saving results

| Button | Output |
|---|---|
| **Save annotated image** | The displayed image with its boxes, as a PNG suitable for a report or a paper figure |
| **Save YOLO labels (.txt)** | Predictions in YOLO annotation format — useful as a starting point for manual correction, or to extend the dataset |
| **Export image report (CSV)** | One row per detected defect in the current image |
| **Export session report (CSV)** | Every image analyzed since the application was opened |
| **Process whole folder and export** | Analyses every image in the queue, writes each annotated image plus a combined `PECCD_batch_report.csv` into a folder you choose |

The batch option is the one to use for a survey campaign: point it at a folder of several
hundred photographs, choose an output directory, and let it run.

The CSV columns are: timestamp, image, width, height, category, confidence, x1, y1, x2,
y2, box_area_pct. They open directly in Excel.

---

## 6. Important cautions

**This tool assists an inspection; it does not perform one.**

* A photograph in which nothing is detected is **not** evidence of a sound structure.
  Defects may be outside the field of view, hidden, or too fine to be
  visible at the acquisition resolution.
* Measurements are in **pixels, not millimetres**. Crack width — the quantity most
  assessment procedures depend on — cannot be derived from these results without a scale
  reference.
* The categories *simple crack* and *deep crack* are visually similar and are the two the
  model confuses most often.
* The *hole* category is under-represented in the training data and is therefore more
  sensitive to the confidence threshold than the others.

All conclusions about structural safety must be drawn by a qualified engineer.

---

## 7. Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl + O` | Open image(s) |
| `Ctrl + S` | Save the annotated image |
| `←` / `→` | Previous / next image |

---

## 8. Getting help

If something does not work, check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first — it
covers the errors that occur in practice. If the problem persists, open an issue on the
repository and include: the message shown in the dialogue box, the text in the status bar
at the bottom of the window, and the model information line at the top right.
