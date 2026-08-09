# Publishing checklist

Everything needed to put this repository online, in order. Delete this file once you are
done — it is scaffolding, not documentation.

---

## 1. Fill in the placeholders

Search the repository for these and replace them:

| Placeholder | Appears in | Replace with |
|---|---|---|
| `<AliMMayya>` | `README.md`, `CITATION.cff` | Your GitHub username or organisation |
| `Nondestructive testing and evaluation` | `README.md`, `CITATION.cff` | Journal name, volume, DOI, once accepted |
| *see paper* | `README.md` results table | The mAP50-95 value of YOLOV12SDSDA |

The relative links (`../../releases`, `docs/...`) resolve automatically once the
repository exists — they need no editing.

---

## 2. Verify the checkpoint before you release it

This is the one step that is easy to skip and expensive to get wrong. Run the inspection
snippet in `models/README.md` against the `.pt` file you intend to publish and confirm:

* the `DualAttentionDS` block count is **not zero**, if you are publishing the proposed
  architecture;
* the parameter count matches the architecture named in the README (≈2.6 M for
  YOLOv12n, ≈9.3 M for YOLOv12s);
* the six class names are present and in the expected order.

Publishing a baseline checkpoint under the name of the proposed model is the kind of
discrepancy a reader will notice and report.

---

## 3. Create the repository

```bash
cd PECCD-Detect
git init
git add .
git commit -m "Initial release: PECCD-Detect v1.0.0"
git branch -M main
git remote add origin https://github.com/<your-account>/PECCD-Detect.git
git push -u origin main
```

Then, in the GitHub web interface:

* **Add file → Create new file**, name it `LICENSE`, choose the
  *GNU Affero General Public License v3.0* template.
* **Settings → About** (the gear beside the repository description): add a one-line
  description and the topics `deep-learning`, `object-detection`, `yolo`,
  `structural-health-monitoring`, `earthquake`, `crack-detection`, `concrete`.

---

## 4. Build and attach the executable

Follow `build/BUILD.md`. Then:

* **Releases → Draft a new release**, tag `v1.0.0`, title
  *PECCD-Detect v1.0.0 — initial release*.
* Attach `PECCD-Detect-win64.zip` and `bestV12CrackClass.pt`.
* In the release notes state: what the tool does in two sentences, the SHA-256 checksums,
  the system requirements, and the note about the SmartScreen warning.

---

## 5. Test as a stranger would

Open the repository in a private browser window and follow your own quick-start
instructions on a computer without Python. If anything is ambiguous at that point, it
will be ambiguous for every reader.

---

## 6. Link it from the paper

Add the repository URL to the manuscript, next to the existing data-availability
statement:

> **Code availability.** The graphical application described in Section 5.9, together
> with its source code and the trained model, is openly available at
> https://github.com/<your-account>/PECCD-Detect

Reviewers who ask for a GUI expect to be able to reach it. Consider archiving the
repository on Zenodo as well: connecting Zenodo to GitHub mints a DOI for each release,
which makes the software independently citable and gives it a permanent archive
independent of GitHub.

---

## 7. Optional, worth doing later

* **A short screen recording** (30–60 seconds: load model, open image, detect, export)
  converted to an animated GIF and embedded in the README. It communicates the tool
  faster than any paragraph.
* **A `sample_images/` folder** with four or five representative photographs, so a user
  can test the application before finding their own data. Keep it under a few megabytes.
* **A GitHub Actions workflow** that builds the executable on every tagged commit, if you
  expect to publish updates.
