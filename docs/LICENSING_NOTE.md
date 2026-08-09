# Licensing note

This document explains why this repository is licensed under the GNU Affero General
Public License v3.0 (AGPL-3.0), and what that means when the compiled application is
distributed. It is written for researchers rather than lawyers, and it is not legal
advice; if the stakes are high for your institution, ask your technology transfer
office.

## The short version

The application runs on the **Ultralytics YOLO** framework, which is published under
AGPL-3.0. The licence string is embedded in the trained checkpoint itself:

```
license: AGPL-3.0 (https://ultralytics.com/license)
```

AGPL-3.0 is a strong copyleft licence. When a work that incorporates AGPL-licensed code
is distributed — including as a compiled binary — the complete corresponding source code
of that work must be made available to the recipients under the same licence.

**Consequence:** shipping `PECCD-Detect.exe` without also publishing its source would not
satisfy the licence. This repository therefore publishes both. Users who simply want to
run the tool download the executable and never open the source; the source is present so
that the distribution is compliant.

## Why "executable only" is not achievable here anyway

Even setting the licence aside, a PyInstaller executable does not conceal source code in
any meaningful sense. PyInstaller bundles the Python bytecode into the binary, and
publicly available tools extract and decompile it in a few minutes. Anyone motivated to
read the code will read it.

If genuine obfuscation were required, the options would be:

* **Nuitka** — compiles Python to C and then to native machine code, which is
  substantially harder to reverse-engineer than bytecode.
* **A server-side deployment** — the model runs on your infrastructure and users submit
  images through a web interface, so no code is distributed at all. Note that AGPL-3.0
  specifically closes this loophole: network users of an AGPL work are entitled to its
  source.

Neither changes the AGPL obligation. Publishing the source is the straightforward path,
and for an academic release it is also the one that reviewers and readers expect.

## What this means for your users

Anyone may download, run, study, modify and redistribute this software, provided that
redistributed versions remain under AGPL-3.0 and carry their source. For most academic
users this is invisible: they download the executable and use it.

## If a permissive licence is required

If your institution or an industrial partner needs to distribute the tool under
different terms, the dependency on Ultralytics must be removed or relicensed. Two routes
exist:

1. **Purchase an Ultralytics Enterprise Licence**, which permits distribution without the
   AGPL source-disclosure obligation. Contact Ultralytics directly.
2. **Replace the inference back end** with a framework under a permissive licence, and
   export the trained weights to a neutral format such as ONNX, then run them through
   ONNX Runtime (MIT). The trained weights themselves are your work; it is the Ultralytics
   code around them that carries the AGPL condition. This route requires reimplementing
   the pre-processing and NMS post-processing, which is a few hundred lines of work.

Option 2 is worth considering if you expect industrial uptake, since it would also remove
the torch/torchvision version fragility documented in `docs/TROUBLESHOOTING.md`.

## Adding the licence file

GitHub can insert the official text for you: in the repository, choose
**Add file → Create new file**, type `LICENSE` as the filename, and a
*Choose a licence template* button appears. Select **GNU Affero General Public License
v3.0**. Alternatively, download the text directly:

```bash
curl -o LICENSE https://www.gnu.org/licenses/agpl-3.0.txt
```

The dataset is a separate work and is governed by the terms stated on its Mendeley Data
record, not by this licence.
