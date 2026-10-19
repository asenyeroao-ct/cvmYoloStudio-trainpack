# cvmYoloStudio-trainpack

A component of YOLO Studio: the module `ul_bridge`, which passes requests from YOLO Studio to
[Ultralytics](https://github.com/ultralytics/ultralytics) and returns the results.

- License: GNU AGPL-3.0-or-later (see `LICENSE`).
- Source: this repository. YOLO Studio: https://github.com/asenyeroao-ct/cvmYoloStudio
- Third-party components: see `THIRD_PARTY_NOTICES.md`. They are installed separately and are not included here.
- Releases: each GitHub Release has `ul_bridge-<version>.zip` and its `.sha256`. YOLO Studio downloads the
  version it is paired with.

The protocol is internal to YOLO Studio and changes together with it.
