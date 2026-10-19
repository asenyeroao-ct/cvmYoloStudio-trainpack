"""Builds the release asset: dist/ul_bridge-<version>.zip and dist/ul_bridge-<version>.zip.sha256.

The zip is reproducible: fixed entry order, timestamps and permissions, stored (not compressed), and text
files with LF line endings, so the same commit gives the same bytes on any platform.

Layout inside the zip (extracted as-is into a Python site folder):

    ul_bridge/__init__.py
    ul_bridge/__main__.py
    ul_bridge/LICENSE
    ul_bridge/README.md
    ul_bridge/THIRD_PARTY_NOTICES.md

    python scripts/build_release.py [--tag vX.Y.Z] [--out dist]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ("ul_bridge/__init__.py", "ul_bridge/__init__.py"),
    ("ul_bridge/__main__.py", "ul_bridge/__main__.py"),
    ("LICENSE", "ul_bridge/LICENSE"),
    ("README.md", "ul_bridge/README.md"),
    ("THIRD_PARTY_NOTICES.md", "ul_bridge/THIRD_PARTY_NOTICES.md"),
]


def constants() -> dict:
    """__version__ and PROTOCOL_VERSION from ul_bridge/__init__.py (read, not imported)."""
    tree = ast.parse((ROOT / "ul_bridge" / "__init__.py").read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("__version__", "PROTOCOL_VERSION"):
                    out[target.id] = node.value.value
    return out


def build(out_dir: Path) -> tuple[Path, str]:
    version = constants()["__version__"]
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"ul_bridge-{version}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for src, arc in FILES:
            data = (ROOT / src).read_bytes().replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, data)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    (out_dir / f"{zip_path.name}.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="ascii", newline="\n")
    return zip_path, digest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--print", choices=["version", "protocol"], default="")
    args = ap.parse_args()
    c = constants()
    if args.print:
        print(c["__version__"] if args.print == "version" else c["PROTOCOL_VERSION"])
        return 0
    if args.tag and args.tag != f"v{c['__version__']}":
        print(f"tag {args.tag} does not match __version__ {c['__version__']}", file=sys.stderr)
        return 1
    zip_path, digest = build(Path(args.out))
    print(f"{zip_path.name} {zip_path.stat().st_size} bytes sha256={digest} protocol={c['PROTOCOL_VERSION']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
