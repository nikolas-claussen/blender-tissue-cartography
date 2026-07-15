#!/usr/bin/env python3
"""Patch the pymeshlab Windows wheel for the Blender add-on.

Two changes are applied to the stock wheel downloaded by download_wheels.sh:

1. ``pymeshlab/__init__.py`` is replaced with
   ``wheel_patch/windows/pymeshlab/__init__.py``, which does not call
   ``load_default_plugins()`` at import time (loading
   ``filter_texture_defragmentation.dll`` crashes Blender on Windows,
   see DEVELOPER_README.md). The add-on loads the plugins it needs via
   ``pymeshlab_runtime.py``.

2. The wheel version is bumped ``2025.7.post1`` -> ``2025.7.post1+btc1``.
   Blender's extension wheel manager decides whether a wheel needs
   (re)installation purely by its dist-info directory name in the shared
   extensions site-packages. Without the bump, a machine that already has
   the *stock* ``pymeshlab-2025.7.post1`` extracted (e.g. from an earlier
   add-on version) would keep it forever and the patched wheel would never
   be installed. The local version tag makes Blender remove the stale stock
   package and extract the patched one.

Usage (from pymeshlab_addon_src/, after download_wheels.sh):

    python3 patch_windows_wheel.py

Reads  wheels/pymeshlab-2025.7.post1-cp313-cp313-win_amd64.whl
Writes wheels/pymeshlab-2025.7.post1+btc1-cp313-cp313-win_amd64.whl
and removes the input wheel so the build does not bundle both.
"""

import base64
import csv
import hashlib
import io
import pathlib
import sys
import zipfile

SRC_VERSION = "2025.7.post1"
DST_VERSION = SRC_VERSION + "+btc1"
WHEEL_TAGS = "cp313-cp313-win_amd64"

ROOT = pathlib.Path(__file__).resolve().parent
SRC_WHEEL = ROOT / "wheels" / f"pymeshlab-{SRC_VERSION}-{WHEEL_TAGS}.whl"
DST_WHEEL = ROOT / "wheels" / f"pymeshlab-{DST_VERSION}-{WHEEL_TAGS}.whl"
PATCH_INIT = ROOT / "wheel_patch" / "windows" / "pymeshlab" / "__init__.py"

SRC_DATA_DIR = f"pymeshlab-{SRC_VERSION}.data"
SRC_INFO_DIR = f"pymeshlab-{SRC_VERSION}.dist-info"
DST_DATA_DIR = f"pymeshlab-{DST_VERSION}.data"
DST_INFO_DIR = f"pymeshlab-{DST_VERSION}.dist-info"

INIT_PATH = f"{SRC_DATA_DIR}/purelib/pymeshlab/__init__.py"
METADATA_PATH = f"{SRC_INFO_DIR}/METADATA"
RECORD_PATH = f"{SRC_INFO_DIR}/RECORD"


def _rename(path):
    if path.startswith(SRC_DATA_DIR + "/"):
        return DST_DATA_DIR + path[len(SRC_DATA_DIR):]
    if path.startswith(SRC_INFO_DIR + "/"):
        return DST_INFO_DIR + path[len(SRC_INFO_DIR):]
    return path


def _record_hash(data):
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _remap_record(record_data, changed_files):
    """Rename paths and refresh hash/size for the files whose content changed."""
    rows_out = []
    for row in csv.reader(io.StringIO(record_data, newline="")):
        if not row:
            continue
        path = row[0]
        if path in changed_files:
            data = changed_files[path]
            row = [path, _record_hash(data), str(len(data))]
        row[0] = _rename(row[0])
        rows_out.append(row)
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows_out)
    return out.getvalue().encode("utf8")


def main():
    if not SRC_WHEEL.exists():
        if DST_WHEEL.exists():
            print(f"{SRC_WHEEL.name} not found but {DST_WHEEL.name} exists — nothing to do.")
            return 0
        print(f"Input wheel not found: {SRC_WHEEL}\nRun download_wheels.sh first.", file=sys.stderr)
        return 1

    patched_init = PATCH_INIT.read_bytes()

    with zipfile.ZipFile(SRC_WHEEL) as zin:
        metadata = zin.read(METADATA_PATH).replace(
            f"Version: {SRC_VERSION}".encode(), f"Version: {DST_VERSION}".encode(), 1
        )
        changed_files = {INIT_PATH: patched_init, METADATA_PATH: metadata}
        record = _remap_record(zin.read(RECORD_PATH).decode("utf8"), changed_files)
        replacements = dict(changed_files)
        replacements[RECORD_PATH] = record

        with zipfile.ZipFile(DST_WHEEL, "w", zipfile.ZIP_DEFLATED) as zout:
            for member in zin.infolist():
                data = replacements.get(member.filename, None)
                if data is None:
                    data = zin.read(member.filename)
                member.filename = _rename(member.filename)
                zout.writestr(member, data)

    SRC_WHEEL.unlink()
    print(f"Wrote {DST_WHEEL.name} (patched __init__.py, version {DST_VERSION})")
    print(f"Removed {SRC_WHEEL.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
