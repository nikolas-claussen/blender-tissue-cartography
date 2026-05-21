#!/usr/bin/env bash
# Download pymeshlab wheels for all supported platforms into wheels/.
# Run from the pymeshlab_remesh_addon/ directory before building the extension.
# Requires pip >= 22 and Python 3.x in PATH.
set -euo pipefail

DEST="$(dirname "$0")/wheels"
PYMLVER="pymeshlab==2023.12.post2"
PY="311"

mkdir -p "$DEST"

pip download "$PYMLVER" --no-deps \
    --platform macosx_11_0_arm64 \
    --only-binary=:all: --python-version "$PY" \
    -d "$DEST"

pip download "$PYMLVER" --no-deps \
    --platform manylinux_2_31_x86_64 \
    --only-binary=:all: --python-version "$PY" \
    -d "$DEST"

pip download "$PYMLVER" --no-deps \
    --platform win_amd64 \
    --only-binary=:all: --python-version "$PY" \
    -d "$DEST"

echo "Wheels downloaded to $DEST:"
ls "$DEST"
