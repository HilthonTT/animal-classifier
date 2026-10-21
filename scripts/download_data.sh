#!/usr/bin/env bash
# Downloads the binary version of CIFAR-10 (~163 MB) into data/cifar-10-batches-bin.
# CIFAR-10: Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.
set -euo pipefail

URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5="c32a1d4ab5d03f1284b67883e8d87530"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
TARGET="$DATA_DIR/cifar-10-batches-bin"
ARCHIVE="$DATA_DIR/cifar-10-binary.tar.gz"

if [[ -f "$TARGET/test_batch.bin" ]]; then
    echo "CIFAR-10 is already in $TARGET"
    exit 0
fi

mkdir -p "$DATA_DIR"
echo "Downloading $URL"
if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "$ARCHIVE" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$ARCHIVE" "$URL"
else
    echo "error: neither curl nor wget is installed" >&2
    exit 1
fi

if command -v md5sum >/dev/null 2>&1; then
    actual="$(md5sum "$ARCHIVE" | cut -d' ' -f1)"
elif command -v md5 >/dev/null 2>&1; then
    actual="$(md5 -q "$ARCHIVE")"
else
    actual="$MD5"
    echo "warning: no md5 tool found, skipping checksum verification"
fi
if [[ "$actual" != "$MD5" ]]; then
    echo "error: checksum mismatch (expected $MD5, got $actual); delete $ARCHIVE and retry" >&2
    exit 1
fi

echo "Extracting..."
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
rm "$ARCHIVE"
echo "Done: $TARGET"
