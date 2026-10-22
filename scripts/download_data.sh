#!/usr/bin/env bash
# Downloads the binary version of CIFAR-10 (~163 MB) into data/cifar-10-batches-bin,
# or with `cifar100` the binary version of CIFAR-100 (~161 MB) into data/cifar-100-binary.
# Alex Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009.
set -euo pipefail

case "${1:-cifar10}" in
    cifar10)
        NAME="CIFAR-10"
        URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
        MD5="c32a1d4ab5d03f1284b67883e8d87530"
        FOLDER="cifar-10-batches-bin"
        MARKER="test_batch.bin"
        ;;
    cifar100)
        NAME="CIFAR-100"
        URL="https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
        MD5="03b5dce01913d631647c71ecec9e9cb8"
        FOLDER="cifar-100-binary"
        MARKER="test.bin"
        ;;
    *)
        echo "usage: $0 [cifar10|cifar100]" >&2
        exit 1
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
TARGET="$DATA_DIR/$FOLDER"
ARCHIVE="$DATA_DIR/$(basename "$URL")"

if [[ -f "$TARGET/$MARKER" ]]; then
    echo "$NAME is already in $TARGET"
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
