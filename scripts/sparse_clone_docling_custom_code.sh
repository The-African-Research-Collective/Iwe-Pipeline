#!/bin/bash
set -e

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

git clone --no-checkout --depth 1 https://github.com/huggingface/finepdfs.git "$TMP_DIR"
cd "$TMP_DIR"
git sparse-checkout init --cone
git sparse-checkout set docling_code/custom_code
GIT_LFS_SKIP_SMUDGE=1 git checkout
cd -

mkdir -p docling_code
mv "$TMP_DIR/docling_code/custom_code" docling_code/custom_code