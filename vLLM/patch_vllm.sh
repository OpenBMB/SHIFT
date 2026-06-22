#!/usr/bin/env bash
set -euo pipefail

# Locate the installed vLLM package in the current Python environment.
VLLM_DIR=$(python - <<'PY'
import pathlib
import vllm

print(pathlib.Path(vllm.__file__).resolve().parent)
PY
)

# Locate the patch files.
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$VLLM_DIR/model_executor/models"

echo "Detected vLLM directory: $VLLM_DIR"
echo "Patch source directory: $PATCH_DIR"
echo "Target directory: $TARGET_DIR"

# Check required patch files.
if [[ ! -f "$PATCH_DIR/qwen3.py" ]]; then
  echo "Error: missing $PATCH_DIR/qwen3.py"
  exit 1
fi

if [[ ! -f "$PATCH_DIR/llama.py" ]]; then
  echo "Error: missing $PATCH_DIR/llama.py"
  exit 1
fi

# Check target directory.
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: target directory does not exist: $TARGET_DIR"
  exit 1
fi

# Backup original files.
cp "$TARGET_DIR/qwen3.py" "$TARGET_DIR/qwen3.py.bak"
cp "$TARGET_DIR/llama.py" "$TARGET_DIR/llama.py.bak"

# Apply patches.
cp "$PATCH_DIR/qwen3.py" "$TARGET_DIR/qwen3.py"
cp "$PATCH_DIR/llama.py" "$TARGET_DIR/llama.py"

echo "vLLM files patched successfully."
echo "Patched files:"
echo "  $TARGET_DIR/qwen3.py"
echo "  $TARGET_DIR/llama.py"
echo "Backups:"
echo "  $TARGET_DIR/qwen3.py.bak"
echo "  $TARGET_DIR/llama.py.bak"
