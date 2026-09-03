#!/usr/bin/env bash
# Replace the PyPI pyrealsense2 wheel in this project's .venv with the
# CUDA + BUILD_WITH_CUDA_ZEROCOPY build installed under /usr/local.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"
SYS_PKG="/usr/local/lib/python3.12/dist-packages/pyrealsense2"
SYS_LIB="/usr/local/lib/librealsense2.so"

die() { echo "error: $*" >&2; exit 1; }

[[ -d "$VENV" ]] || die "no .venv at $VENV — create it first (python3 -m venv .venv)"
[[ -d "$SYS_PKG" ]] || die "missing $SYS_PKG — build/install librealsense with -DBUILD_PYTHON_BINDINGS=ON -DBUILD_WITH_CUDA=ON -DBUILD_WITH_CUDA_ZEROCOPY=ON"
[[ -e "$SYS_LIB" ]] || die "missing $SYS_LIB"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "uninstalling PyPI pyrealsense2 from venv (if present)…"
python -m pip uninstall -y pyrealsense2 >/dev/null 2>&1 || true

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
DEST="${SITE}/pyrealsense2"

rm -rf "${DEST}" "${SITE}"/pyrealsense2-*.dist-info
ln -s "$SYS_PKG" "$DEST"
echo "linked $DEST -> $SYS_PKG"

# Persist LD_LIBRARY_PATH when the venv is activated.
ACTIVATE="$VENV/bin/activate"
MARKER="# >>> realsense-vlm zerocopy >>>"
if ! grep -qF "$MARKER" "$ACTIVATE"; then
  cat >> "$ACTIVATE" <<'EOF'

# >>> realsense-vlm zerocopy >>>
# Prefer /usr/local CUDA+ZEROCOPY librealsense2 over ROS / PyPI embeds.
export LD_LIBRARY_PATH=/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/usr/local/bin${PATH:+:$PATH}
# <<< realsense-vlm zerocopy <<<
EOF
  echo "patched $ACTIVATE with LD_LIBRARY_PATH=/usr/local/lib"
else
  echo "activate hook already present"
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/rs_zerocopy_env.sh"

python - <<'PY'
import pyrealsense2 as rs
print("pyrealsense2:", rs.__file__)
print("version:", rs.__version__, getattr(rs, "__full_version__", ""))
assert hasattr(rs, "gpu_frame"), "gpu_frame missing — wrong module"
print("gpu_frame API: ok")
PY

echo
echo "done. Activate and test:"
echo "  source .venv/bin/activate"
echo "  python test_zerocopy.py"
