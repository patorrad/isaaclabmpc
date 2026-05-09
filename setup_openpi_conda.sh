#!/usr/bin/env bash
# Creates a conda environment named 'openpi' with JAX+CUDA and the openpi package.
# Also installs openpi-client into the existing isaaclab env so world.py can reach
# the policy server without pulling in JAX.
#
# Usage:
#   bash setup_openpi_conda.sh
#
# After running, start the pi0.5 server with:
#   conda activate openpi
#   cd ~/Documents/openpi
#   python scripts/serve_policy.py --env=DROID --checkpoint_dir=checkpoints/pi0.5 --port=8000
set -euo pipefail

CONDA_ENV="openpi"
OPENPI_DIR="$HOME/Documents/openpi"
ISAACLAB_ENV="env_isaaclab"   # adjust if your Isaac Lab env has a different name

# ── 1. Resolve conda ────────────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null)" || {
    echo "ERROR: conda not found. Install Miniconda first."
    exit 1
}
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 2. Create openpi env ─────────────────────────────────────────────────────
if conda env list | grep -qE "^${CONDA_ENV}\s"; then
    echo "[openpi env] Already exists — skipping creation."
else
    echo "[openpi env] Creating conda env '${CONDA_ENV}' with Python 3.11 ..."
    conda create -n "$CONDA_ENV" python=3.11 -y
fi
conda activate "$CONDA_ENV"

# ── 3. Install uv inside the conda env ──────────────────────────────────────
# pip cannot resolve openpi's complex dependency graph (resolution-too-deep).
# uv's resolver handles it without issues and works fine inside conda envs.
echo "[openpi env] Installing uv ..."
pip install uv --quiet

# ── 4. Clone openpi ──────────────────────────────────────────────────────────
if [ ! -d "$OPENPI_DIR" ]; then
    echo "[openpi env] Cloning Physical-Intelligence/openpi ..."
    cd "$HOME/Documents"
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
        https://github.com/Physical-Intelligence/openpi.git
else
    echo "[openpi env] openpi already cloned at $OPENPI_DIR — skipping."
fi

# ── 5. Install openpi + JAX via uv ───────────────────────────────────────────
# UV_PROJECT_ENVIRONMENT points uv at the active conda env so it installs
# into the same site-packages that `python` uses in this shell.
echo "[openpi env] Installing openpi package and JAX 0.5.3+CUDA12 via uv ..."
cd "$OPENPI_DIR"
UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" \
    GIT_LFS_SKIP_SMUDGE=1 \
    uv pip install --python "$CONDA_PREFIX/bin/python" \
        "jax[cuda12]==0.5.3" \
        -e "."

# ── 6. Download pi0.5 checkpoint from Google Cloud Storage ──────────────────
# Checkpoints are hosted at gs://openpi-assets/checkpoints/, NOT on HuggingFace.
# openpi.shared.download uses gsutil (or fsspec fallback) and caches to
# ~/.cache/openpi by default. OPENPI_DATA_HOME overrides the cache root.
echo "[openpi env] Downloading pi0.5 base checkpoint from GCS (~8 GB) ..."
python - <<'PYEOF'
from openpi.shared import download
path = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
print(f"Checkpoint cached at: {path}")
PYEOF

# ── 7. Install lightweight openpi-client in the Isaac Lab env ────────────────
# world.py imports openpi_client to talk to the server over WebSocket.
# It does NOT need JAX — only this thin client package.
if conda env list | grep -qE "^${ISAACLAB_ENV}\s"; then
    echo "[isaaclab env] Installing openpi-client in '${ISAACLAB_ENV}' ..."
    conda run -n "$ISAACLAB_ENV" pip install --quiet \
        "git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
else
    echo "[isaaclab env] WARNING: conda env '${ISAACLAB_ENV}' not found."
    echo "  Run manually after activating your Isaac Lab env:"
    echo "    pip install 'git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client'"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Setup complete!"
echo ""
echo "Terminal 1 — start pi0.5 server (openpi env):"
echo "  conda activate $CONDA_ENV"
echo "  cd $OPENPI_DIR"
echo "  python scripts/serve_policy.py --env droid policy:default --port 8000"
echo ""
echo "Terminal 2 — run Isaac Lab push world (isaaclab env):"
echo "  conda activate $ISAACLAB_ENV"
echo "  cd /home/paolo/Documents/isaaclabmpc"
echo "  python examples/ur16e_push_stick/world.py"
echo "======================================================================"
