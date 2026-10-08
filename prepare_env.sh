#!/usr/bin/env bash
# Fast environment setup with uv (https://docs.astral.sh/uv/).
#
# Usage:
#   bash prepare_env.sh               # create .venv and install requeriments.txt
#   bash prepare_env.sh --flash-attn  # also build flash-attn (slow, needs CUDA toolkit)
#
# Environment variables:
#   TORCH_BACKEND   torch build to install: auto (detect from the NVIDIA driver,
#                   default), cu128, cu126, cpu, ...
#   VENV_DIR        virtual environment path (default: .venv)
#   PYTHON_VERSION  Python version for the environment (default: 3.12)
set -euo pipefail

cd "$(dirname "$0")"

TORCH_BACKEND="${TORCH_BACKEND:-auto}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

INSTALL_FLASH_ATTN=false
for arg in "$@"; do
    case "$arg" in
        --flash-attn) INSTALL_FLASH_ATTN=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" -r requeriments.txt --torch-backend "$TORCH_BACKEND"

if [ "$INSTALL_FLASH_ATTN" = true ]; then
    uv pip install --python "$VENV_DIR/bin/python" setuptools wheel ninja packaging psutil
    uv pip install --python "$VENV_DIR/bin/python" flash-attn --no-build-isolation
fi

"$VENV_DIR/bin/python" -c "import torch; print(f'torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
echo "Done. Activate with: source $VENV_DIR/bin/activate"
