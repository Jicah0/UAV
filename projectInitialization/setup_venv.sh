#!/usr/bin/env bash
#
# setup_venv.sh
# =============
# Creates a Python virtual environment in this project directory and
# installs the packages listed in requirements.txt.
#
# Usage:
#   chmod +x setup_venv.sh      # if the executable bit didn't survive transfer
#   ./setup_venv.sh
#
# Run this ON THE RASPBERRY PI, not your dev machine - venvs (and the
# compiled packages inside them) don't transfer across CPU architectures.

set -u   # error on unset variables (NOT using -e - failures are handled explicitly below)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
PYTHON_BIN="python3"

echo "=== Raspberry Pi venv setup ==="
echo "Project directory: $SCRIPT_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Sanity checks
# ---------------------------------------------------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: python3 not found on this system."
    exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: requirements.txt not found at $REQUIREMENTS_FILE"
    echo "Place this script in the same directory as requirements.txt, or edit"
    echo "REQUIREMENTS_FILE at the top of this script."
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create (or reuse) the venv
# ---------------------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "A venv already exists at $VENV_DIR"
    read -r -p "Delete and recreate it from scratch? [y/N] " REPLY
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"

    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "ERROR: Virtual environment creation failed."
        exit 1
    fi
fi
echo

# ---------------------------------------------------------------------------
# 3. Install requirements into the venv
# ---------------------------------------------------------------------------
echo "Installing dependencies from requirements.txt ..."
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

if ! pip install -r "$REQUIREMENTS_FILE"; then
    echo
    echo "ERROR: One or more packages failed to install. See output above."
    deactivate
    exit 1
fi

deactivate
echo
echo "=== Setup complete ==="
echo "Virtual environment ready at: $VENV_DIR"
echo "Activate it with:"
echo "    source $VENV_DIR/bin/activate"