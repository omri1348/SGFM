#!/bin/bash

# Setup script to create a uv-based environment for SGFM
# This replaces the conda-based setup

set -e  # Exit on any error

# Create .env file with PROJECT_ROOT variable
PROJECT_ROOT="$(dirname "$0")"
echo "🌍 Setting PROJECT_ROOT in .env to $PROJECT_ROOT"
echo "PROJECT_ROOT=\"$PROJECT_ROOT\"" > .env
# Create directories if they do not exist
mkdir -p $PROJECT_ROOT/hydra
mkdir -p $PROJECT_ROOT/wandb

echo "🚀 Setting up SGFM with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Sync uv
uv sync

# explain how to use environment
echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source .venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "   deactivate"
