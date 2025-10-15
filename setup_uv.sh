#!/bin/bash

# Setup script to create a uv-based environment for SGFM
# This replaces the conda-based setup

set -e  # Exit on any error

# Create .env file with PROJECT_ROOT variable
PROJECT_ROOT="$(pwd)"
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

# Create and activate virtual environment
echo "📦 Creating virtual environment with Python 3.9..."
uv venv --python 3.9 .venv

# Activate the virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Suppress NetworkX backend warnings
export PYTHONWARNINGS="ignore::RuntimeWarning:networkx.utils.backends"

# Install PyTorch with CUDA support first (similar to conda channels)
echo "🔥 Installing PyTorch with CUDA 11.8 support..."
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
echo "🌐 Installing PyTorch Geometric..."
uv pip install torch-geometric==2.5.2

# Install PyTorch extensions
echo "⚡ Installing PyTorch extensions..."
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install the project in editable mode (this will install other dependencies from pyproject.toml)
echo "📋 Installing project dependencies..."
uv pip install -e .

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "   source .venv/bin/activate"
echo ""
echo "To deactivate the environment, run:"
echo "   deactivate"