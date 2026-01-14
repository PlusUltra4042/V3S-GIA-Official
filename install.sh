#!/bin/bash

echo "=========================================================="
echo "   Setting up environment for V3S_GIA (CCS Submission)    "
echo "=========================================================="

# 1. Create Conda environment 'V3S_GIA' with Python 3.10
echo "[1/3] Creating Conda environment 'V3S_GIA'..."
conda create -n V3S_GIA python=3.10 -y

# 2. Activate environment
# Note: 'source activate' is safer for scripts than 'conda activate'
source activate V3S_GIA || conda activate V3S_GIA

# 3. Install PyTorch 2.1.0 with CUDA 12.1 support
# Critical: Must specify index-url for cu121 to match your A800 GPU
echo "[2/3] Installing PyTorch 2.1.0 (CUDA 12.1)..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies from requirements.txt
echo "[3/3] Installing dependencies (diffusers, lpips, etc.)..."
pip install -r requirements.txt

echo "=========================================================="
echo "   Setup Finished! Please run: conda activate V3S_GIA     "
echo "=========================================================="