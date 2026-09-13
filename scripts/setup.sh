#!/usr/bin/env bash
set -euo pipefail

echo "Installing dependencies (PyTorch is provided by Colab)..."
pip install --upgrade pip
pip install \
    transformers>=4.51 \
    accelerate>=1.4 \
    peft>=0.11 \
    datasets>=2.19 \
    evaluate>=0.4 \
    jiwer>=3.0 \
    librosa>=0.10 \
    soundfile>=0.12 \
    edge-tts>=6.1 \
    gradio>=4.0 \
    pytest>=8.0

echo "Setup complete."
