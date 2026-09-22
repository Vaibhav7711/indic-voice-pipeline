"""Inference engines that serve the same model faster than eager PyTorch.

Each must match the explicit runner on the same audio (checked in
``scripts/gpu_validation.py``). Imported lazily: none is required to run
the reference path.
"""
