# CUDA Inference Benchmark

This directory contains the PyTorch/CUDA inference script used for the
Ubuntu benchmark in the 6D Pose Estimation project.

The implementation uses standard PyTorch/torchvision with CUDA.
No custom CUDA kernels are used.

The script is provided mainly to show how inference and timing were
implemented.

- `run_pipeline_real_timed.py` — inference pipeline with timing measurements
